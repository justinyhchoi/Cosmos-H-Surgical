#!/bin/bash
# CholecTrack20 YOLOv7 Evaluation Pipeline
# Runs after training completion to generate predictions and evaluate with DetEval

set -e

YOLOV7_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODEL_WEIGHTS="${1:-${YOLOV7_ROOT}/runs/train/cholectrack20_e6e_surgitrack/weights/best.pt}"
MODEL_CFG="${2:-${YOLOV7_ROOT}/cfg/training/yolov7-e6e.yaml}"
GT_ROOT="/raid/cholectrack20"
PRED_ROOT="/raid/cholectrack20_predictions_yolov7"
CONDA_ENV="yolov7"

echo "============================================"
echo "CholecTrack20 YOLOv7 Evaluation Pipeline"
echo "============================================"
echo "Model:         $MODEL_WEIGHTS"
echo "YOLOv7 Root:   $YOLOV7_ROOT"
echo "Model cfg:     $MODEL_CFG"
echo "GT Root:       $GT_ROOT"
echo "Predictions:   $PRED_ROOT"
echo "============================================"

# Step 1: Verify model exists
if [[ ! -f "$MODEL_WEIGHTS" ]]; then
    echo "ERROR: Model weights not found at $MODEL_WEIGHTS"
    exit 1
fi
echo "✓ Model weights found ($(stat -c%s "$MODEL_WEIGHTS" | numfmt --to=iec-i --suffix=B))"

# Step 2: Create prediction directory
mkdir -p "$PRED_ROOT"
echo "✓ Prediction directory: $PRED_ROOT"

# Step 3: Generate validation predictions using direct YOLOv7 inference
echo ""
echo "Step 1: Generating validation predictions..."
cd "$YOLOV7_ROOT"
export MODEL_WEIGHTS MODEL_CFG PRED_ROOT GT_ROOT

python3 << 'PYTHON_SCRIPT'
import torch
import os
import json
import numpy as np
from pathlib import Path
from collections import defaultdict
import sys
from PIL import Image

# Configuration
MODEL_PATH = os.environ["MODEL_WEIGHTS"]
MODEL_CFG = os.environ["MODEL_CFG"]
VAL_DIR = Path("/raid/cholectrack20_yolo/images/val")
GT_ROOT = Path(os.environ["GT_ROOT"])
PRED_ROOT = Path(os.environ["PRED_ROOT"])
CONF_THRESH = 0.5
IOU_THRESH = 0.3

print(f"Model: {MODEL_PATH}")
print(f"Val images: {VAL_DIR}")
print(f"Predictions: {PRED_ROOT}")
print(f"NMS IoU: {IOU_THRESH}")

# Load YOLOv7 model
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(f"\nLoading model on {device}...")
from models.yolo import Model
from utils.general import non_max_suppression
model = Model(MODEL_CFG).to(device)
ckpt = torch.load(MODEL_PATH, map_location=device, weights_only=False)
model.load_state_dict(ckpt['model'].state_dict() if 'model' in ckpt else ckpt, strict=False)
model.eval()

# Get validation videos (from GT)
val_videos = sorted([d.name for d in (GT_ROOT / 'Validation').iterdir() if d.is_dir()])
print(f"Found {len(val_videos)} validation videos: {val_videos}")

# Run inference on each validation video
total_predictions = 0
for vid in val_videos:
    vid_dir = VAL_DIR / vid
    if not vid_dir.exists():
        print(f"  ⊘ {vid}: directory not found, skipping")
        continue
    
    images = sorted(vid_dir.glob('*.png'))
    if not images:
        print(f"  ⊘ {vid}: no images found")
        continue
    
    print(f"\n  Processing {vid} ({len(images)} images)...")
    
    # Get GT to extract video dimensions
    gt_file = GT_ROOT / 'Validation' / vid / f'{vid}.json'
    with open(gt_file) as f:
        gt_data = json.load(f)
    
    img_w = gt_data['video']['width']
    img_h = gt_data['video']['height']
    
    # Collect predictions
    predictions = []
    for i, img_path in enumerate(images):
        if (i + 1) % 200 == 0:
            print(f"    [{i+1}/{len(images)}]", end=' ', flush=True)
        
        frame_id = int(img_path.stem)
        
        # Inference
        with torch.no_grad():
            img = torch.from_numpy(np.array(Image.open(img_path).convert('RGB'))).float().to(device) / 255.0
            if len(img.shape) == 2:
                img = img.unsqueeze(-1).repeat(1, 1, 3)
            elif img.shape[2] == 4:
                img = img[..., :3]
            
            # Resize to model input
            orig_shape = img.shape[:2]
            img = img.permute(2, 0, 1).unsqueeze(0)  # CHW -> BCHW
            
            # Simple letterbox
            imgsz = 640
            ratio = imgsz / max(img.shape[2:])
            new_shape = (int(img.shape[2] * ratio), int(img.shape[3] * ratio))
            img_resized = torch.nn.functional.interpolate(img, size=new_shape, mode='bilinear', align_corners=False)
            
            # Pad
            pad_h = imgsz - new_shape[0]
            pad_w = imgsz - new_shape[1]
            img_padded = torch.nn.functional.pad(img_resized, (0, pad_w, 0, pad_h), value=114/255)
            
            # Predict
            pred = model(img_padded)[0]
            pred = non_max_suppression(pred, conf_thres=CONF_THRESH, iou_thres=IOU_THRESH)[0]
            
            if pred is not None and len(pred):
                # Convert NMS xyxy predictions from resized model coordinates to original image space.
                # Padding is only applied on the right/bottom, so there is no top-left offset to subtract.
                x1 = (pred[:, 0] / ratio).clamp(0, img_w)
                y1 = (pred[:, 1] / ratio).clamp(0, img_h)
                x2 = (pred[:, 2] / ratio).clamp(0, img_w)
                y2 = (pred[:, 3] / ratio).clamp(0, img_h)
                conf = pred[:, 4]
                cls_id = pred[:, 5].long()
                
                # Format: frame_id,category,x1,y1,x2,y2,score
                for j in range(len(pred)):
                    predictions.append(f"{frame_id},{cls_id[j].item()},{x1[j].item():.2f},{y1[j].item():.2f},{x2[j].item():.2f},{y2[j].item():.2f},{conf[j].item():.4f}")
    
    print()
    
    # Write predictions
    pred_file = PRED_ROOT / f"{vid}.txt"
    with open(pred_file, 'w') as f:
        f.write('\n'.join(predictions))
    
    print(f"    Wrote {len(predictions)} predictions to {pred_file.name}")
    total_predictions += len(predictions)

print(f"\n✓ Generated {total_predictions} total predictions")
print(f"✓ Saved to {PRED_ROOT}")

PYTHON_SCRIPT

# Step 4: Run DetEval evaluation
echo ""
echo "Step 2: Running DetEval evaluation..."
cd "$GT_ROOT/DetEval"

python3 eval.py \
    --GT_ROOT "$GT_ROOT" \
    --DET_ROOT "$PRED_ROOT" \
    --SPLIT validation \
    --MODEL_NAME "YOLOv7_CholecTrack20" \
    --CONDITION each

# Step 5: Summary
echo ""
echo "============================================"
echo "✓ Evaluation Complete!"
echo "============================================"
echo "Results saved to:"
echo "  - Predictions: $PRED_ROOT"
echo "  - Logs: ${MODEL_WEIGHTS%/weights/best.pt}/"
echo ""
echo "Next steps:"
echo "1. Check DetEval output for per-class and per-condition metrics"
echo "2. Compare results to YOLOv7 baseline (AP50:95=56.1%, AP50=80.6%, AP75=62.0%)"
echo "3. Analyze challenging conditions: occlusion (80.3%), bleeding (50.3%), smoke, blur"

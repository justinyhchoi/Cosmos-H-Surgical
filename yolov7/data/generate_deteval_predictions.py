#!/usr/bin/env python3
"""
Generate CholecTrack20 DetEval format predictions from YOLOv7 model.
Output format (CSV): frame_id, category, x1, y1, x2, y2, score
"""

import torch
import json
import numpy as np
from pathlib import Path
from collections import defaultdict


def run_inference(model_path, image_dir, conf_thresh=0.5):
    """Run YOLOv7 inference on directory of images."""
    # Load model
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"Loading model from {model_path} on {device}...")
    model = torch.hub.load('ultralytics/yolov7', 'custom', path=model_path, force_reload=True)
    model = model.to(device)
    model.conf = conf_thresh
    model.iou = 0.65
    
    # Get all images
    image_dir = Path(image_dir)
    images = sorted(image_dir.glob('*.png'))
    print(f"Found {len(images)} images in {image_dir}")
    
    # Run inference
    results = defaultdict(list)
    for i, img_path in enumerate(images):
        if (i + 1) % 100 == 0:
            print(f"  Processed {i+1}/{len(images)}")
        
        frame_id = int(img_path.stem)
        result = model(str(img_path), size=640)
        
        # Parse detections
        preds = result.xyxy[0].cpu().numpy()  # [x1, y1, x2, y2, conf, cls]
        for pred in preds:
            x1, y1, x2, y2, conf, cls_id = pred
            results[frame_id].append({
                'x1': float(x1),
                'y1': float(y1),
                'x2': float(x2),
                'y2': float(y2),
                'conf': float(conf),
                'cls': int(cls_id)
            })
    
    return results


def write_deteval_format(results, image_info, output_file):
    """Write predictions in DetEval CSV format.
    
    Format: frame_id, category, x1, y1, x2, y2, score
    where coordinates are in pixel space [0, width] x [0, height]
    """
    img_width = image_info.get('width', 854)
    img_height = image_info.get('height', 480)
    
    lines = []
    for frame_id in sorted(results.keys()):
        for det in results[frame_id]:
            x1 = det['x1']
            y1 = det['y1']
            x2 = det['x2']
            y2 = det['y2']
            score = det['conf']
            cls_id = det['cls']
            
            # frame_id, category, x1, y1, x2, y2, score
            lines.append(f"{frame_id},{cls_id},{x1:.2f},{y1:.2f},{x2:.2f},{y2:.2f},{score:.4f}")
    
    with open(output_file, 'w') as f:
        f.write('\n'.join(lines))
    
    print(f"Wrote {len(lines)} detections to {output_file}")


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', required=True, help='Path to trained YOLOv7 weights')
    parser.add_argument('--val_dir', default='/raid/cholectrack20_yolo/images/val', help='Validation images directory')
    parser.add_argument('--gt_root', default='/raid/cholectrack20', help='Ground truth root')
    parser.add_argument('--out_dir', default='/raid/cholectrack20_predictions', help='Output directory for predictions')
    parser.add_argument('--conf', type=float, default=0.5, help='Confidence threshold')
    args = parser.parse_args()
    
    # Create output directory
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Get validation videos
    val_videos = ['VID110', 'VID30']
    
    for vid in val_videos:
        print(f"\nProcessing {vid}...")
        vid_img_dir = Path(args.val_dir) / vid
        if not vid_img_dir.exists():
            print(f"  WARNING: {vid_img_dir} not found, skipping")
            continue
        
        # Load ground truth to get image dimensions
        gt_file = Path(args.gt_root) / 'Validation' / vid / f'{vid}.json'
        with open(gt_file) as f:
            gt_data = json.load(f)
        
        img_width = gt_data['video']['width']
        img_height = gt_data['video']['height']
        
        # Run inference
        results = run_inference(args.model_path, vid_img_dir, conf_thresh=args.conf)
        
        # Write predictions
        out_file = out_dir / f'{vid}.txt'
        write_deteval_format(results, gt_data['video'], out_file)
    
    print(f"\nPredictions saved to {out_dir}")
    print("Now run DetEval:")
    print(f"  python /raid/cholectrack20/DetEval/eval.py \\")
    print(f"    --GT_ROOT /raid/cholectrack20 \\")
    print(f"    --DET_ROOT {out_dir} \\")
    print(f"    --SPLIT validation \\")
    print(f"    --MODEL_NAME cholectrack20_yolov7")


if __name__ == '__main__':
    main()

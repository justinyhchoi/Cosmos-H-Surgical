#!/bin/bash
# Train YOLOv7-E6E on CholecTrack20 with 4x V100 DDP
# Usage: bash train_cholectrack_e6e.sh [--resume]
#
# E6E uses train_aux.py (auxiliary head losses: ComputeLossAuxOTA)
# Batch 16 total across 4 GPUs = 4 per GPU. This fits YOLOv7-E6E at 1280px on 32GB V100s.

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# -------- configurable --------
WEIGHTS="yolov7-e6e.pt"
CFG="cfg/training/yolov7-e6e.yaml"
DATA="data/cholectrack20.yaml"
HYP="data/hyp.surgitrack.yaml"
IMG_SIZE=1280
BATCH_SIZE=16         # total across 4 GPUs
EPOCHS=132
WORKERS=8
NAME="cholectrack20_e6e_surgitrack"
PROJECT="runs/train"
NGPUS=4
NMS_IOU=0.3
# ------------------------------

RESUME_FLAG=""
if [[ "$1" == "--resume" ]]; then
    # Find last run's last.pt
    LAST_PT=$(ls -t "${PROJECT}/${NAME}*/weights/last.pt" 2>/dev/null | head -1)
    if [[ -z "$LAST_PT" ]]; then
        echo "ERROR: No checkpoint found to resume from in ${PROJECT}/${NAME}*/"
        exit 1
    fi
    RESUME_FLAG="--resume $LAST_PT"
    echo "Resuming from: $LAST_PT"
fi

echo "================================================"
echo " YOLOv7-E6E CholecTrack20 Training"
echo "================================================"
echo "  Weights  : $WEIGHTS"
echo "  Config   : $CFG"
echo "  Data     : $DATA"
echo "  Hyp      : $HYP"
echo "  Img size : ${IMG_SIZE}x${IMG_SIZE}"
echo "  Batch    : $BATCH_SIZE (${NGPUS} GPUs = $((BATCH_SIZE / NGPUS)) per GPU)"
echo "  Epochs   : $EPOCHS"
echo "  Optimizer: Adam"
echo "  NMS IoU  : $NMS_IOU"
echo "  Name     : $NAME"
echo "================================================"

# Verify dataset exists
if [[ ! -f /raid/cholectrack20_yolo/train.txt ]]; then
    echo "ERROR: /raid/cholectrack20_yolo/train.txt not found."
    echo "Run data/cholectrack20_prepare_v2.py first to convert the dataset."
    exit 1
fi

python -m torch.distributed.run \
    --nproc_per_node=$NGPUS \
    --master_port=9999 \
    train_aux.py \
    --weights "$WEIGHTS" \
    --cfg "$CFG" \
    --data "$DATA" \
    --hyp "$HYP" \
    --img-size $IMG_SIZE $IMG_SIZE \
    --batch-size $BATCH_SIZE \
    --epochs $EPOCHS \
    --workers $WORKERS \
    --project "$PROJECT" \
    --name "$NAME" \
    --iou-thres $NMS_IOU \
    --adam \
    --sync-bn \
    --device 0,1,2,3 \
    $RESUME_FLAG

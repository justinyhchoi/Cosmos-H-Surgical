# CholecTrack20 YOLOv7-X Detector Model Card

## Model Summary

This model card documents the local YOLOv7-X detector checkpoint trained in this workspace for CholecTrack20 surgical tool detection. The best evaluated local checkpoint at the time of writing is:

```text
yolov7/runs/train/cholectrack20_yolov7x_coco_854/weights/best.pt
```
OR
```text
/raid/justinchoi/YOLO/cholectrack20_train_yolov7x_854p.pt
```
The checkpoint is a YOLOv7-X detector fine-tuned from the COCO-pretrained `yolov7x.pt` checkpoint on the converted CholecTrack20 YOLO-format dataset at `/raid/cholectrack20_yolo`.

## Dataset

- Dataset: CholecTrack20
- Local raw dataset: `/raid/cholectrack20`
- Local YOLO-format dataset: `/raid/cholectrack20_yolo`
- Dataset config: `data/cholectrack20.yaml`
- Classes: `grasper`, `bipolar`, `hook`, `scissors`, `clipper`, `irrigator`, `specimen-bag`
- Table 2-style evaluation split used here: CholecTrack20 test split, 13,367 images and 26,475 labeled tool boxes

The YOLO-format dataset was prepared with `data/cholectrack20_prepare_v2.py`, which converts normalized CholecTrack20 top-left boxes `[x, y, w, h]` into YOLO center-format labels `[class, cx, cy, w, h]`.

## Training Recipe

The strongest local evaluated checkpoint uses:

| Field | Value |
| :-- | :-- |
| Architecture | YOLOv7-X |
| Config | `cfg/training/yolov7x.yaml` |
| Initial weights | `yolov7x.pt` COCO-pretrained checkpoint |
| Dataset config | `data/cholectrack20.yaml` |
| Hyperparameters | `data/hyp.scratch.p5.yaml` |
| Requested image size | 854 x 854 |
| Effective training image size | 864 x 864, rounded to model stride |
| Total batch size | 32 |
| Per-GPU batch size | 8 |
| Epochs configured | 100 |
| Optimizer | SGD |
| Distributed training | 4x GPU DDP with SyncBatchNorm |
| Run directory | `runs/train/cholectrack20_yolov7x_coco_854` |

Training command used for the final run:

```bash
cd yolov7

CUDA_VISIBLE_DEVICES=0,1,2,3 \
NCCL_DEBUG=INFO \
TORCH_DISTRIBUTED_DEBUG=DETAIL \
python train_distributed.py \
  --nproc-per-node 4 \
  --data data/cholectrack20.yaml \
  --cfg cfg/training/yolov7x.yaml \
  --weights yolov7x.pt \
  --img-size 854 854 \
  --batch-size 32 \
  --epochs 100 \
  --name cholectrack20_yolov7x_coco_854 \
  --hyp data/hyp.scratch.p5.yaml \
  --workers 8 \
  --sync-bn
```

### Key Hyperparameters

| Field | Value |
| :-- | --: |
| `lr0` | 0.01 |
| `lrf` | 0.1 |
| `momentum` | 0.937 |
| `weight_decay` | 0.0005 |
| `warmup_epochs` | 3.0 |
| `box` | 0.05 |
| `cls` | 0.3 |
| `obj` | 0.7 |
| `mosaic` | 1.0 |
| `mixup` | 0.15 |
| `paste_in` | 0.15 |
| `translate` | 0.2 |
| `scale` | 0.9 |
| `fliplr` | 0.5 |

## Training Metrics

Best training epoch by mAP50:95:

| Epoch | Precision | Recall | mAP50 | mAP50:95 |
| :-- | --: | --: | --: | --: |
| 44/99 | 90.5 | 75.3 | 81.6 | 49.0 |

Best training epoch by mAP50:

| Epoch | Precision | Recall | mAP50 | mAP50:95 |
| :-- | --: | --: | --: | --: |
| 34/99 | 87.2 | 76.6 | 81.7 | 48.7 |

Final epoch metrics:

| Epoch | Precision | Recall | mAP50 | mAP50:95 |
| :-- | --: | --: | --: | --: |
| 99/99 | 87.8 | 77.3 | 80.8 | 48.8 |

## Evaluation

Table 2-style evaluation artifacts:

```text
runs/test/cholectrack20_table2_yolov7x_coco_854/predictions.json
runs/test/cholectrack20_table2_yolov7x_coco_854/table2_metrics.csv
runs/test/cholectrack20_table2_yolov7x_coco_854/table2_metrics.json
```

### Validation And Inference Commands

Set the checkpoint path once, then reuse it in the commands below:

```bash
cd yolov7
export YOLOV7X_CHOLECTRACK20_WEIGHTS=/raid/justinchoi/YOLO/cholectrack20_train_yolov7x_854p.pt
```

Single-GPU Table 2-style validation on the CholecTrack20 test split:

```bash
CUDA_VISIBLE_DEVICES=0 \
python data/cholectrack20_table2_eval.py \
  --weights "$YOLOV7X_CHOLECTRACK20_WEIGHTS" \
  --data data/cholectrack20.yaml \
  --split test \
  --img-size 854 \
  --batch-size 16 \
  --device 0 \
  --name cholectrack20_table2_yolov7x_coco_854_single_gpu
```

CUDA validation with all four GPUs visible. This script still runs the model on `cuda:0`; use this form mainly to keep the environment consistent and increase the batch size only if GPU 0 has enough memory:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 \
python data/cholectrack20_table2_eval.py \
  --weights "$YOLOV7X_CHOLECTRACK20_WEIGHTS" \
  --data data/cholectrack20.yaml \
  --split test \
  --img-size 854 \
  --batch-size 32 \
  --device 0 \
  --name cholectrack20_table2_yolov7x_coco_854_cuda0_batch32
```

Single-GPU inference on a finite image directory, image file, or video file:

```bash
CUDA_VISIBLE_DEVICES=0 \
python detect.py \
  --weights "$YOLOV7X_CHOLECTRACK20_WEIGHTS" \
  --source /raid/cholectrack20_yolo/images/test \
  --img-size 854 \
  --conf-thres 0.25 \
  --iou-thres 0.45 \
  --device 0 \
  --name cholectrack20_yolov7x_854_single_gpu_infer
```

Multi-GPU parallel inference on a finite image directory or video file:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 \
python detect_parallel.py \
  --weights "$YOLOV7X_CHOLECTRACK20_WEIGHTS" \
  --source /raid/cholectrack20_yolo/images/test \
  --gpus 0,1,2,3 \
  --img-size 854 \
  --conf-thres 0.25 \
  --iou-thres 0.45 \
  --name cholectrack20_yolov7x_854_4gpu_infer \
  --no-half
```

Use `--source /path/to/image_or_video_or_directory` to run inference on other finite sources. Live streams and webcams should use `detect.py`; `detect_parallel.py` is intended for finite image/video workloads.

### Overall Metrics

| Split | Images | Labels | AP50 | AP75 | AP50:95 |
| :-- | --: | --: | --: | --: | --: |
| CholecTrack20 test split, Table 2-style eval | 13,367 | 26,475 | 83.4 | 62.8 | 56.5 |

### Per-Class Metrics

| Class | Labels | AP50 | AP75 | AP50:95 |
| :-- | --: | --: | --: | --: |
| grasper | 14,661 | 93.4 | 77.0 | 66.7 |
| bipolar | 1,048 | 90.3 | 68.3 | 59.6 |
| hook | 7,905 | 95.9 | 75.2 | 67.7 |
| scissors | 202 | 88.3 | 73.1 | 64.7 |
| clipper | 567 | 93.9 | 78.6 | 66.2 |
| irrigator | 567 | 57.6 | 21.4 | 26.3 |
| specimen-bag | 1,525 | 64.1 | 46.3 | 44.3 |

### Visual Challenge Metrics

| Condition | Images | Labels | AP50 | AP75 | AP50:95 |
| :-- | --: | --: | --: | --: | --: |
| Bleeding | 7,471 | 14,388 | 81.5 | 61.0 | 54.8 |
| Blur | 127 | 218 | 65.8 | 29.0 | 33.6 |
| Smoke | 1,537 | 3,350 | 73.4 | 54.0 | 48.9 |
| Crowded | 3,538 | 10,950 | 82.8 | 61.3 | 54.9 |
| Occluded | 9,234 | 22,342 | 83.3 | 62.0 | 55.9 |
| Reflection | 34 | 52 | 64.3 | 45.7 | 40.4 |
| Foul Lens | 615 | 1,495 | 77.0 | 55.7 | 49.6 |
| Trocar | 334 | 682 | 53.2 | 37.8 | 35.0 |

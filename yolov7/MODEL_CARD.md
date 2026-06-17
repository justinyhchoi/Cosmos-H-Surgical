# CholecTrack20 YOLOv7-X Detector Model Card

## Model Summary

This model card documents the local YOLOv7-X detector checkpoint trained in this workspace for CholecTrack20 surgical tool detection. The best evaluated local checkpoint at the time of writing is:

```text
yolov7/runs/train/cholectrack20_yolov7x_coco_854/weights/best.pt
```

The checkpoint is a YOLOv7-X detector fine-tuned from the COCO-pretrained `yolov7x.pt` checkpoint on the converted CholecTrack20 YOLO-format dataset at `/raid/cholectrack20_yolo`.

## Intended Use

The model is intended for research experiments on surgical tool detection in CholecTrack20 frames. It can be used as a detector baseline for downstream tracking experiments, error analysis, and training-pipeline debugging.

It is not intended for clinical deployment, surgical decision support, or patient-facing use.

## Dataset

- Dataset: CholecTrack20
- Local raw dataset: `/raid/cholectrack20`
- Local YOLO-format dataset: `/raid/cholectrack20_yolo`
- Dataset config: `data/cholectrack20.yaml`
- Classes: `grasper`, `bipolar`, `hook`, `scissors`, `clipper`, `irrigator`, `specimen-bag`
- Table 2-style evaluation set used here: 13,367 images and 26,475 labeled tool boxes

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

### Overall Metrics

| Split | Images | Labels | AP50 | AP75 | AP50:95 |
| :-- | --: | --: | --: | --: | --: |
| CholecTrack20 Table 2-style eval | 13,367 | 26,475 | 83.4 | 62.8 | 56.5 |

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

## Comparison To Published CholecTrack20 Detector Benchmarks

The CholecTrack20 paper reports the following YOLO detector benchmark values in Table 2:

| Model | AP50 | AP75 | AP50:95 | FPS |
| :-- | --: | --: | --: | --: |
| YOLOv7 | 80.6 | 62.0 | 56.1 | 20.6 |
| YOLOv8 | 79.1 | 62.4 | 55.6 | 29.0 |
| YOLOv9 | 80.2 | 62.6 | 56.5 | 23.7 |
| YOLOv10 | 80.1 | 62.1 | 55.8 | 28.6 |
| Local YOLOv7-X | 83.4 | 62.8 | 56.5 | Not measured |

The local YOLOv7-X checkpoint matches or exceeds the published AP values in this local Table 2-style evaluation, but it should not be described as an official benchmark reproduction unless the evaluation protocol, split construction, preprocessing, and reporting environment are independently verified.

## Known Gaps

- FPS was not measured for the final YOLOv7-X checkpoint.
- The evaluation script computes Table 2-style detector metrics but is not an official CholecTrack20 benchmark submission.
- The local evaluation depends on the converted YOLO dataset and generated COCO-style eval files in this workspace.
- The final model uses COCO-pretrained `yolov7x.pt`, while SurgiTrack states its YOLOv7 detector was pretrained on MOT20 and CrowdHuman before CholecTrack20 fine-tuning.
- YOLOv7-X has a larger compute and memory footprint than the earlier YOLOv7 p5 baseline.

## Licenses And Data Use

Follow the licenses for YOLOv7, CholecTrack20, Cosmos-H-Surgical, and any checkpoint weights used. CholecTrack20 is a research dataset with non-clinical-use constraints; verify the dataset license before redistributing trained weights or predictions.

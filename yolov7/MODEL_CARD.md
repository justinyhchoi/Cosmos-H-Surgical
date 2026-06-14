# CholecTrack20 YOLOv7 Detector Model Card

## Model Summary

This model card documents the local YOLOv7 detector checkpoints trained in this workspace for CholecTrack20 surgical tool detection. The best evaluated local checkpoint at the time of writing is:

```text
yolov7/runs/train/cholectrack20_yolov7_full/weights/best.pt
```

The checkpoint is a YOLOv7 detector fine-tuned from COCO-pretrained `yolov7.pt` on the converted CholecTrack20 YOLO-format dataset at `/raid/cholectrack20_yolo`.

## Intended Use

The model is intended for research experiments on surgical tool detection in CholecTrack20 frames. It can be used as a detector baseline for downstream tracking experiments, error analysis, and training-pipeline debugging.

It is not intended for clinical deployment, surgical decision support, or patient-facing use.

## Dataset

- Dataset: CholecTrack20
- Local raw dataset: `/raid/cholectrack20`
- Local YOLO-format dataset: `/raid/cholectrack20_yolo`
- Classes: `grasper`, `bipolar`, `hook`, `scissors`, `clipper`, `irrigator`, `specimen-bag`
- Local validation labels used in this evaluation: 4,106 labeled tool boxes across 2,461 labeled validation images

## Training Recipe

The strongest local evaluated checkpoint uses:

| Field | Value |
| :-- | :-- |
| Architecture | YOLOv7 |
| Config | `cfg/training/yolov7.yaml` |
| Initial weights | `yolov7.pt` COCO-pretrained checkpoint |
| Dataset config | `data/cholectrack20.yaml` |
| Hyperparameters | `data/hyp.scratch.p5.yaml` |
| Image size | 640 x 640 |
| Batch size | 16 |
| Epochs configured | 50 |
| Optimizer | SGD |
| Device | single GPU |

A separate YOLOv7-E6E launcher, `train_cholectrack_e6e.sh`, is provided for 4xV100 training with SurgiTrack-aligned optimizer settings where supported by this YOLOv7 codebase.

## Evaluation

Evaluation command:

```bash
cd yolov7
python data/cholectrack20_table2_eval.py \
  --weights runs/train/cholectrack20_yolov7_full/weights/best.pt \
  --data data/cholectrack20.yaml \
  --split val \
  --img-size 640 \
  --batch-size 32 \
  --device 0 \
  --name cholectrack20_table2_yolov7_full
```

Generated files:

```text
runs/test/cholectrack20_table2_yolov7_full/predictions.json
runs/test/cholectrack20_table2_yolov7_full/table2_metrics.csv
runs/test/cholectrack20_table2_yolov7_full/table2_metrics.json
```

### Validation Metrics

| Split | AP50 | AP75 | AP50:95 |
| :-- | --: | --: | --: |
| CholecTrack20 validation | 44.6 | 34.3 | 29.6 |

### Per-Class AP50

| Class | AP50 |
| :-- | --: |
| grasper | 43.6 |
| bipolar | 50.9 |
| hook | 48.5 |
| scissors | 39.3 |
| clipper | 65.6 |
| irrigator | 4.5 |
| specimen-bag | 59.7 |

### Visual Challenge AP50

| Condition | AP50 |
| :-- | --: |
| Bleeding | 19.0 |
| Blur | 100.0 |
| Smoke | 44.0 |
| Crowded | 23.5 |
| Occluded | 51.6 |
| Foul Lens | 13.3 |
| Trocar | 27.0 |

The blur subset contains only 2 validation images and 3 labels, so that value is not stable.

## Comparison To Published CholecTrack20 Detector Benchmarks

The CholecTrack20 paper reports the following YOLO detector benchmark values in Table 2:

| Model | AP50 | AP75 | AP50:95 | FPS |
| :-- | --: | --: | --: | --: |
| YOLOv7 | 80.6 | 62.0 | 56.1 | 20.6 |
| YOLOv8 | 79.1 | 62.4 | 55.6 | 29.0 |
| YOLOv9 | 80.2 | 62.6 | 56.5 | 23.7 |
| YOLOv10 | 80.1 | 62.1 | 55.8 | 28.6 |

The local checkpoint is far below those values and should not be described as reproducing the paper.

## Known Gaps

- The strongest local evaluated checkpoint is COCO-pretrained, while SurgiTrack states its YOLOv7 detector was pretrained on MOT20 and CrowdHuman before CholecTrack20 fine-tuning.
- The local result is measured on validation data, not the official held-out paper test split.
- The local YOLOv7-E6E experiment uses a different architecture scale than the 123M-parameter detector reported in SurgiTrack.
- The provided SurgiTrack-aligned hyperparameter file does not implement the paper's Plateau learning-rate scheduler.
- The evaluation script computes Table 2-style detector metrics but is not an official CholecTrack20 benchmark submission.

## Licenses And Data Use

Follow the licenses for YOLOv7, CholecTrack20, Cosmos-H-Surgical, and any checkpoint weights used. CholecTrack20 is a research dataset with non-clinical-use constraints; verify the dataset license before redistributing trained weights or predictions.

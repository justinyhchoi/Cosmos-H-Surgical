#!/usr/bin/env python3
"""Evaluate a YOLOv7 checkpoint with CholecTrack20 Table 2-style metrics."""

import argparse
import json
import sys
import tempfile
from pathlib import Path

FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
import yaml
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from tqdm import tqdm

from models.experimental import attempt_load
from utils.datasets import create_dataloader
from utils.general import check_dataset, check_img_size, colorstr, non_max_suppression, scale_coords, xyxy2xywh
from utils.torch_utils import select_device


TOOL_NAMES = ['grasper', 'bipolar', 'hook', 'scissors', 'clipper', 'irrigator', 'specimen-bag']
CONDITIONS = {
    'Bleeding': 'bleeding',
    'Blur': 'blurred',
    'Smoke': 'smoke',
    'Crowded': 'crowded',
    'Occluded': 'occluded',
    'Reflection': 'reflection',
    'Foul Lens': 'stainedlens',
    'Trocar': 'undercoverage',
}

def annotation_json_path(dataset_root, raw_root, raw_split, video_id):
    override_path = Path(dataset_root) / 'overrides' / raw_split / video_id / f'{video_id}.json'
    if override_path.exists():
        return override_path
    return Path(raw_root) / raw_split / video_id / f'{video_id}.json'


def normalize_bbox(bbox, image_width, image_height):
    """Return [x, y, w, h] normalized to image size."""
    x, y, box_w, box_h = bbox
    if max(abs(x), abs(y), abs(box_w), abs(box_h)) > 2.0:
        x /= image_width
        y /= image_height
        box_w /= image_width
        box_h /= image_height
    return x, y, box_w, box_h


def load_yaml(path):
    with open(path) as file:
        return yaml.safe_load(file)


def frame_key(path):
    image_path = Path(path)
    return image_path.parent.name, int(image_path.stem)


def build_coco_gt(dataset_root, raw_root, split, image_files):
    raw_split = {'val': 'Validation', 'test': 'Testing', 'train': 'Training'}[split]
    image_set = {frame_key(path) for path in image_files}
    images = []
    annotations = []
    categories = [{'id': index + 1, 'name': name} for index, name in enumerate(TOOL_NAMES)]
    image_lookup = {}
    ann_id = 1
    image_id = 1

    for video_dir in sorted((Path(raw_root) / raw_split).glob('VID*')):
        video_id = video_dir.name
        if (raw_split, video_id) in EXCLUDED_VIDEOS:
            continue
        json_path = annotation_json_path(dataset_root, raw_root, raw_split, video_id)
        if not json_path.exists():
            continue
        with open(json_path) as file:
            video_data = json.load(file)
        width = video_data['video']['width']
        height = video_data['video']['height']

        for frame_text, frame_annotations in sorted(video_data['annotations'].items(), key=lambda item: int(item[0])):
            key = (video_id, int(frame_text))
            if key not in image_set:
                continue
            image_lookup[key] = image_id
            images.append({
                'id': image_id,
                'file_name': str(Path(dataset_root) / 'images' / split / video_id / f'{int(frame_text):06d}.png'),
                'width': width,
                'height': height,
                'video_id': video_id,
                'frame_id': int(frame_text),
            })
            for ann in frame_annotations:
                x, y, box_w, box_h = normalize_bbox(ann['tool_bbox'], width, height)
                coco_ann = {
                    'id': ann_id,
                    'image_id': image_id,
                    'category_id': int(ann['instrument']) + 1,
                    'bbox': [x * width, y * height, box_w * width, box_h * height],
                    'area': box_w * width * box_h * height,
                    'iscrowd': int(ann.get('iscrowd', 0)),
                    'conditions': {name: int(ann.get(field, 0)) for name, field in CONDITIONS.items()},
                }
                annotations.append(coco_ann)
                ann_id += 1
            image_id += 1

    return {
        'info': {'description': f'CholecTrack20 {split} converted for COCOeval'},
        'images': images,
        'annotations': annotations,
        'categories': categories,
    }, image_lookup


def run_inference(weights, data, split, imgsz, batch_size, conf_thres, iou_thres, device_name, half_precision):
    device = select_device(device_name, batch_size=batch_size)
    model = attempt_load(weights, map_location=device)
    stride = max(int(model.stride.max()), 32)
    imgsz = check_img_size(imgsz, s=stride)
    if half_precision and device.type != 'cpu':
        model.half()
    model.eval()

    opt = argparse.Namespace(
        single_cls=False,
        augment=False,
        cache_images=False,
        rect=True,
        workers=8,
        quad=False,
        image_weights=False,
        rank=-1,
    )
    dataloader = create_dataloader(data[split], imgsz, batch_size, stride, opt, pad=0.5, rect=True,
                                   prefix=colorstr(f'{split}: '))[0]
    if device.type != 'cpu':
        model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))

    predictions = []
    image_files = []
    for img, targets, paths, shapes in tqdm(dataloader, desc='predict'):
        image_files.extend(paths)
        img = img.to(device, non_blocking=True)
        img = img.half() if half_precision and device.type != 'cpu' else img.float()
        img /= 255.0

        with torch.no_grad():
            out = model(img, augment=False)[0]
            out = non_max_suppression(out, conf_thres=conf_thres, iou_thres=iou_thres, labels=[], multi_label=True)

        for image_index, pred in enumerate(out):
            if pred is None or len(pred) == 0:
                continue
            pred_native = pred.clone()
            scale_coords(img[image_index].shape[1:], pred_native[:, :4], shapes[image_index][0], shapes[image_index][1])
            boxes = xyxy2xywh(pred_native[:, :4])
            boxes[:, :2] -= boxes[:, 2:] / 2
            video_id, frame_id = frame_key(paths[image_index])
            for row, box in zip(pred.tolist(), boxes.tolist()):
                predictions.append({
                    'video_id': video_id,
                    'frame_id': frame_id,
                    'category_id': int(row[5]) + 1,
                    'bbox': [round(float(value), 3) for value in box],
                    'score': round(float(row[4]), 5),
                })

    return predictions, image_files


def evaluate_subset(coco_gt, predictions, image_ids=None, category_ids=None):
    if image_ids is None:
        image_ids = [image['id'] for image in coco_gt.dataset['images']]
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json') as pred_file:
        json.dump(predictions, pred_file)
        pred_file.flush()
        coco_dt = coco_gt.loadRes(pred_file.name) if predictions else COCO()
    evaluator = COCOeval(coco_gt, coco_dt, 'bbox')
    evaluator.params.imgIds = sorted(image_ids)
    if category_ids is not None:
        evaluator.params.catIds = sorted(category_ids)
    evaluator.evaluate()
    evaluator.accumulate()
    evaluator.summarize()
    return evaluator.stats.copy()


def subset_coco(coco_data, image_ids):
    image_id_set = set(image_ids)
    return {
        'info': coco_data['info'],
        'images': [image for image in coco_data['images'] if image['id'] in image_id_set],
        'annotations': [ann for ann in coco_data['annotations'] if ann['image_id'] in image_id_set],
        'categories': coco_data['categories'],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', default='runs/train/cholectrack20_yolov7_full/weights/best.pt')
    parser.add_argument('--data', default='data/cholectrack20.yaml')
    parser.add_argument('--raw-root', default='/raid/cholectrack20')
    parser.add_argument('--split', choices=['train', 'val', 'test'], default='val')
    parser.add_argument('--img-size', type=int, default=640)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--conf-thres', type=float, default=0.001)
    parser.add_argument('--iou-thres', type=float, default=0.65)
    parser.add_argument('--device', default='0')
    parser.add_argument('--project', default='runs/test')
    parser.add_argument('--name', default='cholectrack20_table2')
    parser.add_argument('--no-half', action='store_true')
    args = parser.parse_args()

    data = load_yaml(args.data)
    check_dataset(data)
    save_dir = Path(args.project) / args.name
    save_dir.mkdir(parents=True, exist_ok=True)

    predictions_with_keys, image_files = run_inference(
        args.weights, data, args.split, args.img_size, args.batch_size, args.conf_thres,
        args.iou_thres, args.device, not args.no_half)
    coco_data, image_lookup = build_coco_gt(data['path'], args.raw_root, args.split, image_files)

    predictions = []
    for pred in predictions_with_keys:
        image_id = image_lookup.get((pred.pop('video_id'), pred.pop('frame_id')))
        if image_id is None:
            continue
        pred['image_id'] = image_id
        predictions.append(pred)

    gt_path = save_dir / 'cholectrack20_coco_gt.json'
    pred_path = save_dir / 'predictions.json'
    with open(gt_path, 'w') as file:
        json.dump(coco_data, file)
    with open(pred_path, 'w') as file:
        json.dump(predictions, file)

    coco_gt = COCO(str(gt_path))
    overall = evaluate_subset(coco_gt, predictions)
    rows = [{
        'section': 'overall',
        'name': 'all',
        'AP50': overall[1] * 100,
        'AP75': overall[2] * 100,
        'AP50_95': overall[0] * 100,
        'images': len(coco_data['images']),
        'labels': len(coco_data['annotations']),
    }]

    for category in coco_data['categories']:
        stats = evaluate_subset(coco_gt, predictions, category_ids=[category['id']])
        rows.append({
            'section': 'category',
            'name': category['name'],
            'AP50': stats[1] * 100,
            'AP75': stats[2] * 100,
            'AP50_95': stats[0] * 100,
            'images': len(coco_data['images']),
            'labels': sum(1 for ann in coco_data['annotations'] if ann['category_id'] == category['id']),
        })

    for condition in CONDITIONS:
        condition_image_ids = sorted({ann['image_id'] for ann in coco_data['annotations'] if ann['conditions'][condition]})
        if not condition_image_ids:
            continue
        condition_data = subset_coco(coco_data, condition_image_ids)
        condition_gt_path = save_dir / f"condition_{condition.lower().replace(' ', '_')}_gt.json"
        with open(condition_gt_path, 'w') as file:
            json.dump(condition_data, file)
        condition_gt = COCO(str(condition_gt_path))
        condition_predictions = [pred for pred in predictions if pred['image_id'] in set(condition_image_ids)]
        stats = evaluate_subset(condition_gt, condition_predictions)
        rows.append({
            'section': 'condition',
            'name': condition,
            'AP50': stats[1] * 100,
            'AP75': stats[2] * 100,
            'AP50_95': stats[0] * 100,
            'images': len(condition_data['images']),
            'labels': len(condition_data['annotations']),
        })

    metrics_path = save_dir / 'table2_metrics.json'
    csv_path = save_dir / 'table2_metrics.csv'
    with open(metrics_path, 'w') as file:
        json.dump(rows, file, indent=2)
    with open(csv_path, 'w') as file:
        file.write('section,name,AP50,AP75,AP50_95,images,labels\n')
        for row in rows:
            file.write('{section},{name},{AP50:.3f},{AP75:.3f},{AP50_95:.3f},{images},{labels}\n'.format(**row))

    print('\nCholecTrack20 Table 2-style metrics')
    print('section,name,AP50,AP75,AP50_95,images,labels')
    for row in rows:
        print('{section},{name},{AP50:.3f},{AP75:.3f},{AP50_95:.3f},{images},{labels}'.format(**row))
    print(f'\nSaved: {csv_path}')
    print(f'Saved: {metrics_path}')


if __name__ == '__main__':
    main()
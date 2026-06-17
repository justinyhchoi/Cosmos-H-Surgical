#!/usr/bin/env python3
"""
Convert CholecTrack20 dataset to YOLOv7 format (v2 - proper structure).

Annotation bbox in CholecTrack20 JSON: [x, y, w, h] (top-left corner format, COCO style).
The local release stores normalized values, while the README example shows pixel values.
YOLOv7 label format: <class> <x_center_norm> <y_center_norm> <w_norm> <h_norm>

Output structure:
  /raid/cholectrack20_yolo/images/train/VID*/*.png
  /raid/cholectrack20_yolo/labels/train/VID*/*.txt
  /raid/cholectrack20_yolo/train.txt -> list of image paths

Classes (0-indexed):
  0: grasper, 1: bipolar, 2: hook, 3: scissors, 4: clipper, 5: irrigator, 6: specimen-bag
"""

import os
import json
import glob
import argparse
import shutil
from pathlib import Path


CLASSES = ['grasper', 'bipolar', 'hook', 'scissors', 'clipper', 'irrigator', 'specimen-bag']
INSTRUMENT_ID_TO_CLASS = {i: i for i in range(7)}


def annotation_json_path(dataset_root, split_name, vid, out_root):
    override_path = Path(out_root) / 'overrides' / split_name / vid / f'{vid}.json'
    if override_path.exists():
        return override_path
    return Path(dataset_root) / split_name / vid / f'{vid}.json'


def yolo_split_name(split_name):
    return {'Training': 'train', 'Validation': 'val', 'Testing': 'test'}[split_name]


def extracted_frames_dir(out_root, split_name, vid):
    return Path(out_root) / 'extracted_frames' / yolo_split_name(split_name) / vid


def normalize_bbox(bbox, image_width, image_height):
    """Return [x, y, w, h] normalized to image size.

    CholecTrack20 annotations in the local copy are already normalized, but the
    public README uses pixel-valued examples. Pixel boxes are much larger than 1,
    while normalized boxes may slightly exceed [0, 1] at image boundaries.
    """
    x, y, w, h = bbox
    if max(abs(x), abs(y), abs(w), abs(h)) > 2.0:
        x /= image_width
        y /= image_height
        w /= image_width
        h /= image_height
    return x, y, w, h


def extract_annotated_frames(vid_dir, json_file, frames_dir):
    """Extract annotated frames from a split MP4 and resize to annotation size."""
    try:
        import cv2
    except Exception as exc:
        print(f"  WARNING: OpenCV unavailable, skipping frame extraction: {exc}")
        return False

    def needs_extraction(frame_path, width, height):
        if not frame_path.exists():
            return True
        image = cv2.imread(str(frame_path))
        return image is None or image.shape[1] != width or image.shape[0] != height

    mp4_files = list(Path(vid_dir).glob('*.mp4'))
    if not mp4_files:
        return False

    data = json.load(open(json_file))
    ann_frame_ids = set(int(k) for k in data['annotations'].keys())
    image_width = data['video']['width']
    image_height = data['video']['height']
    frames_dir.mkdir(parents=True, exist_ok=True)

    missing_or_bad = [
        fid for fid in sorted(ann_frame_ids)
        if needs_extraction(frames_dir / f'{fid:06d}.png', image_width, image_height)
    ]
    if not missing_or_bad:
        print(f"  {Path(vid_dir).name}: annotated frames already extracted ({len(ann_frame_ids)} frames)")
        return True

    print(f"  Extracting {Path(vid_dir).name}: {len(missing_or_bad)} missing or wrong-size annotated frames...")
    capture = cv2.VideoCapture(str(mp4_files[0]))
    if not capture.isOpened():
        print(f"    ERROR: Could not open {mp4_files[0]}")
        return False

    target_frames = set(missing_or_bad)
    last_target_frame = max(target_frames)
    saved = 0
    failed = 0
    frame_index = 0
    while frame_index < last_target_frame:
        ok, frame = capture.read()
        frame_index += 1
        if frame_index not in target_frames:
            continue
        if not ok or frame is None:
            failed += 1
            continue
        if frame.shape[1] != image_width or frame.shape[0] != image_height:
            frame = cv2.resize(frame, (image_width, image_height), interpolation=cv2.INTER_AREA)
        if cv2.imwrite(str(frames_dir / f'{frame_index:06d}.png'), frame):
            saved += 1
        else:
            failed += 1
    capture.release()
    print(f"    Saved {saved}/{len(missing_or_bad)} frames" + (f", failed {failed}" if failed else ""))
    return failed == 0


def convert_split(dataset_root, split_name, out_root):
    """Convert one split (Training/Validation/Testing) to YOLO format with proper directory structure."""
    split_dir = Path(dataset_root) / split_name
    img_list_lines = []

    for vid_dir in sorted(split_dir.iterdir()):
        if not vid_dir.is_dir() or not vid_dir.name.startswith('VID'):
            continue
        vid = vid_dir.name
        if (split_name, vid) in EXCLUDED_VIDEOS:
            print(f"  Skipping excluded video {split_name}/{vid}")
            continue
        json_file = annotation_json_path(dataset_root, split_name, vid, out_root)
        if not json_file.exists():
            print(f"  WARNING: No JSON found in {vid_dir}")
            continue

        data = json.load(open(json_file))
        annotations = data['annotations']  # dict: frame_id_str -> list of tool dicts
        image_width = data['video']['width']
        image_height = data['video']['height']

        # Source image directory
        if split_name == 'Testing':
            src_img_dir = extracted_frames_dir(out_root, split_name, vid)
        else:
            src_img_dir = vid_dir / 'Frames'
            if not src_img_dir.exists() and list(vid_dir.glob('*.mp4')):
                src_img_dir = extracted_frames_dir(out_root, split_name, vid)
                extract_annotated_frames(vid_dir, json_file, src_img_dir)

        # Output directories
        yolo_split = yolo_split_name(split_name)
        out_img_dir = Path(out_root) / 'images' / yolo_split / vid
        out_label_dir = Path(out_root) / 'labels' / yolo_split / vid
        out_img_dir.mkdir(parents=True, exist_ok=True)
        out_label_dir.mkdir(parents=True, exist_ok=True)

        if not src_img_dir.exists():
            print(f"  WARNING: No frames found in {src_img_dir}")
            continue

        for frame_id_str in sorted(annotations.keys(), key=lambda x: int(x)):
            frame_stem = f"{int(frame_id_str):06d}"
            src_img_path = src_img_dir / f"{frame_stem}.png"
            if not src_img_path.exists():
                continue
            tools = annotations.get(frame_id_str, [])

            # Output paths
            out_img_path = out_img_dir / (frame_stem + '.png')
            label_file = out_label_dir / (frame_stem + '.txt')

            # Copy image (or create symlink for training/val to save space)
            if src_img_path.resolve() != out_img_path.resolve():
                if split_name == 'Testing':
                    shutil.copy2(src_img_path, out_img_path)
                else:
                    # Symlink for training/validation to save space
                    if out_img_path.exists() or out_img_path.is_symlink():
                        out_img_path.unlink()
                    os.symlink(src_img_path.resolve(), out_img_path)

            # Write labels
            label_lines = []
            for tool in tools:
                bbox = tool['tool_bbox']   # [x, y, w, h] (top-left corner, COCO)
                cat_id = tool['instrument']
                if cat_id not in INSTRUMENT_ID_TO_CLASS:
                    continue
                cls = INSTRUMENT_ID_TO_CLASS[cat_id]
                x1, y1, w, h = normalize_bbox(bbox, image_width, image_height)
                # Convert to YOLO center format
                cx = x1 + w / 2.0
                cy = y1 + h / 2.0
                # Clip to [0,1]
                cx = max(0.0, min(1.0, cx))
                cy = max(0.0, min(1.0, cy))
                w = max(0.0, min(1.0, w))
                h = max(0.0, min(1.0, h))
                label_lines.append(f"{cls} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")

            with open(label_file, 'w') as f:
                f.write('\n'.join(label_lines))

            img_list_lines.append(str(out_img_path))

    return img_list_lines


def extract_test_frames(dataset_root, out_root):
    """Extract only annotated test frames and resize to the annotation size."""
    split_dir = Path(dataset_root) / 'Testing'
    for vid_dir in sorted(split_dir.iterdir()):
        if not vid_dir.is_dir() or not vid_dir.name.startswith('VID'):
            continue
        vid = vid_dir.name
        json_files = list(vid_dir.glob('*.json'))
        if not json_files:
            print(f"  WARNING: Missing MP4 or JSON in {vid_dir}")
            continue
        extract_annotated_frames(vid_dir, json_files[0], extracted_frames_dir(out_root, 'Testing', vid))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_root', default='/raid/cholectrack20')
    parser.add_argument('--out_root', default='/raid/cholectrack20_yolo')
    parser.add_argument('--skip_extract', action='store_true')
    args = parser.parse_args()

    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    # Extract test frames if needed
    if not args.skip_extract:
        print("Extracting test frames...")
        extract_test_frames(args.dataset_root, args.out_root)

    # Convert splits
    for split in ['Training', 'Validation', 'Testing']:
        print(f"Converting {split}...")
        img_lines = convert_split(args.dataset_root, split, args.out_root)
        
        # Write image list file
        if split == 'Training':
            list_file = out_root / 'train.txt'
        elif split == 'Validation':
            list_file = out_root / 'val.txt'
        else:
            list_file = out_root / 'test.txt'
        
        if img_lines:
            with open(list_file, 'w') as f:
                f.write('\n'.join(img_lines))
            print(f"  Wrote {len(img_lines)} image paths to {list_file}")
        else:
            with open(list_file, 'w') as f:
                pass
            print(f"  No images for {split}")

    print(f"\nDataset prepared at: {out_root}")
    print(f"Images at: {out_root}/images/{{train,val,test}}/")
    print(f"Labels at: {out_root}/labels/{{train,val,test}}/")


if __name__ == '__main__':
    main()

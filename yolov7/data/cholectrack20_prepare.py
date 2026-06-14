#!/usr/bin/env python3
"""
Convert CholecTrack20 dataset to YOLOv7 format.

Annotation bbox in CholecTrack20 JSON: [x_center_norm, y_center_norm, w_norm, h_norm]
YOLOv7 label format:  <class> <x_center_norm> <y_center_norm> <w_norm> <h_norm>

Classes (0-indexed, matching CholecTrack20):
  0: grasper
  1: bipolar
  2: hook
  3: scissors
  4: clipper
  5: irrigator
  6: specimen-bag
"""

import os
import json
import glob
import argparse
from pathlib import Path


CLASSES = ['grasper', 'bipolar', 'hook', 'scissors', 'clipper', 'irrigator', 'specimen-bag']

# Map CholecTrack20 instrument id -> 0-indexed class
INSTRUMENT_ID_TO_CLASS = {i: i for i in range(7)}


def convert_split(dataset_root, split_name, out_root):
    """Convert one split (Training/Validation/Testing) to YOLO labels."""
    split_dir = Path(dataset_root) / split_name
    img_list_lines = []

    for vid_dir in sorted(split_dir.iterdir()):
        if not vid_dir.is_dir() or not vid_dir.name.startswith('VID'):
            continue
        vid = vid_dir.name
        json_files = list(vid_dir.glob('*.json'))
        if not json_files:
            print(f"  WARNING: No JSON found in {vid_dir}")
            continue
        json_file = json_files[0]

        data = json.load(open(json_file))
        video_info = data['video']
        annotations = data['annotations']  # dict: frame_id_str -> list of tool dicts

        # Images: for Testing, look in out_root/test_frames/VID; else in dataset Frames/
        if split_name == 'Testing':
            img_dir = Path(out_root) / 'test_frames' / vid
        else:
            img_dir = vid_dir / 'Frames'

        # Labels go next to images for YOLO to find them
        label_dir = img_dir
        label_dir.mkdir(parents=True, exist_ok=True)

        frame_files = sorted(img_dir.glob('*.png')) if img_dir.exists() else []
        if not frame_files:
            print(f"  WARNING: No frames found in {img_dir}")
            continue

        for img_path in frame_files:
            frame_stem = img_path.stem
            frame_id_str = str(int(frame_stem))  # strip leading zeros
            tools = annotations.get(frame_id_str, [])

            label_lines = []
            for tool in tools:
                bbox = tool['tool_bbox']   # [x1_norm, y1_norm, w_norm, h_norm] (COCO top-left format)
                cat_id = tool['instrument']
                if cat_id not in INSTRUMENT_ID_TO_CLASS:
                    continue
                cls = INSTRUMENT_ID_TO_CLASS[cat_id]
                x1, y1, w, h = bbox
                # Convert to YOLO center format
                cx = x1 + w / 2.0
                cy = y1 + h / 2.0
                # Clip to [0,1]
                cx = max(0.0, min(1.0, cx))
                cy = max(0.0, min(1.0, cy))
                w  = max(0.0, min(1.0, w))
                h  = max(0.0, min(1.0, h))
                label_lines.append(f"{cls} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")

            label_file = label_dir / (img_path.stem + '.txt')
            with open(label_file, 'w') as f:
                f.write('\n'.join(label_lines))

            img_list_lines.append(str(img_path.resolve()))

    return img_list_lines


def extract_test_frames(dataset_root, out_root):
    """Extract frames from test set MP4s at 25fps, save only annotated frames to out_root."""
    import subprocess
    import json as json_mod
    split_dir = Path(dataset_root) / 'Testing'
    for vid_dir in sorted(split_dir.iterdir()):
        if not vid_dir.is_dir() or not vid_dir.name.startswith('VID'):
            continue
        vid = vid_dir.name
        mp4_files = list(vid_dir.glob('*.mp4'))
        json_files = list(vid_dir.glob('*.json'))
        if not mp4_files or not json_files:
            print(f"  WARNING: Missing MP4 or JSON in {vid_dir}")
            continue
        mp4 = mp4_files[0]
        data = json_mod.load(open(json_files[0]))
        ann_frame_ids = set(int(k) for k in data['annotations'].keys())

        frames_dir = Path(out_root) / 'test_frames' / vid
        frames_dir.mkdir(parents=True, exist_ok=True)
        # Check if already extracted
        existing = set(int(f.stem) for f in frames_dir.glob('*.png'))
        if existing >= ann_frame_ids:
            print(f"  {vid}: frames already extracted ({len(existing)} frames)")
            continue

        print(f"  Extracting {vid} frames at 25fps (keeping {len(ann_frame_ids)} annotated frames)...")
        import tempfile, shutil
        with tempfile.TemporaryDirectory() as tmpdir:
            cmd = [
                'ffmpeg', '-y', '-i', str(mp4),
                '-vf', 'fps=25',
                os.path.join(tmpdir, '%06d.png'),
                '-loglevel', 'error'
            ]
            subprocess.run(cmd, check=True)
            # ffmpeg names frames starting at 000001 = 25fps frame 1
            # annotation keys are 25fps frame numbers (1-indexed)
            for fid in sorted(ann_frame_ids):
                src = os.path.join(tmpdir, f'{fid:06d}.png')
                dst = frames_dir / f'{fid:06d}.png'
                if os.path.exists(src):
                    shutil.copy2(src, dst)
        n = len(list(frames_dir.glob('*.png')))
        print(f"    Saved {n}/{len(ann_frame_ids)} annotated frames")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_root', default='/raid/cholectrack20')
    parser.add_argument('--out_root', default='/raid/cholectrack20_yolo')
    parser.add_argument('--skip_extract', action='store_true', help='Skip test frame extraction')
    args = parser.parse_args()

    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    if not args.skip_extract:
        print("\nExtracting test set frames from MP4s...")
        extract_test_frames(args.dataset_root, args.out_root)

    split_map = {
        'Training':   'train',
        'Validation': 'val',
        'Testing':    'test',
    }

    for split_name, split_key in split_map.items():
        print(f"\nConverting {split_name}...")
        lines = convert_split(args.dataset_root, split_name, out_root)
        txt_path = out_root / f'{split_key}.txt'
        with open(txt_path, 'w') as f:
            f.write('\n'.join(lines) + '\n')
        print(f"  Wrote {len(lines)} image paths to {txt_path}")

    print("\nDone. Dataset prepared at:", out_root)


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""
Convert CholecTrack20 dataset to YOLOv7 format (v2 - proper structure).

Annotation bbox in CholecTrack20 JSON: [x_norm, y_norm, w_norm, h_norm] (top-left corner format, COCO style)
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


def convert_split(dataset_root, split_name, out_root):
    """Convert one split (Training/Validation/Testing) to YOLO format with proper directory structure."""
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
        annotations = data['annotations']  # dict: frame_id_str -> list of tool dicts

        # Source image directory
        if split_name == 'Testing':
            src_img_dir = Path(out_root) / 'test_frames' / vid
        else:
            src_img_dir = vid_dir / 'Frames'

        # Output directories
        out_img_dir = Path(out_root) / 'images' / split_name.lower() / vid
        out_label_dir = Path(out_root) / 'labels' / split_name.lower() / vid
        out_img_dir.mkdir(parents=True, exist_ok=True)
        out_label_dir.mkdir(parents=True, exist_ok=True)

        frame_files = sorted(src_img_dir.glob('*.png')) if src_img_dir.exists() else []
        if not frame_files:
            print(f"  WARNING: No frames found in {src_img_dir}")
            continue

        for src_img_path in frame_files:
            frame_stem = src_img_path.stem
            frame_id_str = str(int(frame_stem))  # strip leading zeros
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
                bbox = tool['tool_bbox']   # [x_norm, y_norm, w_norm, h_norm] (top-left corner, COCO)
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
                w = max(0.0, min(1.0, w))
                h = max(0.0, min(1.0, h))
                label_lines.append(f"{cls} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")

            with open(label_file, 'w') as f:
                f.write('\n'.join(label_lines))

            img_list_lines.append(str(out_img_path.resolve()))

    return img_list_lines


def extract_test_frames(dataset_root, out_root):
    """Extract frames from test set MP4s at 25fps, save only annotated frames."""
    try:
        import subprocess
    except:
        print("  WARNING: subprocess not available, skipping test frame extraction")
        return
    
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
        data = json.load(open(json_files[0]))
        ann_frame_ids = set(int(k) for k in data['annotations'].keys())

        frames_dir = Path(out_root) / 'test_frames' / vid
        frames_dir.mkdir(parents=True, exist_ok=True)
        
        # Check if already extracted
        existing = set(int(f.stem) for f in frames_dir.glob('*.png'))
        if existing >= ann_frame_ids:
            print(f"  {vid}: frames already extracted ({len(existing)} frames)")
            continue

        print(f"  Extracting {vid} frames at 25fps (keeping {len(ann_frame_ids)} annotated frames)...")
        import tempfile
        with tempfile.TemporaryDirectory() as tmpdir:
            cmd = [
                'ffmpeg', '-y', '-i', str(mp4),
                '-vf', 'fps=25',
                os.path.join(tmpdir, '%06d.png'),
                '-loglevel', 'error'
            ]
            try:
                subprocess.run(cmd, check=True)
                # ffmpeg names frames starting at 000001 = 25fps frame 1
                for fid in sorted(ann_frame_ids):
                    src = os.path.join(tmpdir, f'{fid:06d}.png')
                    dst = frames_dir / f'{fid:06d}.png'
                    if os.path.exists(src):
                        shutil.copy2(src, dst)
                n = len(list(frames_dir.glob('*.png')))
                print(f"    Saved {n}/{len(ann_frame_ids)} annotated frames")
            except Exception as e:
                print(f"    ERROR: {e}")


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

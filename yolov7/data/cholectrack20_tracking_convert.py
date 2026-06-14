#!/usr/bin/env python3
"""
Convert CholecTrack20 to tracking dataset format with consecutive frames.

Output format:
{
  "video_id": "VID01",
  "frames": ["frame_t.jpg", "frame_t+1.jpg", "frame_t+2.jpg"],
  "target_frame": "frame_t+2.jpg",
  "boxes_t": [...],
  "boxes_t_plus_1": [...],
  "boxes_t_plus_2": [...]
}

Boxes are normalized to [x1_norm, y1_norm, x2_norm, y2_norm] in [0,1].
"""

import os
import json
import glob
import argparse
from pathlib import Path
from collections import defaultdict


CLASSES = ['grasper', 'bipolar', 'hook', 'scissors', 'clipper', 'irrigator', 'specimen-bag']
OPERATORS = {0: 'null', 1: 'left', 2: 'right', 3: 'right'}  # Simplified: left/right/null


def get_operator_name(op_id):
    """Map operator ID to left/right/null."""
    return OPERATORS.get(op_id, 'null')


def convert_bbox_format(tool_bbox, img_width, img_height):
    """
    Convert CholecTrack20 bbox format to normalized corner format.
    
    Input: tool_bbox = [x_norm, y_norm, w_norm, h_norm] (already normalized to [0,1])
    Output: [x1_norm, y1_norm, x2_norm, y2_norm]
    """
    x_norm, y_norm, w_norm, h_norm = tool_bbox
    # Already normalized, just convert from center/top-left to corners
    # Assuming x_norm, y_norm are top-left corners (COCO format)
    x1_norm = x_norm
    y1_norm = y_norm
    x2_norm = x_norm + w_norm
    y2_norm = y_norm + h_norm
    # Clip to [0, 1]
    x1_norm = max(0.0, min(1.0, x1_norm))
    y1_norm = max(0.0, min(1.0, y1_norm))
    x2_norm = max(0.0, min(1.0, x2_norm))
    y2_norm = max(0.0, min(1.0, y2_norm))
    return [x1_norm, y1_norm, x2_norm, y2_norm]


def create_samples(dataset_root, split_name, out_root, sequence_length=3):
    """
    Create tracking samples with consecutive frames.
    
    Args:
        dataset_root: Path to CholecTrack20 root
        split_name: 'Training', 'Validation', or 'Testing'
        out_root: Output directory
        sequence_length: Number of consecutive frames per sample (default 3: t, t+1, t+2)
    """
    split_dir = Path(dataset_root) / split_name
    samples = []
    
    for vid_dir in sorted(split_dir.iterdir()):
        if not vid_dir.is_dir() or not vid_dir.name.startswith('VID'):
            continue
        vid = vid_dir.name
        
        json_files = list(vid_dir.glob('*.json'))
        if not json_files:
            print(f"  WARNING: No JSON for {vid}")
            continue
        
        data = json.load(open(json_files[0]))
        video_info = data['video']
        annotations = data['annotations']  # dict: frame_id_str -> list of tool dicts
        img_width = video_info['width']
        img_height = video_info['height']
        
        # Determine image directory
        if split_name == 'Testing':
            img_dir = Path(out_root) / 'test_frames' / vid
        else:
            img_dir = vid_dir / 'Frames'
        
        if not img_dir.exists():
            print(f"  WARNING: No frames for {vid}")
            continue
        
        # Get sorted frame IDs
        frame_ids = sorted([int(k) for k in annotations.keys()])
        
        # Get frame file mapping
        frame_files = {int(f.stem): str(f.resolve()) for f in img_dir.glob('*.png')}
        
        # Create consecutive samples
        for i in range(len(frame_ids) - sequence_length + 1):
            seq_frame_ids = frame_ids[i:i + sequence_length]
            
            # Check all frames exist
            if not all(fid in frame_files for fid in seq_frame_ids):
                continue
            
            # Build sample
            sample = {
                'video_id': vid,
                'frames': [frame_files[fid] for fid in seq_frame_ids[:-1]],
                'target_frame': frame_files[seq_frame_ids[-1]],
            }
            
            # Add boxes for each time step
            for step, fid in enumerate(seq_frame_ids):
                frame_id_str = str(fid)
                tools = annotations.get(frame_id_str, [])
                
                boxes = []
                for tool in tools:
                    bbox_norm = convert_bbox_format(
                        tool['tool_bbox'],
                        img_width,
                        img_height
                    )
                    box_dict = {
                        'class': CLASSES[tool['instrument']],
                        'operator': get_operator_name(tool.get('operator', 0)),
                        'bbox': bbox_norm,
                        'visibility': tool.get('visibility', 1),
                        'occluded': tool.get('occluded', 0),
                        'bleeding': tool.get('bleeding', 0),
                        'smoke': tool.get('smoke', 0),
                        'blurred': tool.get('blurred', 0),
                    }
                    boxes.append(box_dict)
                
                if step == 0:
                    sample['boxes_t'] = boxes
                elif step == 1:
                    sample['boxes_t_plus_1'] = boxes
                elif step == 2:
                    sample['boxes_t_plus_2'] = boxes
                else:
                    sample[f'boxes_t_plus_{step}'] = boxes
            
            samples.append(sample)
    
    return samples


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_root', default='/raid/cholectrack20')
    parser.add_argument('--out_root', default='/raid/cholectrack20_tracking')
    parser.add_argument('--sequence_length', type=int, default=3,
                        help='Number of consecutive frames per sample')
    args = parser.parse_args()
    
    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    
    split_map = {
        'Training': 'train',
        'Validation': 'val',
        'Testing': 'test',
    }
    
    for split_name, split_key in split_map.items():
        print(f"\nConverting {split_name} split...")
        samples = create_samples(
            args.dataset_root,
            split_name,
            args.out_root,
            sequence_length=args.sequence_length
        )
        
        out_file = out_root / f'{split_key}_samples.json'
        with open(out_file, 'w') as f:
            json.dump(samples, f, indent=2)
        
        print(f"  Created {len(samples)} samples -> {out_file}")
    
    print(f"\nDone! Dataset prepared at: {out_root}")


if __name__ == '__main__':
    main()

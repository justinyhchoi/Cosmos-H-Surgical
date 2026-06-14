#!/usr/bin/env python3
"""
Inspect and visualize CholecTrack20 tracking dataset samples.
"""

import json
import argparse
from pathlib import Path
from collections import Counter


def analyze_samples(samples_file):
    """Analyze dataset statistics."""
    with open(samples_file) as f:
        samples = json.load(f)
    
    print(f"\n{'='*60}")
    print(f"Dataset: {samples_file}")
    print(f"{'='*60}")
    print(f"Total samples: {len(samples)}")
    
    if not samples:
        return
    
    # Collect statistics
    videos = Counter()
    classes = Counter()
    operators = Counter()
    conditions_counts = Counter()
    
    for sample in samples:
        videos[sample['video_id']] += 1
        
        for step in ['boxes_t', 'boxes_t_plus_1', 'boxes_t_plus_2']:
            if step in sample:
                for box in sample[step]:
                    classes[box['class']] += 1
                    operators[box['operator']] += 1
                    
                    # Count challenging conditions
                    if box.get('occluded'): conditions_counts['occluded'] += 1
                    if box.get('bleeding'): conditions_counts['bleeding'] += 1
                    if box.get('smoke'): conditions_counts['smoke'] += 1
                    if box.get('blurred'): conditions_counts['blurred'] += 1
                    if not box.get('visibility'): conditions_counts['not_visible'] += 1
    
    print(f"\nVideos: {len(videos)}")
    for vid in sorted(videos.keys()):
        print(f"  {vid}: {videos[vid]} samples")
    
    print(f"\nTool classes (total boxes: {sum(classes.values())}):")
    for cls in sorted(classes.keys()):
        pct = 100 * classes[cls] / sum(classes.values())
        print(f"  {cls:15s}: {classes[cls]:6d} ({pct:5.1f}%)")
    
    print(f"\nOperators (total boxes: {sum(operators.values())}):")
    for op in sorted(operators.keys()):
        pct = 100 * operators[op] / sum(operators.values())
        print(f"  {op:15s}: {operators[op]:6d} ({pct:5.1f}%)")
    
    print(f"\nChallenging conditions:")
    total_boxes = sum(classes.values())
    for cond in sorted(conditions_counts.keys()):
        pct = 100 * conditions_counts[cond] / total_boxes
        print(f"  {cond:15s}: {conditions_counts[cond]:6d} ({pct:5.1f}%)")
    
    print(f"\nSample structure:")
    sample = samples[0]
    print(f"  Keys: {list(sample.keys())}")
    print(f"  Number of context frames: {len(sample['frames'])}")
    print(f"  Number of annotation time steps: {len([k for k in sample.keys() if k.startswith('boxes')])}")
    if sample.get('boxes_t'):
        print(f"  Example box fields: {list(sample['boxes_t'][0].keys())}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('samples_file', help='Path to samples JSON file')
    args = parser.parse_args()
    
    analyze_samples(args.samples_file)


if __name__ == '__main__':
    main()

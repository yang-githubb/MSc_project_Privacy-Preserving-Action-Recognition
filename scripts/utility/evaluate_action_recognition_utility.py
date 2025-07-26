#!/usr/bin/env python3
"""
Action Recognition Utility Evaluation Script

This script evaluates how well action recognition performance is preserved after anonymization.
It compares Top-1 accuracy on original vs anonymized videos to measure utility retention.

Usage:
    python evaluate_action_recognition_utility.py --original_dir /path/to/original --anonymized_dir /path/to/anonymized
"""

import os
import argparse
import numpy as np
import torch
from pathlib import Path
import json
from collections import defaultdict

# MMAction2 imports
import sys
sys.path.append('mmaction2')
from mmaction.apis import inference_recognizer, init_recognizer
from mmaction.datasets import build_dataloader, build_dataset
from mmaction.core.evaluation import top_k_accuracy


def load_action_model(config_path, checkpoint_path, device='cuda'):
    """Load pre-trained action recognition model."""
    model = init_recognizer(config_path, checkpoint_path, device=device)
    return model


def evaluate_video_actions(model, video_path, label_map_path=None):
    """Evaluate action recognition on a single video."""
    try:
        # Get prediction
        result = inference_recognizer(model, video_path)
        
        # Load label map if provided
        if label_map_path and os.path.exists(label_map_path):
            with open(label_map_path, 'r') as f:
                label_map = [line.strip() for line in f.readlines()]
            predicted_label = label_map[result[0][0]]
            confidence = result[0][1]
        else:
            predicted_label = result[0][0]
            confidence = result[0][1]
            
        return predicted_label, confidence
        
    except Exception as e:
        print(f"Error processing {video_path}: {e}")
        return None, 0.0


def evaluate_directory(model, directory, label_map_path=None, ground_truth_file=None):
    """Evaluate all videos in a directory."""
    results = {}
    video_files = []
    
    # Find video files
    for ext in ['.mp4', '.avi', '.mov', '.mkv']:
        video_files.extend(Path(directory).glob(f'*{ext}'))
    
    print(f"Found {len(video_files)} video files in {directory}")
    
    # Load ground truth if provided
    ground_truth = {}
    if ground_truth_file and os.path.exists(ground_truth_file):
        with open(ground_truth_file, 'r') as f:
            for line in f:
                parts = line.strip().split(',')
                if len(parts) >= 2:
                    video_name = parts[0]
                    true_label = parts[1]
                    ground_truth[video_name] = true_label
    
    # Evaluate each video
    for video_path in video_files:
        video_name = video_path.stem
        predicted_label, confidence = evaluate_video_actions(model, str(video_path), label_map_path)
        
        if predicted_label is not None:
            results[video_name] = {
                'predicted_label': predicted_label,
                'confidence': confidence,
                'true_label': ground_truth.get(video_name, 'unknown')
            }
    
    return results


def calculate_accuracy(results, ground_truth_file=None):
    """Calculate Top-1 accuracy from results."""
    if not results:
        return 0.0
    
    correct = 0
    total = 0
    
    for video_name, result in results.items():
        predicted = result['predicted_label']
        true_label = result['true_label']
        
        if true_label != 'unknown':
            total += 1
            if str(predicted).lower() == str(true_label).lower():
                correct += 1
    
    return (correct / total * 100) if total > 0 else 0.0


def main():
    parser = argparse.ArgumentParser(description="Evaluate Action Recognition Utility Preservation")
    parser.add_argument('--original_dir', type=str, required=True,
                       help='Directory containing original videos')
    parser.add_argument('--anonymized_dir', type=str, required=True,
                       help='Directory containing anonymized videos')
    parser.add_argument('--config', type=str, 
                       default='mmaction2/configs/recognition/tsn/tsn_r50_1x1x3_100e_kinetics400_rgb.py',
                       help='MMAction2 config file')
    parser.add_argument('--checkpoint', type=str,
                       default='mmaction2/checkpoints/tsn_r50_1x1x3_100e_kinetics400_rgb_20200614-e508be42.pth',
                       help='Model checkpoint path')
    parser.add_argument('--label_map', type=str,
                       help='Path to label map file (optional)')
    parser.add_argument('--ground_truth', type=str,
                       help='Path to ground truth file (optional)')
    parser.add_argument('--output', type=str, default='action_recognition_results.json',
                       help='Output file for results')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to use (cuda/cpu)')
    
    args = parser.parse_args()
    
    print("🎬 Action Recognition Utility Evaluation")
    print("=" * 60)
    print("Setup:")
    print("  • Original videos:", args.original_dir)
    print("  • Anonymized videos:", args.anonymized_dir)
    print("  • Goal: Measure utility preservation after anonymization")
    print("  • High accuracy retention = Good utility preservation")
    print("  • Low accuracy retention = Poor utility preservation")
    print()
    
    # Load model
    print("🧠 Loading action recognition model...")
    try:
        model = load_action_model(args.config, args.checkpoint, args.device)
        print("✅ Model loaded successfully")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        print("Please ensure MMAction2 is properly installed and checkpoint is available")
        return
    
    # Evaluate original videos
    print(f"\n📹 Evaluating original videos in: {args.original_dir}")
    original_results = evaluate_directory(model, args.original_dir, args.label_map, args.ground_truth)
    original_accuracy = calculate_accuracy(original_results, args.ground_truth)
    
    # Evaluate anonymized videos
    print(f"\n🔐 Evaluating anonymized videos in: {args.anonymized_dir}")
    anonymized_results = evaluate_directory(model, args.anonymized_dir, args.label_map, args.ground_truth)
    anonymized_accuracy = calculate_accuracy(anonymized_results, args.ground_truth)
    
    # Calculate utility retention
    accuracy_drop = original_accuracy - anonymized_accuracy
    utility_retention = max(0, 100 - abs(accuracy_drop))
    
    # Display results
    print("\n" + "=" * 60)
    print("🎬 ACTION RECOGNITION UTILITY RESULTS")
    print("=" * 60)
    
    print(f"📊 Original Videos Top-1 Accuracy: {original_accuracy:.2f}%")
    print(f"📊 Anonymized Videos Top-1 Accuracy: {anonymized_accuracy:.2f}%")
    print(f"📊 Accuracy Drop: {accuracy_drop:.2f}%")
    print(f"📊 Utility Retention: {utility_retention:.2f}%")
    print()
    
    # Utility interpretation
    print("🎯 UTILITY ASSESSMENT:")
    if utility_retention >= 90:
        utility_level = "EXCELLENT"
        utility_emoji = "🟢"
    elif utility_retention >= 75:
        utility_level = "GOOD"
        utility_emoji = "🟡"
    elif utility_retention >= 50:
        utility_level = "MODERATE"
        utility_emoji = "🟠"
    else:
        utility_level = "POOR"
        utility_emoji = "🔴"
    
    print(f"{utility_emoji} Utility Level: {utility_level}")
    print(f"   • {utility_retention:.1f}% of action recognition performance is preserved")
    print(f"   • This indicates how well anonymization maintains video utility")
    print()
    
    if utility_retention >= 75:
        print("✅ GOOD NEWS: Anonymization preserves utility well!")
        print("   The high utility retention suggests actions are still recognizable.")
    else:
        print("⚠️  CONCERN: Anonymization may be affecting utility!")
        print("   The low utility retention suggests actions are being obscured.")
        print("   Consider adjusting anonymization parameters.")
    
    # Save detailed results
    results_summary = {
        'original_accuracy': original_accuracy,
        'anonymized_accuracy': anonymized_accuracy,
        'accuracy_drop': accuracy_drop,
        'utility_retention': utility_retention,
        'utility_level': utility_level,
        'original_results': original_results,
        'anonymized_results': anonymized_results
    }
    
    with open(args.output, 'w') as f:
        json.dump(results_summary, f, indent=2)
    
    print(f"\n💾 Detailed results saved to: {args.output}")
    print("\n" + "=" * 60)


if __name__ == "__main__":
    main() 
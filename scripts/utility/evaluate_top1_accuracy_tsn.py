#!/usr/bin/env python3
"""
Top-1 Accuracy Evaluation for DeepPrivacy2 Anonymization Impact

This script evaluates how DeepPrivacy2 full-body anonymization affects task-level 
performance using Top-1 accuracy on a UCF101 subset (10 classes × 10 videos = 100 total videos).

Uses MMAction2's TSN model for action recognition evaluation.
"""

# Fix for NumPy deprecation: np.int was removed in NumPy ≥ 1.24
import numpy as np
np.int = int  # patch deprecated alias

import os
import argparse
import csv
import json
import warnings
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm

# UCF101 class labels
UCF101_LABELS = [
    'ApplyEyeMakeup', 'ApplyLipstick', 'Archery', 'BabyCrawling', 'BalanceBeam',
    'BandMarching', 'BaseballPitch', 'Basketball', 'BasketballDunk', 'BenchPress',
    'Biking', 'Billiards', 'BlowDryHair', 'BlowingCandles', 'BodyWeightSquats',
    'Bowling', 'BoxingPunchingBag', 'BoxingSpeedBag', 'BreastStroke', 'BrushingTeeth',
    'CleanAndJerk', 'CliffDiving', 'CricketBowling', 'CricketShot', 'CuttingInKitchen',
    'Diving', 'Drumming', 'Fencing', 'FieldHockeyPenalty', 'FloorGymnastics',
    'FrisbeeCatch', 'FrontCrawl', 'GolfSwing', 'Haircut', 'Hammering', 'HammerThrow',
    'HandstandPushups', 'HandstandWalking', 'HeadMassage', 'HighJump', 'HorseRace',
    'HorseRiding', 'HulaHoop', 'IceDancing', 'JavelinThrow', 'JugglingBalls',
    'JumpingJack', 'JumpRope', 'Kayaking', 'Knitting', 'LongJump', 'Lunges',
    'MilitaryParade', 'Mixing', 'MoppingFloor', 'Nunchucks', 'ParallelBars',
    'PizzaTossing', 'PlayingCello', 'PlayingDaf', 'PlayingDhol', 'PlayingFlute',
    'PlayingGuitar', 'PlayingPiano', 'PlayingSitar', 'PlayingTabla', 'PlayingViolin',
    'PoleVault', 'PommelHorse', 'PullUps', 'Punch', 'PushUps', 'Rafting',
    'RockClimbingIndoor', 'RopeClimbing', 'Rowing', 'SalsaSpin', 'ShavingBeard',
    'Shotput', 'SkateBoarding', 'Skiing', 'Skijet', 'SkyDiving', 'SoccerJuggling',
    'SoccerPenalty', 'StillRings', 'SumoWrestling', 'Surfing', 'Swing',
    'TableTennisShot', 'TaiChi', 'TennisSwing', 'ThrowDiscus', 'TrampolineJumping',
    'Typing', 'UnevenBars', 'VolleyballSpiking', 'WalkingWithDog', 'WallPushups',
    'WritingOnBoard', 'YoYo'
]

# Suppress NumPy deprecation warnings
warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.filterwarnings('ignore', message='.*np.int.*')

# MMAction2 imports
try:
    from mmaction.apis import inference_recognizer, init_recognizer
except Exception as e:
    print(f"Error importing MMAction2: {e}")
    exit(1)

def load_ground_truth(csv_path):
    """Load ground truth labels from CSV file"""
    gt = {}
    with open(csv_path, newline='') as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) >= 2:
                video = Path(row[0]).stem
                label = row[1].strip()
                gt[video] = label
    return gt

def create_all_matching_videos(original_dir, anonymized_dir):
    """Create a list of all matching original/anonymized video pairs"""
    all_pairs = []

    for class_name in os.listdir(original_dir):
        class_dir = os.path.join(original_dir, class_name)
        if not os.path.isdir(class_dir):
            continue

        for file in os.listdir(class_dir):
            if not file.endswith(('.avi', '.mp4', '.mov')):
                continue

            video_name = Path(file).stem
            original_path = os.path.join(class_dir, file)
            anonymized_path = os.path.join(anonymized_dir, class_name, f"{video_name}_anonymized.mp4")

            if os.path.exists(anonymized_path):
                all_pairs.append({
                    'class_name': class_name,
                    'video_name': video_name,
                    'original_path': original_path,
                    'anonymized_path': anonymized_path
                })

    return all_pairs

def evaluate_videos(model, video_list, ground_truth=None, max_videos=None):
    """Evaluate Top-1 accuracy on a list of videos"""
    correct, total = 0, 0
    per_class_results = defaultdict(lambda: [0, 0])  # [correct, total]
    results = []
    
    # Limit number of videos if specified
    if max_videos:
        video_list = video_list[:max_videos]
        print(f"Limiting evaluation to {max_videos} videos for testing")
    
    for video_info in tqdm(video_list, desc="Evaluating videos"):
        video_path = video_info['original_path'] if 'original_path' in video_info else video_info['video_path']
        video_name = video_info['video_name']
        class_name = video_info['class_name']
        
        # Get ground truth label
        if ground_truth and video_name in ground_truth:
            true_label = ground_truth[video_name].lower()
        else:
            true_label = class_name.lower()
        
        try:
            # Run inference
            result = inference_recognizer(model, video_path)
            
            if result and len(result) > 0:
                pred_class = result[0][0]  # integer index
                # Get the class name from UCF101 labels
                try:
                    if 0 <= pred_class < len(UCF101_LABELS):
                        pred_label = UCF101_LABELS[pred_class].lower()
                    else:
                        # Fallback: use the class index as label
                        pred_label = f"class_{pred_class}".lower()
                except Exception as e:
                    print(f"Warning: Could not get label for class {pred_class}: {e}")
                    pred_label = f"class_{pred_class}".lower()
                
                # Check if prediction is correct
                is_correct = pred_label == true_label
                
                if is_correct:
                    correct += 1
                    per_class_results[class_name][0] += 1
                
                per_class_results[class_name][1] += 1
                total += 1
                
                # Debug info for first few videos
                if total <= 5:
                    print(f"  Video: {video_name}")
                    print(f"    True: {true_label}, Pred: {pred_label}, Correct: {is_correct}")
                    print(f"    Confidence: {result[0][1]:.4f}")
                
                # Store result
                results.append({
                    'video_name': video_name,
                    'class_name': class_name,
                    'true_label': true_label,
                    'pred_label': pred_label,
                    'pred_confidence': result[0][1] if len(result[0]) > 1 else None,
                    'is_correct': is_correct,
                    'video_path': video_path
                })
                
        except Exception as e:
            print(f"Error processing {video_path}: {e}")
            continue
    
    # Calculate accuracy
    accuracy = 100 * correct / total if total > 0 else 0
    
    return accuracy, correct, total, per_class_results, results

def main():
    parser = argparse.ArgumentParser(description="Evaluate Top-1 Accuracy on Original vs Anonymized Videos")

    # Only keep essential inputs
    parser.add_argument('--original_dir', type=str, required=True,
                        help='Path to original UCF101 videos')
    parser.add_argument('--anonymized_dir', type=str, required=True,
                        help='Path to anonymized UCF101 videos')
    parser.add_argument('--device', type=str, default='auto',
                        help='Device to use for inference (auto, cuda, cpu)')

    args = parser.parse_args()

    # HARDCODED CONFIGURATION
    args.config = '/work/tc067/tc067/s2737744/mmaction2/configs/recognition/tsn/tsn_r50_1x1x3_75e_ucf101_rgb.py'
    args.checkpoint = '/work/tc067/tc067/s2737744/mmaction2/checkpoints/tsn_r50_1x1x3_75e_ucf101_rgb_20201023-d85ab600.pth'
    args.output_file = '/work/tc067/tc067/s2737744/output/top1_accuracy_results.json'
    
    # Auto-detect device if 'auto' is specified
    if args.device == 'auto':
        try:
            import torch
            if torch.cuda.is_available():
                args.device = 'cuda'
                print(f"CUDA available, using GPU")
            else:
                args.device = 'cpu'
                print(f"CUDA not available, using CPU")
        except ImportError:
            args.device = 'cpu'
            print(f"PyTorch not available, using CPU")
    
    # Check if directories exist
    if not os.path.exists(args.original_dir):
        print(f"Error: Original directory not found: {args.original_dir}")
        return
    
    if not os.path.exists(args.anonymized_dir):
        print(f"Error: Anonymized directory not found: {args.anonymized_dir}")
        return
    
    # Create evaluation subset
    print("Creating evaluation subset...")
    subset_videos = create_all_matching_videos(args.original_dir, args.anonymized_dir)
    
    if not subset_videos:
        print("Error: No matching video pairs found for evaluation")
        return
    
    print(f"Created subset with {len(subset_videos)} video pairs")
    
    # Initialize model
    print("Initializing TSN model...")
    try:
    
        # Initialize model
        model = init_recognizer(args.config, args.checkpoint, device=args.device)
        print(f"Model loaded successfully on {args.device}")
        
    except Exception as e:
        print(f"Error initializing model: {e}")
        print("Make sure MMAction2 is properly installed and the checkpoint is available")
        return
    
    # Prepare video lists for evaluation
    original_videos = []
    anonymized_videos = []
    
    for video_info in subset_videos:
        # Original video
        original_videos.append({
            'video_path': video_info['original_path'],
            'video_name': video_info['video_name'],
            'class_name': video_info['class_name']
        })
        
        # Anonymized video
        anonymized_videos.append({
            'video_path': video_info['anonymized_path'],
            'video_name': video_info['video_name'],
            'class_name': video_info['class_name']
        })
    
    # Evaluate original videos
    print("\n" + "="*50)
    print("EVALUATING ORIGINAL VIDEOS")
    print("="*50)
    orig_acc, orig_corr, orig_tot, orig_per_class, orig_results = evaluate_videos(
        model, original_videos, None, max_videos=10
    )
    print(f"Original Top-1 Accuracy: {orig_acc:.2f}% ({orig_corr}/{orig_tot})")
    
    # Evaluate anonymized videos
    print("\n" + "="*50)
    print("EVALUATING ANONYMIZED VIDEOS")
    print("="*50)
    anon_acc, anon_corr, anon_tot, anon_per_class, anon_results = evaluate_videos(
        model, anonymized_videos, None, max_videos=10
    )
    print(f"Anonymized Top-1 Accuracy: {anon_acc:.2f}% ({anon_corr}/{anon_tot})")
    
    # Calculate performance drop
    accuracy_drop = orig_acc - anon_acc
    relative_drop = (accuracy_drop / orig_acc * 100) if orig_acc > 0 else 0
    
    # Print summary
    print("\n" + "="*50)
    print("EVALUATION SUMMARY")
    print("="*50)
    print(f"Total Videos Evaluated: {len(subset_videos)}")
    print(f"Original Accuracy: {orig_acc:.2f}%")
    print(f"Anonymized Accuracy: {anon_acc:.2f}%")
    print(f"Absolute Accuracy Drop: {accuracy_drop:.2f}%")
    print(f"Relative Accuracy Drop: {relative_drop:.2f}%")
    
    # Interpret results
    if accuracy_drop < 5:
        impact_level = "MINIMAL"
    elif accuracy_drop < 15:
        impact_level = "MODERATE"
    elif accuracy_drop < 30:
        impact_level = "SIGNIFICANT"
    else:
        impact_level = "SEVERE"
    
    print(f"Anonymization Impact: {impact_level}")
    
    # Per-class analysis
    print(f"\nPer-Class Analysis:")
    all_classes = set(orig_per_class.keys()) | set(anon_per_class.keys())
    
    for class_name in sorted(all_classes):
        orig_corr, orig_tot = orig_per_class[class_name]
        anon_corr, anon_tot = anon_per_class[class_name]
        
        if orig_tot > 0 and anon_tot > 0:
            orig_acc = 100 * orig_corr / orig_tot
            anon_acc = 100 * anon_corr / anon_tot
            drop = orig_acc - anon_acc
            
            print(f"  {class_name}: {orig_acc:.1f}% → {anon_acc:.1f}% (drop: {drop:.1f}%)")
    
    # Save detailed results if requested
    if args.output_file:
        detailed_results = {
            'evaluation_config': {
                'total_videos': len(subset_videos)
            },
            'summary': {
                'original_accuracy': orig_acc,
                'anonymized_accuracy': anon_acc,
                'accuracy_drop': accuracy_drop,
                'relative_drop': relative_drop,
                'impact_level': impact_level
            },
            'per_class_results': {
                'original': dict(orig_per_class),
                'anonymized': dict(anon_per_class)
            },
            'detailed_results': {
                'original': orig_results,
                'anonymized': anon_results
            }
        }
        
        with open(args.output_file, 'w') as f:
            json.dump(detailed_results, f, indent=2, default=str)
        print(f"\nDetailed results saved to: {args.output_file}")

if __name__ == "__main__":
    main() 
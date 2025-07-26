#!/usr/bin/env python3
"""
Comprehensive Privacy-Utility Evaluation Script

This script provides a unified evaluation of both privacy protection and utility preservation
after video anonymization. It combines Re-ID evaluation (privacy) and action recognition
evaluation (utility) to give a complete picture of the privacy-utility trade-off.

Usage:
    python comprehensive_evaluation.py --original_videos /path/to/original --anonymized_videos /path/to/anonymized
"""

import os
import argparse
import json
import subprocess
import sys
from pathlib import Path
import time

def run_command(cmd, description):
    """Run a command and handle errors."""
    print(f"\n🔄 {description}")
    print(f"Command: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        print(f"✅ {description} completed successfully")
        return result.stdout
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed:")
        print(f"Error: {e.stderr}")
        return None

def extract_frames(video_dir, output_dir, description):
    """Extract frames from videos."""
    print(f"\n📹 {description}")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Use the existing extract_frames script
    cmd = [
        'python', 'scripts/extract_frames.py',
        '--input_dir', video_dir,
        '--output_dir', output_dir
    ]
    
    return run_command(cmd, f"Extracting frames for {description}")

def extract_person_crops(frames_dir, crops_dir, description):
    """Extract person crops from frames."""
    print(f"\n👤 {description}")
    
    # Create output directory
    os.makedirs(crops_dir, exist_ok=True)
    
    cmd = [
        'python', 'scripts/re-id/extract_person_crops_yolov5.py',
        '--input_dir', frames_dir,
        '--output_dir', crops_dir
    ]
    
    return run_command(cmd, f"Extracting person crops for {description}")

def organize_reid_dataset(crops_dir, reid_dir, description):
    """Organize crops into Re-ID format."""
    print(f"\n📁 {description}")
    
    # Create output directory
    os.makedirs(reid_dir, exist_ok=True)
    
    cmd = [
        'python', 'scripts/re-id/organize_reid_dataset.py',
        '--input_dir', crops_dir,
        '--output_dir', reid_dir
    ]
    
    return run_command(cmd, f"Organizing Re-ID dataset for {description}")

def evaluate_privacy(query_dir, gallery_dir):
    """Evaluate privacy using Re-ID."""
    print(f"\n🔐 Evaluating Privacy (Re-ID)")
    
    cmd = [
        'python', 'scripts/re-id/evaluate_reid_osnet.py',
        '--query_dir', query_dir,
        '--gallery_dir', gallery_dir
    ]
    
    output = run_command(cmd, "Privacy evaluation")
    
    # Parse results from output
    if output:
        lines = output.split('\n')
        rank1_acc = None
        map_score = None
        
        for line in lines:
            if 'Rank-1 Accuracy:' in line:
                rank1_acc = float(line.split(':')[1].strip().replace('%', ''))
            elif 'mAP Score:' in line:
                map_score = float(line.split(':')[1].strip().replace('%', ''))
        
        return {
            'rank1_accuracy': rank1_acc,
            'map_score': map_score,
            'raw_output': output
        }
    
    return None

def evaluate_utility(original_dir, anonymized_dir, ground_truth=None):
    """Evaluate utility using action recognition."""
    print(f"\n🎬 Evaluating Utility (Action Recognition)")
    
    cmd = [
        'python', 'scripts/evaluate_action_recognition_utility.py',
        '--original_dir', original_dir,
        '--anonymized_dir', anonymized_dir
    ]
    
    if ground_truth:
        cmd.extend(['--ground_truth', ground_truth])
    
    output = run_command(cmd, "Utility evaluation")
    
    # Parse results from output
    if output:
        lines = output.split('\n')
        original_acc = None
        anonymized_acc = None
        utility_retention = None
        
        for line in lines:
            if 'Original Videos Top-1 Accuracy:' in line:
                original_acc = float(line.split(':')[1].strip().replace('%', ''))
            elif 'Anonymized Videos Top-1 Accuracy:' in line:
                anonymized_acc = float(line.split(':')[1].strip().replace('%', ''))
            elif 'Utility Retention:' in line:
                utility_retention = float(line.split(':')[1].strip().replace('%', ''))
        
        return {
            'original_accuracy': original_acc,
            'anonymized_accuracy': anonymized_acc,
            'utility_retention': utility_retention,
            'raw_output': output
        }
    
    return None

def generate_comprehensive_report(privacy_results, utility_results, output_file):
    """Generate a comprehensive evaluation report."""
    
    print("\n" + "=" * 80)
    print("🔍 COMPREHENSIVE PRIVACY-UTILITY EVALUATION REPORT")
    print("=" * 80)
    
    # Privacy Assessment
    if privacy_results:
        rank1 = privacy_results.get('rank1_accuracy', 0)
        map_score = privacy_results.get('map_score', 0)
        
        print(f"\n🔐 PRIVACY ASSESSMENT (Re-Identification)")
        print(f"   Rank-1 Accuracy: {rank1:.2f}%")
        print(f"   mAP Score: {map_score:.2f}%")
        
        # Privacy interpretation
        if rank1 < 10:
            privacy_level = "EXCELLENT"
            privacy_emoji = "🟢"
        elif rank1 < 25:
            privacy_level = "GOOD"
            privacy_emoji = "🟡"
        elif rank1 < 50:
            privacy_level = "MODERATE"
            privacy_emoji = "🟠"
        else:
            privacy_level = "POOR"
            privacy_emoji = "🔴"
        
        print(f"   {privacy_emoji} Privacy Level: {privacy_level}")
    
    # Utility Assessment
    if utility_results:
        original_acc = utility_results.get('original_accuracy', 0)
        anonymized_acc = utility_results.get('anonymized_accuracy', 0)
        utility_retention = utility_results.get('utility_retention', 0)
        
        print(f"\n🎬 UTILITY ASSESSMENT (Action Recognition)")
        print(f"   Original Accuracy: {original_acc:.2f}%")
        print(f"   Anonymized Accuracy: {anonymized_acc:.2f}%")
        print(f"   Utility Retention: {utility_retention:.2f}%")
        
        # Utility interpretation
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
        
        print(f"   {utility_emoji} Utility Level: {utility_level}")
    
    # Overall Assessment
    print(f"\n📊 OVERALL ASSESSMENT")
    
    if privacy_results and utility_results:
        rank1 = privacy_results.get('rank1_accuracy', 0)
        utility_retention = utility_results.get('utility_retention', 0)
        
        # Combined score (lower is better for privacy, higher is better for utility)
        privacy_score = max(0, 100 - rank1)  # Convert to privacy score
        combined_score = (privacy_score + utility_retention) / 2
        
        print(f"   Privacy Score: {privacy_score:.1f}%")
        print(f"   Utility Score: {utility_retention:.1f}%")
        print(f"   Combined Score: {combined_score:.1f}%")
        
        # Overall recommendation
        if combined_score >= 80:
            overall_level = "EXCELLENT"
            overall_emoji = "🟢"
            recommendation = "Anonymization achieves excellent privacy-utility balance!"
        elif combined_score >= 60:
            overall_level = "GOOD"
            overall_emoji = "🟡"
            recommendation = "Anonymization provides good privacy-utility balance."
        elif combined_score >= 40:
            overall_level = "MODERATE"
            overall_emoji = "🟠"
            recommendation = "Anonymization has moderate privacy-utility trade-off."
        else:
            overall_level = "POOR"
            overall_emoji = "🔴"
            recommendation = "Anonymization needs improvement for better balance."
        
        print(f"   {overall_emoji} Overall Level: {overall_level}")
        print(f"   💡 Recommendation: {recommendation}")
    
    # Save comprehensive results
    comprehensive_results = {
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'privacy_results': privacy_results,
        'utility_results': utility_results,
        'summary': {
            'privacy_level': privacy_level if privacy_results else 'N/A',
            'utility_level': utility_level if utility_results else 'N/A',
            'overall_level': overall_level if privacy_results and utility_results else 'N/A',
            'combined_score': combined_score if privacy_results and utility_results else 'N/A'
        }
    }
    
    with open(output_file, 'w') as f:
        json.dump(comprehensive_results, f, indent=2)
    
    print(f"\n💾 Comprehensive results saved to: {output_file}")
    print("\n" + "=" * 80)

def main():
    parser = argparse.ArgumentParser(description="Comprehensive Privacy-Utility Evaluation")
    parser.add_argument('--original_videos', type=str, required=True,
                       help='Directory containing original videos')
    parser.add_argument('--anonymized_videos', type=str, required=True,
                       help='Directory containing anonymized videos')
    parser.add_argument('--ground_truth', type=str,
                       help='Path to ground truth file for action recognition (optional)')
    parser.add_argument('--output', type=str, default='comprehensive_evaluation_results.json',
                       help='Output file for comprehensive results')
    parser.add_argument('--skip_frames', action='store_true',
                       help='Skip frame extraction if already done')
    parser.add_argument('--skip_crops', action='store_true',
                       help='Skip person crop extraction if already done')
    parser.add_argument('--skip_reid_org', action='store_true',
                       help='Skip Re-ID dataset organization if already done')
    
    args = parser.parse_args()
    
    print("🔍 Comprehensive Privacy-Utility Evaluation")
    print("=" * 60)
    print("This script evaluates both privacy protection and utility preservation")
    print("after video anonymization using a unified pipeline.")
    print()
    
    # Create temporary directories
    temp_dir = Path('temp_evaluation')
    temp_dir.mkdir(exist_ok=True)
    
    original_frames = temp_dir / 'original_frames'
    anonymized_frames = temp_dir / 'anonymized_frames'
    original_crops = temp_dir / 'original_crops'
    anonymized_crops = temp_dir / 'anonymized_crops'
    reid_gallery = temp_dir / 'reid_gallery'
    reid_query = temp_dir / 'reid_query'
    
    # Step 1: Extract frames (if not skipped)
    if not args.skip_frames:
        extract_frames(args.original_videos, str(original_frames), "Original Videos")
        extract_frames(args.anonymized_videos, str(anonymized_frames), "Anonymized Videos")
    else:
        print("⏭️  Skipping frame extraction")
    
    # Step 2: Extract person crops (if not skipped)
    if not args.skip_crops:
        extract_person_crops(str(original_frames), str(original_crops), "Original Frames")
        extract_person_crops(str(anonymized_frames), str(anonymized_crops), "Anonymized Frames")
    else:
        print("⏭️  Skipping person crop extraction")
    
    # Step 3: Organize Re-ID datasets (if not skipped)
    if not args.skip_reid_org:
        organize_reid_dataset(str(original_crops), str(reid_gallery), "Original Crops")
        organize_reid_dataset(str(anonymized_crops), str(reid_query), "Anonymized Crops")
    else:
        print("⏭️  Skipping Re-ID dataset organization")
    
    # Step 4: Evaluate Privacy (Re-ID)
    privacy_results = evaluate_privacy(str(reid_query), str(reid_gallery))
    
    # Step 5: Evaluate Utility (Action Recognition)
    utility_results = evaluate_utility(args.original_videos, args.anonymized_videos, args.ground_truth)
    
    # Step 6: Generate comprehensive report
    generate_comprehensive_report(privacy_results, utility_results, args.output)
    
    print("\n✅ Comprehensive evaluation completed!")
    print(f"📄 Results saved to: {args.output}")
    print(f"🗂️  Temporary files in: {temp_dir}")

if __name__ == "__main__":
    main() 
import os
import shutil
import random
import argparse
from pathlib import Path

def organize_reid_data(original_cropped_dir, anonymized_cropped_dir, output_dir, 
                      anonymized_query_count=1, original_gallery_count=1):
    """
    Organize cropped frames for re-ID evaluation
    
    This function organizes the data as follows:
    - Puts one anonymized frame in query (ID 1)
    - Puts remaining anonymized frames in gallery (ID 1) 
    - Puts one original frame in gallery (ID 1, same person)
    - This allows testing if anonymized frames match each other vs original frame
    - High score for original frame = bad anonymization, low score = good anonymization
    
    Args:
        original_cropped_dir (str): Directory with cropped original frames
        anonymized_cropped_dir (str): Directory with cropped anonymized frames
        output_dir (str): Output directory for organized re-ID data
        anonymized_query_count (int): Number of anonymized frames to put in query (default: 1)
        original_gallery_count (int): Number of original frames to put in gallery (default: 1)
    """
    # Create output directories for query and gallery
    query_dir = os.path.join(output_dir, "query")
    gallery_dir = os.path.join(output_dir, "gallery")
    
    os.makedirs(query_dir, exist_ok=True)
    os.makedirs(gallery_dir, exist_ok=True)
    
    # Get all anonymized frames
    anonymized_frames = []
    for filename in os.listdir(anonymized_cropped_dir):
        if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
            anonymized_frames.append(filename)
    
    # Get all original frames
    original_frames = []
    for filename in os.listdir(original_cropped_dir):
        if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
            original_frames.append(filename)
    
    # Shuffle frames for random selection
    random.shuffle(anonymized_frames)
    random.shuffle(original_frames)
    
    # Put one anonymized frame in query (ID 1)
    if anonymized_frames:
        query_frame = anonymized_frames[0]
        shutil.copy2(
            os.path.join(anonymized_cropped_dir, query_frame),
            os.path.join(query_dir, f"0001_c1_{query_frame}")
        )
        
        # Put remaining anonymized frames in gallery (ID 1)
        for i, frame in enumerate(anonymized_frames[1:], 1):
            gallery_name = f"0001_c1_{i:03d}_{frame}"
            shutil.copy2(
                os.path.join(anonymized_cropped_dir, frame),
                os.path.join(gallery_dir, gallery_name)
            )
    
    # Put one original frame in gallery (ID 1, same person)
    if original_frames:
        original_frame = original_frames[0]
        gallery_name = f"0001_c2_001_{original_frame}"
        shutil.copy2(
            os.path.join(original_cropped_dir, original_frame),
            os.path.join(gallery_dir, gallery_name)
        )

def organize_batch_reid_data(cropped_base_dir, output_base_dir, seed=42):
    """
    Organize multiple video directories for batch re-ID evaluation
    
    Args:
        cropped_base_dir (str): Base directory containing cropped video subdirectories
        output_base_dir (str): Base directory for organized re-ID data
        seed (int): Random seed for reproducibility
    """
    random.seed(seed)
    
    # Process each video subdirectory
    for video_dir in os.listdir(cropped_base_dir):
        video_path = os.path.join(cropped_base_dir, video_dir)
        if os.path.isdir(video_path):
            original_cropped_dir = os.path.join(video_path, "original_cropped")
            anonymized_cropped_dir = os.path.join(video_path, "anonymized_cropped")
            
            if os.path.exists(original_cropped_dir) and os.path.exists(anonymized_cropped_dir):
                print(f"Organizing re-ID data for {video_dir}")
                output_dir = os.path.join(output_base_dir, video_dir)
                organize_reid_data(original_cropped_dir, anonymized_cropped_dir, output_dir)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Organize cropped frames for re-ID evaluation")
    parser.add_argument('--original_cropped_dir', type=str, required=True,
                       help='Directory with cropped original frames')
    parser.add_argument('--anonymized_cropped_dir', type=str, required=True,
                       help='Directory with cropped anonymized frames')
    parser.add_argument('--output_dir', type=str, default="datasets/reid_eval",
                       help='Output directory for organized re-ID data')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for reproducibility')
    parser.add_argument('--batch_mode', action='store_true',
                       help='Process multiple video directories in batch mode')
    parser.add_argument('--cropped_base_dir', type=str,
                       help='Base directory containing cropped video subdirectories (for batch mode)')
    parser.add_argument('--output_base_dir', type=str,
                       help='Base directory for organized re-ID data (for batch mode)')
    
    args = parser.parse_args()
    
    # Set random seed for reproducible shuffling
    random.seed(args.seed)
    
    # Organize the data for re-ID evaluation
    if args.batch_mode and args.cropped_base_dir and args.output_base_dir:
        # Batch processing mode
        os.makedirs(args.output_base_dir, exist_ok=True)
        organize_batch_reid_data(args.cropped_base_dir, args.output_base_dir, args.seed)
    else:
        # Single video processing mode
        organize_reid_data(
            args.original_cropped_dir,
            args.anonymized_cropped_dir,
            args.output_dir
        )
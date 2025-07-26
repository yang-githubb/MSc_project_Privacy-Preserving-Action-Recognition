import os
import shutil
import random
import argparse
from pathlib import Path

def organize_reid_data(original_cropped_dir, anonymized_cropped_dir, output_dir, 
                      frames_per_person=1, use_multiple_frames=False):
    """
    Organize cropped frames for re-ID evaluation (DeepPrivacy2 exact method)
    
    This function organizes the data as follows:
    - Puts ORIGINAL frames in query (camera c1) - keep query as original
    - Puts ANONYMIZED frames in gallery (camera c2) - anonymize gallery
    - Each query has exactly one matching gallery entry with same person ID
    - Uses standard re-ID naming: {person_id}_{camera_id}_{video_id}_{frame_number}.jpg
    - Tests if original identities can be matched in anonymized gallery
    - Lower scores = better anonymization (harder to match original to anonymized)
    
    Args:
        original_cropped_dir (str): Directory with cropped original frames
        anonymized_cropped_dir (str): Directory with cropped anonymized frames
        output_dir (str): Output directory for organized re-ID data
        frames_per_person (int): Number of frames to use per person (default: 1)
        use_multiple_frames (bool): Whether to use multiple frames per person (default: False)
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
    
    # Find matching frame numbers between original and anonymized
    # Extract frame numbers (e.g., "000063.jpg" -> "000063")
    anonymized_frame_numbers = set()
    for frame in anonymized_frames:
        frame_num = frame.split('.')[0]  # Remove extension
        anonymized_frame_numbers.add(frame_num)
    
    original_frame_numbers = set()
    for frame in original_frames:
        frame_num = frame.split('.')[0]  # Remove extension
        original_frame_numbers.add(frame_num)
    
    # Find common frame numbers
    common_frames = anonymized_frame_numbers.intersection(original_frame_numbers)
    common_frames = sorted(list(common_frames))
    
    print(f"Found {len(common_frames)} matching frames between original and anonymized")
    
    # Select frames to use (either single frame per person or multiple)
    if use_multiple_frames:
        selected_frames = common_frames[:frames_per_person]
    else:
        # Use a representative frame (e.g., middle frame)
        if len(common_frames) > 0:
            mid_index = len(common_frames) // 2
            selected_frames = [common_frames[mid_index]]
        else:
            selected_frames = []
    
    print(f"Using {len(selected_frames)} frames for evaluation")
    
    # Organize data with DeepPrivacy2 approach: ORIGINAL in query, ANONYMIZED in gallery
    person_id = 1
    video_id = "video001"  # Since we're using one video
    
    for frame_num in selected_frames:
        # Find the original frame with this number
        original_frame = None
        for frame in original_frames:
            if frame.startswith(frame_num):
                original_frame = frame
                break
        
        # Find the anonymized frame with this number
        anonymized_frame = None
        for frame in anonymized_frames:
            if frame.startswith(frame_num):
                anonymized_frame = frame
                break
        
        if original_frame and anonymized_frame:
            # Query: ORIGINAL frame (camera c1) - DeepPrivacy2 approach
            query_name = f"{person_id:04d}_c1_{video_id}_{frame_num}.jpg"
            shutil.copy2(
                os.path.join(original_cropped_dir, original_frame),
                os.path.join(query_dir, query_name)
            )
            print(f"Query (ORIGINAL): {query_name}")
            
            # Gallery: ANONYMIZED frame (camera c2) - DeepPrivacy2 approach
            gallery_name = f"{person_id:04d}_c2_{video_id}_{frame_num}.jpg"
            shutil.copy2(
                os.path.join(anonymized_cropped_dir, anonymized_frame),
                os.path.join(gallery_dir, gallery_name)
            )
            print(f"Gallery (ANONYMIZED): {gallery_name}")
            
            person_id += 1

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
    parser.add_argument('--frames_per_person', type=int, default=1,
                       help='Number of frames to use per person (default: 1)')
    parser.add_argument('--use_multiple_frames', action='store_true',
                       help='Use multiple frames per person instead of single representative frame')
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
            args.output_dir,
            args.frames_per_person,
            args.use_multiple_frames
        )
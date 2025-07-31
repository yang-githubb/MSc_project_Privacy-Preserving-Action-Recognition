import cv2
import os
import argparse
from pathlib import Path

def extract_frames(video_path, output_dir, frame_interval=1):
    """
    Extract frames from video at specified interval
    
    Args:
        video_path (str): Path to input video file
        output_dir (str): Directory to save extracted frames
        frame_interval (int): Extract every Nth frame (default: 1 = every frame)
    """
    # Open video file
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return
    
    frame_count = 0
    saved_count = 0
    
    # Process video frame by frame
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        # Extract frame at specified interval
        if frame_count % frame_interval == 0:
            frame_filename = f"{saved_count:06d}.jpg"
            frame_path = os.path.join(output_dir, frame_filename)
            cv2.imwrite(frame_path, frame)
            saved_count += 1
            
        frame_count += 1
    
    cap.release()

def get_video_name(video_path):
    """
    Extract video name from path for organizing output directories
    
    Args:
        video_path (str): Path to video file
        
    Returns:
        str: Video name (e.g., 'Archery_g01_c01' from 'v_Archery_g01_c01.avi')
    """
    # Extract filename without extension
    filename = os.path.basename(video_path)
    name_without_ext = os.path.splitext(filename)[0]
    
    # Remove common prefixes like 'v_' for video files
    if name_without_ext.startswith('v_'):
        name_without_ext = name_without_ext[2:]
    
    # Remove '_anonymized' suffix if present
    if name_without_ext.endswith('_anonymized'):
        name_without_ext = name_without_ext[:-12]
    
    return name_without_ext

def process_video_pairs(original_video, anonymized_video, output_base_dir):
    """
    Process a pair of original and anonymized videos by extracting frames
    
    Args:
        original_video (str): Path to original video file
        anonymized_video (str): Path to anonymized video file
        output_base_dir (str): Base directory for output frames
    """
    # Extract video name for organizing output
    video_name = get_video_name(original_video)
    
    # Create output directories for original and anonymized frames
    video_output_dir = os.path.join(output_base_dir, video_name)
    original_frames_dir = os.path.join(video_output_dir, "original_frames")
    anonymized_frames_dir = os.path.join(video_output_dir, "anonymized_frames")
    
    os.makedirs(original_frames_dir, exist_ok=True)
    os.makedirs(anonymized_frames_dir, exist_ok=True)
    
    # Extract frames from both videos
    extract_frames(original_video, original_frames_dir)
    extract_frames(anonymized_video, anonymized_frames_dir)

def process_batch_videos(video_pairs, output_base_dir, frame_interval=1):
    """
    Process multiple video pairs for batch processing
    
    Args:
        video_pairs (list): List of tuples (original_video, anonymized_video)
        output_base_dir (str): Base directory for output frames
        frame_interval (int): Extract every Nth frame
    """
    for original_video, anonymized_video in video_pairs:
        print(f"Processing video pair: {original_video} and {anonymized_video}")
        process_video_pairs(original_video, anonymized_video, output_base_dir)

if __name__ == "__main__":
    import argparse
    import glob
    from pathlib import Path
    
    parser = argparse.ArgumentParser(description="Extract frames from videos")
    parser.add_argument('--input_dir', type=str, required=True,
                       help='Input directory containing videos')
    parser.add_argument('--output_dir', type=str, required=True,
                       help='Output directory for frames')
    parser.add_argument('--frame_interval', type=int, default=5,
                       help='Extract every Nth frame (default: 5 for efficiency)')
    parser.add_argument('--num_workers', type=int, default=8,
                       help='Number of parallel workers (default: 8)')
    parser.add_argument('--skip_existing', action='store_true',
                       help='Skip videos that already have frames extracted')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Find all video files
    video_extensions = ['*.avi', '*.mp4', '*.mov', '*.mkv']
    video_files = []
    
    for ext in video_extensions:
        video_files.extend(glob.glob(os.path.join(args.input_dir, '**', ext), recursive=True))
    
    if not video_files:
        print(f"No video files found in {args.input_dir}")
        exit(0)
    
    print(f"Found {len(video_files)} video files")
    
    # Check for existing work and skip if requested
    if args.skip_existing:
        skipped_count = 0
        for video_path in video_files:
            rel_path = os.path.relpath(video_path, args.input_dir)
            video_name = os.path.splitext(rel_path)[0]
            video_output_dir = os.path.join(args.output_dir, video_name)
            
            # Check if frames already exist
            if os.path.exists(video_output_dir) and len(os.listdir(video_output_dir)) > 0:
                print(f"Skipping {video_name} - frames already exist")
                skipped_count += 1
                video_files.remove(video_path)
        
        print(f"Skipped {skipped_count} videos with existing frames")
        print(f"Processing {len(video_files)} remaining videos")
    
    # Process remaining videos
    for video_path in video_files:
        rel_path = os.path.relpath(video_path, args.input_dir)
        video_name = os.path.splitext(rel_path)[0]
        video_output_dir = os.path.join(args.output_dir, video_name)
        
        print(f"Processing: {video_name}")
        os.makedirs(video_output_dir, exist_ok=True)
        
        # Extract frames
        extract_frames(video_path, video_output_dir, args.frame_interval)
    
    print(f"Frame extraction completed for {len(video_files)} videos")

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
    parser = argparse.ArgumentParser(description="Extract frames from original and anonymized videos")
    parser.add_argument('--original_video', type=str, 
                       default="/work/tc067/tc067/s2737744/Dataset/ucf101/UCF-101/Archery/v_Archery_g01_c01.avi",
                       help='Path to original video')
    parser.add_argument('--anonymized_video', type=str,
                       default="/work/tc067/tc067/s2737744/output/ucf101_anonymized/Archery/v_Archery_g01_c01_anonymized.mp4",
                       help='Path to anonymized video')
    parser.add_argument('--output_dir', type=str,
                       default="datasets/video_frames",
                       help='Output directory for frames')
    parser.add_argument('--frame_interval', type=int, default=1,
                       help='Extract every Nth frame (default: 1)')
    parser.add_argument('--batch_file', type=str,
                       help='Path to batch file containing video pairs (for future batch processing)')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # For now, process single video pair
    # In the future, this can be extended to process batch files
    if args.batch_file and os.path.exists(args.batch_file):
        # Future: Load video pairs from batch file
        # video_pairs = load_video_pairs_from_file(args.batch_file)
        # process_batch_videos(video_pairs, args.output_dir, args.frame_interval)
        pass
    else:
        # Process single video pair
        process_video_pairs(args.original_video, args.anonymized_video, args.output_dir)

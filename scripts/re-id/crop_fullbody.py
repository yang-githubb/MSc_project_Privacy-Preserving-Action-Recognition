import cv2
import numpy as np
import os
import argparse
from pathlib import Path
import torch
from torchvision import transforms
from PIL import Image

def load_pose_model():
    """
    Load pose estimation model (MediaPipe preferred, OpenCV DNN as fallback)
    
    Returns:
        tuple: (pose_model, model_type) where model_type is "mediapipe" or "opencv"
    """
    try:
        # Try to use MediaPipe for pose detection (more accurate)
        import mediapipe as mp
        mp_pose = mp.solutions.pose
        pose = mp_pose.Pose(
            static_image_mode=True,
            model_complexity=2,
            enable_segmentation=True,
            min_detection_confidence=0.5
        )
        return pose, "mediapipe"
    except ImportError:
        # Fallback to OpenCV DNN pose estimation
        cache_dir = os.path.join(os.getcwd(), '.cache')
        prototxt_path = os.path.join(cache_dir, 'pose_deploy.prototxt')
        model_path = os.path.join(cache_dir, 'pose_iter_584000.caffemodel')
        
        if not os.path.exists(prototxt_path) or not os.path.exists(model_path):
            print(f"Pose detection models not found in {cache_dir}")
            print("Please copy pose_deploy.prototxt and pose_iter_584000.caffemodel to the .cache directory")
            return None, "none"
        
        net = cv2.dnn.readNetFromCaffe(prototxt_path, model_path)
        return net, "opencv"

def load_segmentation_model():
    """
    Load segmentation model for background removal
    
    Returns:
        tuple: (seg_model, model_type) where model_type is "mediapipe" or "opencv"
    """
    try:
        # Try to use MediaPipe for segmentation (more accurate)
        import mediapipe as mp
        mp_selfie_segmentation = mp.solutions.selfie_segmentation
        segmentation = mp_selfie_segmentation.SelfieSegmentation(model_selection=1)
        return segmentation, "mediapipe"
    except ImportError:
        # Fallback to OpenCV DNN segmentation
        cache_dir = os.path.join(os.getcwd(), '.cache')
        prototxt_path = os.path.join(cache_dir, 'deeplabv3_xception65_ade20k.prototxt')
        model_path = os.path.join(cache_dir, 'deeplabv3_xception65_ade20k.caffemodel')
        
        if not os.path.exists(prototxt_path) or not os.path.exists(model_path):
            print(f"Segmentation models not found in {cache_dir}")
            print("Please copy deeplabv3_xception65_ade20k.prototxt and deeplabv3_xception65_ade20k.caffemodel to the .cache directory")
            return None, "none"
        
        net = cv2.dnn.readNetFromCaffe(prototxt_path, model_path)
        return net, "opencv"

def detect_pose_mediapipe(image, pose):
    """
    Detect pose using MediaPipe and return bounding box
    
    Args:
        image: Input image (BGR format)
        pose: MediaPipe pose model
        
    Returns:
        tuple: (x_min, y_min, x_max, y_max) bounding box or None if no pose detected
    """
    # Process image with MediaPipe
    results = pose.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    
    if results.pose_landmarks:
        landmarks = results.pose_landmarks.landmark
        h, w = image.shape[:2]
        
        # Extract coordinates from pose landmarks
        x_coords = [landmark.x * w for landmark in landmarks]
        y_coords = [landmark.y * h for landmark in landmarks]
        
        # Calculate bounding box
        x_min, x_max = min(x_coords), max(x_coords)
        y_min, y_max = min(y_coords), max(y_coords)
        
        # Add padding around the pose
        padding = 50
        x_min = max(0, int(x_min - padding))
        y_min = max(0, int(y_min - padding))
        x_max = min(w, int(x_max + padding))
        y_max = min(h, int(y_max + padding))
        
        return (x_min, y_min, x_max, y_max)
    
    return None

def detect_pose_opencv(image, net):
    """
    Detect pose using OpenCV DNN and return bounding box
    
    Args:
        image: Input image (BGR format)
        net: OpenCV DNN pose model
        
    Returns:
        tuple: (x_min, y_min, x_max, y_max) bounding box or None if no pose detected
    """
    # Prepare input blob for DNN
    blob = cv2.dnn.blobFromImage(image, 1.0/255, (368, 368), (0, 0, 0), swapRB=False, crop=False)
    net.setInput(blob)
    output = net.forward()
    
    # Extract keypoints from DNN output
    keypoints = []
    for i in range(output.shape[1]):
        prob_map = output[0, i, :, :]
        _, conf, _, point = cv2.minMaxLoc(prob_map)
        x = (point[0] * image.shape[1]) / output.shape[3]
        y = (point[1] * image.shape[0]) / output.shape[2]
        keypoints.append((x, y, conf))
    
    # Filter valid keypoints and calculate bounding box
    valid_points = [(x, y) for x, y, conf in keypoints if conf > 0.1]
    if valid_points:
        x_coords = [p[0] for p in valid_points]
        y_coords = [p[1] for p in valid_points]
        
        x_min, x_max = min(x_coords), max(x_coords)
        y_min, y_max = min(y_coords), max(y_coords)
        
        # Add padding around the pose
        padding = 50
        x_min = max(0, int(x_min - padding))
        y_min = max(0, int(y_min - padding))
        x_max = min(image.shape[1], int(x_max + padding))
        y_max = min(image.shape[0], int(y_max + padding))
        
        return (x_min, y_min, x_max, y_max)
    
    return None

def remove_background_mediapipe(image, seg_model):
    """
    Remove background using MediaPipe Selfie Segmentation and replace with white
    
    Args:
        image: Input image (BGR format)
        seg_model: MediaPipe segmentation model
        
    Returns:
        numpy.ndarray: Image with white background
    """
    # Convert BGR to RGB for MediaPipe
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Get segmentation mask
    results = seg_model.process(rgb_image)
    mask = results.segmentation_mask
    
    if mask is None:
        return image
    
    # Create white background
    white_background = np.ones_like(image) * 255
    
    # Convert mask to proper format
    mask = np.stack((mask,) * 3, axis=-1)
    
    # Combine foreground (person) with white background
    output_image = image * mask + white_background * (1 - mask)
    
    # Convert to uint8
    output_image = output_image.astype(np.uint8)
    
    return output_image

def remove_background_opencv(image, seg_model):
    """
    Remove background using OpenCV DNN segmentation and replace with white
    
    Args:
        image: Input image (BGR format)
        seg_model: OpenCV DNN segmentation model
        
    Returns:
        numpy.ndarray: Image with white background
    """
    # Prepare input blob
    blob = cv2.dnn.blobFromImage(image, 1.0/255, (513, 513), (0.485, 0.456, 0.406), swapRB=True, crop=False)
    seg_model.setInput(blob)
    output = seg_model.forward()
    
    # Get segmentation mask
    mask = output[0].argmax(axis=0)
    
    # Create binary mask (person vs background)
    person_mask = (mask == 15).astype(np.uint8)  # Class 15 is typically person
    
    # Resize mask to original image size
    h, w = image.shape[:2]
    person_mask = cv2.resize(person_mask, (w, h))
    
    # Create white background
    white_background = np.ones_like(image) * 255
    
    # Combine foreground (person) with white background
    output_image = image * person_mask[:, :, np.newaxis] + white_background * (1 - person_mask[:, :, np.newaxis])
    
    return output_image

def simple_person_crop(image):
    """
    Simple person cropping using basic image processing
    Assumes person is roughly in the center of the frame
    """
    h, w = image.shape[:2]
    
    # Define crop region (center portion of image)
    # This assumes the person is roughly in the center
    crop_w = int(w * 0.6)  # 60% of width
    crop_h = int(h * 0.8)  # 80% of height
    
    # Center the crop
    x_start = (w - crop_w) // 2
    y_start = (h - crop_h) // 2
    
    # Ensure crop is within image bounds
    x_start = max(0, x_start)
    y_start = max(0, y_start)
    x_end = min(w, x_start + crop_w)
    y_end = min(h, y_start + crop_h)
    
    return (x_start, y_start, x_end, y_end)

def crop_fullbody(image_path, output_path, pose_model, model_type, seg_model=None, seg_type="none", remove_bg=True):
    """
    Crop full-body pose from image, remove background, and save to output path
    
    Args:
        image_path (str): Path to input image
        output_path (str): Path to save cropped image
        pose_model: Pose detection model
        model_type (str): Type of model ("mediapipe", "opencv", or "none")
        seg_model: Segmentation model for background removal
        seg_type (str): Type of segmentation model ("mediapipe", "opencv", or "none")
        remove_bg (bool): Whether to remove background
        
    Returns:
        bool: True if pose was detected and cropped successfully, False otherwise
    """
    # Load image
    image = cv2.imread(image_path)
    if image is None:
        return False
    
    # Remove background if requested and model available
    if remove_bg and seg_model is not None:
        if seg_type == "mediapipe":
            image = remove_background_mediapipe(image, seg_model)
        elif seg_type == "opencv":
            image = remove_background_opencv(image, seg_model)
    
    # Detect pose using appropriate model
    if model_type == "mediapipe":
        bbox = detect_pose_mediapipe(image, pose_model)
    elif model_type == "opencv":
        bbox = detect_pose_opencv(image, pose_model)
    else:
        # Fallback to simple center crop
        bbox = simple_person_crop(image)
    
    if bbox is None:
        return False
    
    # Crop the image using detected bounding box
    x_min, y_min, x_max, y_max = bbox
    cropped = image[y_min:y_max, x_min:x_max]
    
    # Save cropped image
    cv2.imwrite(output_path, cropped)
    return True

def process_frames_directory(input_dir, output_dir, pose_model, model_type, seg_model=None, seg_type="none", remove_bg=True):
    """
    Process all frames in a directory by cropping full-body poses and removing background
    
    Args:
        input_dir (str): Directory containing input frames
        output_dir (str): Directory to save cropped poses
        pose_model: Pose detection model
        model_type (str): Type of model ("mediapipe", "opencv", or "none")
        seg_model: Segmentation model for background removal
        seg_type (str): Type of segmentation model ("mediapipe", "opencv", or "none")
        remove_bg (bool): Whether to remove background
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Supported image extensions
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
    processed_count = 0
    total_count = 0
    
    # Process each image file in the directory
    for filename in os.listdir(input_dir):
        if any(filename.lower().endswith(ext) for ext in image_extensions):
            total_count += 1
            input_path = os.path.join(input_dir, filename)
            output_path = os.path.join(output_dir, filename)
            
            if crop_fullbody(input_path, output_path, pose_model, model_type, seg_model, seg_type, remove_bg):
                processed_count += 1
    
    print(f"Processed {processed_count}/{total_count} frames in {input_dir}")

def process_batch_directories(input_base_dir, output_base_dir, pose_model, model_type):
    """
    Process multiple video directories for batch processing
    
    Args:
        input_base_dir (str): Base directory containing video subdirectories
        output_base_dir (str): Base directory for output cropped images
        pose_model: Pose detection model
        model_type (str): Type of model
    """
    # Process each video subdirectory
    for video_dir in os.listdir(input_base_dir):
        video_path = os.path.join(input_base_dir, video_dir)
        if os.path.isdir(video_path):
            # Process original frames
            original_input = os.path.join(video_path, "original_frames")
            original_output = os.path.join(output_base_dir, video_dir, "original_cropped")
            if os.path.exists(original_input):
                print(f"Processing original frames for {video_dir}")
                process_frames_directory(original_input, original_output, pose_model, model_type)
            
            # Process anonymized frames
            anonymized_input = os.path.join(video_path, "anonymized_frames")
            anonymized_output = os.path.join(output_base_dir, video_dir, "anonymized_cropped")
            if os.path.exists(anonymized_input):
                print(f"Processing anonymized frames for {video_dir}")
                process_frames_directory(anonymized_input, anonymized_output, pose_model, model_type)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Crop full-body poses from frames and remove background")
    parser.add_argument('--input_dir', type=str, required=True,
                       help='Input directory with frames')
    parser.add_argument('--output_dir', type=str, required=True,
                       help='Output directory for cropped poses')
    parser.add_argument('--remove_background', action='store_true', default=True,
                       help='Remove background from images (default: True)')
    parser.add_argument('--no_remove_background', action='store_true',
                       help='Disable background removal')
    parser.add_argument('--batch_mode', action='store_true',
                       help='Process multiple video directories in batch mode')
    parser.add_argument('--input_base_dir', type=str,
                       help='Base directory containing video subdirectories (for batch mode)')
    parser.add_argument('--output_base_dir', type=str,
                       help='Base directory for output (for batch mode)')
    
    args = parser.parse_args()
    
    # Determine if background removal should be enabled
    remove_bg = args.remove_background and not args.no_remove_background
    
    # Load pose model
    pose_model, model_type = load_pose_model()
    if model_type == "none":
        print("Using simple center crop as fallback")
    else:
        print(f"Using {model_type} for pose detection")
    
    # Load segmentation model if background removal is enabled
    seg_model = None
    seg_type = "none"
    if remove_bg:
        seg_model, seg_type = load_segmentation_model()
        if seg_type == "none":
            print("Warning: No segmentation model available, skipping background removal")
            remove_bg = False
        else:
            print(f"Using {seg_type} for background removal")
    
    # Process frames
    if args.batch_mode and args.input_base_dir and args.output_base_dir:
        # Batch processing mode
        os.makedirs(args.output_base_dir, exist_ok=True)
        process_batch_directories(args.input_base_dir, args.output_base_dir, pose_model, model_type)
    else:
        # Single directory processing mode
        process_frames_directory(args.input_dir, args.output_dir, pose_model, model_type, seg_model, seg_type, remove_bg) 
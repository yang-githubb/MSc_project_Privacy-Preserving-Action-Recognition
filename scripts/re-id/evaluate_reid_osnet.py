from torchreid.utils import FeatureExtractor
from torchreid.metrics import compute_distance_matrix, evaluate_rank
from PIL import Image
import torch
import os
import numpy as np
from torchvision import transforms

# Initialize OSNet feature extractor for re-identification
extractor = FeatureExtractor(
    model_name='osnet_x1_0',
    model_path=None,
    device='cuda' if torch.cuda.is_available() else 'cpu'
)

# Image preprocessing pipeline for OSNet
preprocess = transforms.Compose([
    transforms.Resize((256, 128)),  # OSNet input size
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],  # ImageNet normalization
                         [0.229, 0.224, 0.225])
])

def load_images_from_folder(folder):
    """
    Load images from folder and extract person IDs from filenames
    
    Args:
        folder (str): Directory containing images
        
    Returns:
        tuple: (images_tensor, person_ids) where images_tensor is batched tensor
    """
    images, ids = [], []
    for fname in sorted(os.listdir(folder)):
        # Load and preprocess image
        img = Image.open(os.path.join(folder, fname)).convert('RGB')
        images.append(preprocess(img).unsqueeze(0))
        # Extract person ID from filename (e.g., "0001_c1_001.jpg" -> 1)
        ids.append(int(fname.split('_')[0]))
    return torch.cat(images), ids

def extract_camids(filenames):
    """
    Extract camera IDs from filenames for re-ID evaluation
    
    Args:
        filenames (list): List of filenames
        
    Returns:
        list: Camera IDs extracted from filenames
    """
    camids = []
    for fname in filenames:
        parts = fname.split('_')
        for part in parts:
            # Look for camera ID pattern (e.g., "c1", "c2")
            if part.startswith('c') and part[1].isdigit():
                camids.append(int(part[1]))
                break
        else:
            # Default camera ID if not found
            camids.append(0)
    return camids

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Evaluate Identity Leakage after Anonymization using OSNet Re-ID.")
    parser.add_argument('--query_dir', type=str, default='datasets/reid_eval/query', 
                       help='Directory with ANONYMIZED query images')
    parser.add_argument('--gallery_dir', type=str, default='datasets/reid_eval/gallery', 
                       help='Directory with ORIGINAL gallery images')
    args = parser.parse_args()

    # Load query and gallery images
    query_filenames = sorted(os.listdir(args.query_dir))
    gallery_filenames = sorted(os.listdir(args.gallery_dir))

    # Extract images and person IDs
    query_imgs, query_ids = load_images_from_folder(args.query_dir)
    gallery_imgs, gallery_ids = load_images_from_folder(args.gallery_dir)

    # Extract camera IDs for re-ID evaluation
    q_camids = extract_camids(query_filenames)
    g_camids = extract_camids(gallery_filenames)

    # Extract features using OSNet
    query_feats = extractor(query_imgs)
    gallery_feats = extractor(gallery_imgs)

    # Compute distance matrix and evaluate re-identification performance
    distmat = compute_distance_matrix(query_feats, gallery_feats, metric='cosine')
    cmc, mAP = evaluate_rank(distmat, query_ids, gallery_ids, q_camids, g_camids)

    # Calculate privacy metrics
    rank1_acc = cmc[0] * 100  # Rank-1 accuracy (percentage)
    map_score = mAP * 100     # Mean Average Precision (percentage)
    
    # Output results
    print(f"Rank-1 Accuracy: {rank1_acc:.2f}%")
    print(f"mAP Score: {map_score:.2f}%")
    
    # Analyze original frame performance
    original_frame_ranks = []
    original_frame_distances = []
    
    for q_idx, query_filename in enumerate(query_filenames):
        query_distances = distmat[q_idx]
        
        # Find the original frame (c2) in gallery
        original_frame_found = False
        for g_idx, gallery_filename in enumerate(gallery_filenames):
            if gallery_filename.split('_')[1] == 'c2':  # Original frame
                original_index = g_idx
                original_filename = gallery_filename
                original_distance = query_distances[g_idx].item()
                original_frame_found = True
                break
        
        # Find rank of the original frame
        if original_frame_found:
            sorted_indices = torch.argsort(query_distances)
            original_rank = (sorted_indices == original_index).nonzero(as_tuple=True)[0].item() + 1
            
            original_frame_ranks.append(original_rank)
            original_frame_distances.append(original_distance)
            
            print(f"Original Frame Rank: {original_rank}")
            print(f"Distance to original frame ({original_filename}): {original_distance:.4f}")
    
    # Calculate R1 and mAP for original frame specifically
    if original_frame_ranks:
        # R1: Percentage of queries where original frame is at rank 1
        r1_original = sum(1 for rank in original_frame_ranks if rank == 1) / len(original_frame_ranks) * 100
        
        # mAP: Mean Average Precision for original frame
        # For single query, mAP = 1/rank if original frame is found, 0 otherwise
        map_original = sum(1.0/rank for rank in original_frame_ranks) / len(original_frame_ranks) * 100
        
        print(f"\nOriginal Frame R1: {r1_original:.2f}%")
        print(f"Original Frame mAP: {map_original:.2f}%")
    
import torch
import numpy as np
from pathlib import Path
from typing import Tuple, List, Dict
import sys

# Extend module search path to enable access to parent directory utilities
parent_dir = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(parent_dir))
from util import get_list_distances_from_preds


def load_inliers_and_labels(
    base_path: str,
    vpr_model: str,
    matcher: str,
    dataset: str,
    threshold_dist: float = 25.0
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Retrieve inlier correspondence counts and reference ground-truth labels for training dataset.
    Integrates image matching results with geographic distance metrics for performance labeling.
    
    Args:
        base_path: Base path to VPR-methods-evaluation directory
        vpr_model: VPR model name (e.g., 'netvlad', 'cosplace')
        matcher: Matcher name (e.g., 'loftr', 'superglue')
        dataset: Dataset name (e.g., 'svox_sun')
        threshold_dist: Distance threshold in meters for correctness (default 25m)
    
    Returns:
        X: Array of inliers_top1 values
        y: Array of correctness labels (1 = correct, 0 = wrong)
    """
    
    X = []  # inliers_top1
    y = []  # is_correct
    
    # Define directory paths for image matching results and prediction metadata
    torch_dir = Path(base_path) / "training_logs" / f"{vpr_model}_image_matching" / matcher / dataset
    preds_dir = Path(base_path) / "training_logs" / f"{vpr_model}_prediction" / dataset / "preds"
    
    print(f"\n[Loading] {vpr_model} + {matcher} from {dataset}")
    print(f"Torch files: {torch_dir}")
    print(f"Predictions: {preds_dir}")
    
    if not torch_dir.exists():
        print(f"Torch directory not found: {torch_dir}")
        return np.array(X), np.array(y)
    
    if not preds_dir.exists():
        print(f"Predictions directory not found: {preds_dir}")
        return np.array(X), np.array(y)
    
    # Enumerate serialized result artifacts with one file per query instance
    torch_files = sorted(torch_dir.glob("*.torch"))
    
    if not torch_files:
        print(f"No torch files found in {torch_dir}")
        return np.array(X), np.array(y)
    
    print(f"Found {len(torch_files)} queries")
    
    count_loaded = 0
    for torch_file in torch_files:
        # Parse query identifier from serialized filename convention
        query_id = torch_file.stem
        
        # Locate associated prediction metadata file
        txt_file = preds_dir / f"{query_id}.txt"
        
        if not txt_file.exists():
            continue
        
        try:
            # Deserialize image matching result artifact containing correspondence statistics
            results = torch.load(torch_file, weights_only=False)
            
            # Extract inlier correspondence count from primary database candidate match
            inliers_top1 = results[0]['num_inliers']
            
            # Parse geographic distance measurements from structured prediction file
            distances = get_list_distances_from_preds(str(txt_file))
            
            # Retrieve geographic distance metric of top-ranked database candidate
            geo_dist_top1 = distances[0]
            
            # Assign binary label indicating localization success based on distance threshold
            is_correct = 1 if geo_dist_top1 <= threshold_dist else 0
            
            X.append(inliers_top1)
            y.append(is_correct)
            count_loaded += 1
            
        except Exception as e:
            print(f"Error processing query {query_id}: {e}")
            continue
    
    print(f"Loaded {count_loaded} queries")
    
    X = np.array(X)
    y = np.array(y)
    
    print(f"\n[Summary] Dataset statistics aggregation completed: {len(X)} samples")
    print(f"  Correct predictions: {sum(y)} ({100*sum(y)/len(y):.1f}%)")
    print(f"  Incorrect predictions: {len(y)-sum(y)} ({100*(len(y)-sum(y))/len(y):.1f}%)")
    
    return X, y


def get_inliers_statistics(X: np.ndarray, y: np.ndarray) -> Dict:
    """  
    Compute comprehensive distributional statistics for inlier correspondence counts.
    Stratifies analysis by prediction outcome (correct vs. incorrect) to characterize
    feature-label relationship characteristics.
    
    Args:
        X: Array of inliers values
        y: Array of correctness labels
    
    Returns:
        Dictionary with stratified statistical summaries
    """
    
    # Partition feature vectors according to outcome classification
    X_correct = X[y == 1]
    X_wrong = X[y == 0]
    
    # Aggregate comprehensive descriptive statistics across stratified subsets
    stats = {
        'correct': {
            'count': len(X_correct),
            'mean': np.mean(X_correct),
            'std': np.std(X_correct),
            'min': np.min(X_correct),
            'max': np.max(X_correct),
            'p25': np.percentile(X_correct, 25),
            'p50': np.percentile(X_correct, 50),
            'p75': np.percentile(X_correct, 75),
        },
        'wrong': {
            'count': len(X_wrong),
            'mean': np.mean(X_wrong),
            'std': np.std(X_wrong),
            'min': np.min(X_wrong),
            'max': np.max(X_wrong),
            'p25': np.percentile(X_wrong, 25),
            'p50': np.percentile(X_wrong, 50),
            'p75': np.percentile(X_wrong, 75),
        }
    }
    
    return stats




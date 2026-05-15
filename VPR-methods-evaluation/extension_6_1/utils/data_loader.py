"""Data loader for Extension 6.1 - Load inliers and labels from .torch and .txt files"""

import torch
import numpy as np
from pathlib import Path
from typing import Tuple, List, Dict
import sys

# Add parent directory to path to access util.py
parent_dir = Path(__file__).parent.parent.parent.parent.parent
sys.path.insert(0, str(parent_dir))
from util import get_list_distances_from_preds


def load_inliers_and_labels(
    base_path: str,
    vpr_model: str,
    matcher: str,
    datasets: List[str],
    threshold_dist: float = 25.0,
    top_k: int = 20
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load inliers_top1 and correctness labels from training datasets.
    
    Args:
        base_path: Base path to VPR-methods-evaluation directory
        vpr_model: VPR model name (e.g., 'netvlad', 'cosplace')
        matcher: Matcher name (e.g., 'loftr', 'superglue')
        datasets: List of dataset names (e.g., ['svox_sun', 'svox_night'])
        threshold_dist: Distance threshold in meters for correctness (default 25m)
        top_k: Number of top predictions to consider (default 20)
    
    Returns:
        X: Array of inliers_top1 values
        y: Array of correctness labels (1 = correct, 0 = wrong)
    """
    
    X = []  # inliers_top1
    y = []  # is_correct
    
    for dataset in datasets:
        # Paths
        torch_dir = Path(base_path) / "training_logs" / f"{vpr_model}_image_matching" / matcher / dataset
        preds_dir = Path(base_path) / "training_logs" / f"{vpr_model}_prediction" / dataset
        
        print(f"\n[Loading] {vpr_model} + {matcher} from {dataset}")
        print(f"Torch files: {torch_dir}")
        print(f"Predictions: {preds_dir}")
        
        if not torch_dir.exists():
            print(f"Torch directory not found: {torch_dir}")
            continue
        
        if not preds_dir.exists():
            print(f"Predictions directory not found: {preds_dir}")
            continue
        
        # List torch files (one per query)
        torch_files = sorted(torch_dir.glob("*.torch"))
        
        if not torch_files:
            print(f"No torch files found in {torch_dir}")
            continue
        
        print(f"Found {len(torch_files)} queries")
        
        count_loaded = 0
        for torch_file in torch_files:
            # Extract query ID from filename (e.g., "000.torch" → "0")
            query_id = torch_file.stem.split('_')[-1].lstrip('0') or '0'
            
            # Corresponding prediction file
            txt_file = preds_dir / f"{query_id}.txt"
            
            if not txt_file.exists():
                continue
            
            try:
                # Load .torch file (results from image matching)
                results = torch.load(torch_file, weights_only=False)
                
                # Extract inliers from top-1 match
                inliers_top1 = results[0]['num_inliers']
                
                # Load distances from prediction file
                distances = get_list_distances_from_preds(str(txt_file))
                
                # Get distance of top-1 prediction
                geo_dist_top1 = distances[0]
                
                # Label: is the top-1 prediction correct?
                is_correct = 1 if geo_dist_top1 <= threshold_dist else 0
                
                X.append(inliers_top1)
                y.append(is_correct)
                count_loaded += 1
                
            except Exception as e:
                print(f"    ⚠️  Error processing query {query_id}: {e}")
                continue
        
        print(f"  ✓ Loaded {count_loaded} queries")
    
    X = np.array(X)
    y = np.array(y)
    
    print(f"\n[Summary] Total loaded: {len(X)} queries")
    print(f"  Correct: {sum(y)} ({100*sum(y)/len(y):.1f}%)")
    print(f"  Wrong: {len(y)-sum(y)} ({100*(len(y)-sum(y))/len(y):.1f}%)")
    
    return X, y


def get_inliers_statistics(X: np.ndarray, y: np.ndarray) -> Dict:
    """
    Compute statistics of inliers distribution.
    
    Args:
        X: Array of inliers values
        y: Array of correctness labels
    
    Returns:
        Dictionary with statistics
    """
    
    X_correct = X[y == 1]
    X_wrong = X[y == 0]
    
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

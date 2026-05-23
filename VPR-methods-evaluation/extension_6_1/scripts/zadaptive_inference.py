"""
For each matcher (LoFTR, SuperGlue):
  For each test query:
    1. Run matching on top-1 database image
    2. Extract inliers_top1
    3. Predict P(correct) using matcher-specific LR model
    4. If P >= threshold[matcher]:
       └─ EASY query: use top-1 ranking, skip full matching
    5. Else:
       └─ HARD query: run full matching on top-20
    6. Calculate recall@1, @5, @10
    7. Measure timing (easy vs hard queries)

Outputs:
  - adaptive_inference_results_{matcher}.txt (metrics per dataset)
  - timing_breakdown_{matcher}.txt (detailed timing)
  - query_decisions_{matcher}.txt (easy/hard per query)
"""

import os
import sys
import json
import pickle
import time
import argparse
from pathlib import Path
from collections import defaultdict
from copy import deepcopy
import numpy as np
from PIL import Image

from matching import get_matcher, available_models
from matching.utils import get_default_device

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))
from util import get_list_distances_from_preds

# Import config
config_path = Path(__file__).parent.parent / "config" / "paths_config.json"
with open(config_path, 'r') as f:
    cfg = json.load(f)

BASE_PATH = cfg['input']['base_path']
TRAINING_LOGS_DIR = cfg['input']['training_logs_dir']
TESTING_LOGS_DIR = cfg['input']['testing_logs_dir']
MATCHERS = cfg['matchers']
TEST_DATASETS = cfg['input']['test_datasets']
THRESHOLD_DIST = cfg['hyperparams']['threshold_dist']
TOP_K = cfg['hyperparams']['top_k']

# Paths
RESULTS_DIR = Path(BASE_PATH) / cfg['output']['base_dir']
MODELS_DIR = RESULTS_DIR / cfg['output']["lr_models"]
THRESHOLD_DIR = RESULTS_DIR / cfg['output']['th_analysis']
INFERENCE_DIR = RESULTS_DIR / cfg['output']['inference']
INFERENCE_DIR.mkdir(parents=True, exist_ok=True)

# VPR models and matchers to test
VPR_MODELS = cfg['vpr_models']


def detect_path_mapping():
    """
    Auto-detect path mapping from config and system.
    Returns (old_prefix, new_prefix) based on BASE_PATH location.
    """
    base_path_str = str(BASE_PATH)
    
    # Check if we're on Windows (local path)
    if "\\" in base_path_str or "C:" in base_path_str or "D:" in base_path_str:
        # We're on Windows, default mapping should be TeamSpace -> Windows
        old_prefix = "/teamspace/studios/this_studio/Visual_Place_Recognition_Project/data/"
        # Extract Windows data path from BASE_PATH structure
        data_path = Path(BASE_PATH).parent.parent / "data"
        new_prefix = str(data_path)
        return old_prefix, new_prefix
    else:
        # We're on Linux/TeamSpace
        old_prefix = "C:\\Users\\leozi\\Desktop\\uni\\Magi\\AML\\Visual_Place_Recognition\\data"
        new_prefix = "/teamspace/studios/this_studio/Visual_Place_Recognition_Project/data"
        return old_prefix, new_prefix


def convert_path(path, old_prefix, new_prefix):
    """
    Convert a path from old_prefix to new_prefix.
    Handles both Windows and Unix path separators.
    """
    if not path:
        return path
    
    # Normalize the path for comparison
    path_normalized = path.replace("\\", "/")
    old_normalized = old_prefix.replace("\\", "/")
    
    # Check if path starts with old prefix
    if path_normalized.startswith(old_normalized):
        # Extract the relative part
        relative_part = path_normalized[len(old_normalized):].lstrip("/")
        
        # Build new path with proper separators
        if "\\" in new_prefix or ":" in new_prefix:
            # Target is Windows path
            new_path = str(Path(new_prefix) / relative_part.replace("/", "\\"))
        else:
            # Target is Unix/TeamSpace path
            new_path = new_prefix.rstrip("/") + "/" + relative_part
        
        return new_path
    
    # Path doesn't match old prefix, return as-is
    return path


def load_lr_models():
    """Load trained LR models from step train_lr."""
    models_file = MODELS_DIR / "lr_models.pkl"
    with open(models_file, 'rb') as f:
        models = pickle.load(f)
    return models


def load_optimal_thresholds():
    """Load optimal thresholds from step threshold_analysis."""
    thresholds_file = THRESHOLD_DIR / "optimal_thresholds.json"
    with open(thresholds_file, 'r') as f:
        thresholds = json.load(f)
    return thresholds


def parse_preds_file(preds_file_path, old_prefix, new_prefix):
    """Parse a single preds.txt file to extract top-k rankings."""
    predictions = []
    positives = []
    
    with open(preds_file_path, 'r') as f:
        lines = f.readlines()
    
    i = 0
    # Skip to "Predictions paths:" section
    while i < len(lines) and "Predictions paths:" not in lines[i]:
        i += 1
    i += 1  # Skip the header line
    
    # Read predictions (top-20)
    while i < len(lines) and lines[i].strip() and "Positives paths:" not in lines[i]:
        path = lines[i].strip()
        if path:
            path = convert_path(path, old_prefix, new_prefix)
            predictions.append(path)
        i += 1
    
    # Skip to "Positives paths:" section
    while i < len(lines) and "Positives paths:" not in lines[i]:
        i += 1
    i += 1  # Skip the header line
    
    # Read positives
    while i < len(lines) and lines[i].strip():
        path = lines[i].strip()
        if path:
            path = convert_path(path, old_prefix, new_prefix)
            positives.append(path)
        i += 1
    
    return predictions[:TOP_K], positives


def run_image_matching(query_img_path, db_img_path, matcher_name, img_size=512):
    """
    Run image matching between query and database image.
    Returns: number of inliers (keypoint matches after geometric verification)
    """
    try:
        if matcher_name.lower() == 'loftr':
            matcher = get_matcher('loftr', device=get_default_device())
        elif matcher_name.lower() == 'superglue':
            matcher = get_matcher('superglue', device=get_default_device())
        else:
            raise ValueError(f"Unknown matcher: {matcher_name}")
        
        img0 = matcher.load_image(query_img_path, resize=img_size)
        img1 = matcher.load_image(db_img_path, resize=img_size)
        
        result = matcher(deepcopy(img0), img1)
        
        inliers = result.get('num_inliers', 0)
        
        return inliers
    
    except Exception as e:
        print(f"  [ERROR] Matching failed: {e}")
        return 0


def is_correct(predicted_idx, preds_file_path, threshold_dist=THRESHOLD_DIST):
    """
    Check if predicted database index corresponds to a correct match.
    Uses geographic distance from prediction file.
    """
    if predicted_idx < 0 or predicted_idx >= TOP_K:
        return False
    
    distances = get_list_distances_from_preds(str(preds_file_path))
    
    geo_dist = distances[predicted_idx]
    return 1 if geo_dist <= threshold_dist else 0


def calculate_recalls(preds_file_path, top_k_list=[1, 5, 10], threshold_dist=THRESHOLD_DIST):
    """Calculate recall@k metrics."""
    distances = get_list_distances_from_preds(str(preds_file_path))
    
    # Count total correct predictions (all with distance <= threshold)
    total_correct = sum(1 for d in distances if d <= threshold_dist)
    
    if total_correct == 0:
        return {f'recall@{k}': 0.0 for k in top_k_list}
    
    recalls = {}
    for k in top_k_list:
        correct_at_k = sum(1 for i in range(min(k, len(distances)))
                          if distances[i] <= threshold_dist)
        recalls[f'recall@{k}'] = correct_at_k / total_correct
    return recalls


def process_matcher(matcher_name, lr_models, thresholds, old_prefix, new_prefix):
    """Process inference for a single matcher."""
    print(f"\n{'='*90}")
    print(f"Processing matcher: {matcher_name.upper()}")
    print(f"{'='*90}")
    
    results_per_dataset = {}
    query_decisions = []  # For logging easy/hard decisions
    
    threshold = thresholds[matcher_name]['threshold']
    print(f"  Optimal threshold: {threshold:.2f}")
    print(f"  Expected easy queries: {thresholds[matcher_name]['expected_easy_pct']:.1f}%\n")
    
    for dataset in TEST_DATASETS:
        print(f"\n  Processing dataset: {dataset}")
        preds_dir = Path(BASE_PATH) / TESTING_LOGS_DIR / f"{VPR_MODELS[0]}_prediction" / dataset / "preds"
        
        if not preds_dir.exists():
            print(f"    [WARN] Preds directory not found: {preds_dir}")
            print(f"    [INFO] Make sure testing_logs is populated with structure:")
            print(f"           {TESTING_LOGS_DIR}/")
            print(f"             ├─ {VPR_MODELS[0]}_prediction/")
            print(f"             │  └─ {dataset}/")
            print(f"             │     └─ preds/")
            print(f"             │        └─ *.txt")
            continue
        
        # Get all preds files
        preds_files = sorted(preds_dir.glob("*.txt"))
        print(f"    Found {len(preds_files)} query files")
        
        # Metrics accumulators
        metrics = {
            'total_queries': 0,
            'easy_queries': 0,
            'hard_queries': 0,
            'recall@1': 0,
            'recall@5': 0,
            'recall@10': 0,
            'time_easy': 0.0,
            'time_hard': 0.0,
            'count_easy': 0,
            'count_hard': 0,
        }
        
        query_times = []
        
        for query_idx, preds_file in enumerate(preds_files):
            if (query_idx + 1) % 100 == 0:
                print(f"Progress: {query_idx + 1}/{len(preds_files)}")
            
            try:
                # Parse preds file
                predictions, positives = parse_preds_file(preds_file, old_prefix, new_prefix)
                
                if not predictions or not positives:
                    continue
                
                metrics['total_queries'] += 1
                query_start_time = time.time()
                
                # === STEP 1: Run matching on top-1 ===
                query_path = None
                with open(preds_file, 'r') as f:
                    lines = f.readlines()
                    for i, line in enumerate(lines):
                        if "Query path:" in line:
                            # Try to extract path from same line
                            query_path = line.split("Query path:")[1].strip()
                            # If empty, it might be on the next line
                            if not query_path and i + 1 < len(lines):
                                query_path = lines[i + 1].strip()
                            query_path = convert_path(query_path, old_prefix, new_prefix)
                            break
                
                if not query_path or not os.path.exists(query_path):
                    continue
                
                top1_path = predictions[0]
                if not os.path.exists(top1_path):
                    continue
                
                # Run matching on top-1 (measure timing)
                match_start = time.time()
                inliers_top1 = run_image_matching(query_path, top1_path, matcher_name)
                match_time_top1 = time.time() - match_start
                
                # === STEP 2: Predict with LR ===
                lr_model = lr_models[matcher_name]
                X_test = np.array([[inliers_top1]])
                prob_correct = lr_model.predict_proba(X_test)[0][1]  # P(correct) 
                
                # === STEP 3: Make decision ===
                is_easy = prob_correct >= threshold
                
                if is_easy:
                    # EASY: use top-1, skip full matching
                    metrics['easy_queries'] += 1
                    metrics['count_easy'] += 1
                    final_ranking = predictions  # Keep original ranking
                    total_match_time = match_time_top1
                    metrics['time_easy'] += total_match_time
                    decision = "EASY"
                
                else:
                    # HARD: run full matching on top-20
                    metrics['hard_queries'] += 1
                    metrics['count_hard'] += 1
                    full_match_start = time.time()
                    
                    # Run full matching on all top-20 predictions
                    inliers_list = []
                    for pred_path in predictions:
                        if not os.path.exists(pred_path):
                            inliers_list.append(0)
                        else:
                            inliers = run_image_matching(query_path, pred_path, matcher_name)
                            inliers_list.append(inliers)
                    
                    # Re-rank predictions by inliers (descending order)
                    ranked_indices = np.argsort(inliers_list)[::-1]
                    final_ranking = [predictions[i] for i in ranked_indices]
                    
                    total_match_time = time.time() - full_match_start
                    metrics['time_hard'] += total_match_time
                    decision = "HARD"
                
                # === STEP 4: Calculate recalls ===
                recalls = calculate_recalls(preds_file, threshold_dist=THRESHOLD_DIST)
                metrics['recall@1'] += recalls['recall@1']
                metrics['recall@5'] += recalls['recall@5']
                metrics['recall@10'] += recalls['recall@10']
                
                query_times.append({
                    'query_id': query_idx,
                    'decision': decision,
                    'prob_correct': prob_correct,
                    'inliers_top1': inliers_top1,
                    'time': total_match_time,
                    'recall@1': recalls['recall@1'],
                })
                
            except Exception as e:
                print(f"[ERROR] Query {query_idx}: {e}")
                continue
        
        # === Compute final metrics ===
        if metrics['total_queries'] > 0:
            metrics['recall@1'] /= metrics['total_queries']
            metrics['recall@5'] /= metrics['total_queries']
            metrics['recall@10'] /= metrics['total_queries']
            
            avg_time_easy = (metrics['time_easy'] / metrics['count_easy'] 
                            if metrics['count_easy'] > 0 else 0)
            avg_time_hard = (metrics['time_hard'] / metrics['count_hard'] 
                            if metrics['count_hard'] > 0 else 0)
            
            easy_pct = 100 * metrics['easy_queries'] / metrics['total_queries']
            hard_pct = 100 * metrics['hard_queries'] / metrics['total_queries']
            
            # Calculate time savings
            baseline_time_per_query = (metrics['time_easy'] + metrics['time_hard']) / metrics['total_queries']
            time_saved_per_easy = baseline_time_per_query - avg_time_easy
            total_time_saved = time_saved_per_easy * metrics['easy_queries']
            
            results_per_dataset[dataset] = {
                'total_queries': metrics['total_queries'],
                'easy_queries': metrics['easy_queries'],
                'hard_queries': metrics['hard_queries'],
                'easy_pct': easy_pct,
                'hard_pct': hard_pct,
                'recall@1': metrics['recall@1'],
                'recall@5': metrics['recall@5'],
                'recall@10': metrics['recall@10'],
                'avg_time_easy': avg_time_easy,
                'avg_time_hard': avg_time_hard,
                'total_time_easy': metrics['time_easy'],
                'total_time_hard': metrics['time_hard'],
                'total_time_saved': total_time_saved,
                'baseline_time_per_query': baseline_time_per_query,
            }
            
            print(f"\n    Results for {dataset}:")
            print(f"      Total queries: {metrics['total_queries']}")
            print(f"      Easy queries: {metrics['easy_queries']} ({easy_pct:.1f}%)")
            print(f"      Hard queries: {metrics['hard_queries']} ({hard_pct:.1f}%)")
            print(f"      Recall@1: {metrics['recall@1']:.4f}")
            print(f"      Recall@5: {metrics['recall@5']:.4f}")
            print(f"      Recall@10: {metrics['recall@10']:.4f}")
            print(f"      Avg time (easy): {avg_time_easy:.4f}s")
            print(f"      Avg time (hard): {avg_time_hard:.4f}s")
            print(f"      Total time saved: {total_time_saved:.2f}s")
    
    return results_per_dataset


def main():
    old_prefix = globals().get('OLD_PREFIX')
    new_prefix = globals().get('NEW_PREFIX')
    
    print("\n" + "="*90)
    print("EXTENSION 6.1 - ADAPTIVE INFERENCE")
    print("="*90)
    
    # Check if testing logs directory exists
    testing_logs_path = Path(BASE_PATH) / TESTING_LOGS_DIR
    if not testing_logs_path.exists():
        print(f"\n[ERROR] Testing logs directory not found: {testing_logs_path}")
        print(f"\nPlease create the directory structure:")
        print(f"  {testing_logs_path}/")
        print(f"    ├─ {VPR_MODELS[0]}_prediction/")
        print(f"    │  ├─ tokyo/preds/")
        print(f"    │  ├─ sf_xs_test/preds/")
        print(f"    │  ├─ svox_sun_test/preds/")
        print(f"    │  └─ svox_night_test/preds/")
        print(f"    └─ {VPR_MODELS[1]}_prediction/")
        print(f"       ├─ tokyo/preds/")
        print(f"       ├─ sf_xs_test/preds/")
        print(f"       ├─ svox_sun_test/preds/")
        print(f"       └─ svox_night_test/preds/")
        print(f"\nEach preds/ folder should contain *.txt files with predictions.")
        return
    
    # Load models and thresholds
    print("\n[Loading] LR models and optimal thresholds...")
    lr_models = load_lr_models()
    thresholds = load_optimal_thresholds()
    
    print(f"  Matchers: {list(lr_models.keys())}")
    print(f"  Test datasets: {TEST_DATASETS}")
    print(f"  Testing logs dir: {TESTING_LOGS_DIR}")
    print(f"  Using prefixes: '{old_prefix}' -> '{new_prefix}'")
    
    # Process each matcher
    all_results = {}
    for matcher in MATCHERS:
        if matcher not in lr_models:
            print(f"\n[WARN] Matcher {matcher} not in trained models, skipping")
            continue
        
        results = process_matcher(matcher, lr_models, thresholds, old_prefix, new_prefix)
        all_results[matcher] = results
    
    # Summary results
    summary_file = INFERENCE_DIR / "adaptive_inference_results.txt"
    with open(summary_file, 'w') as f:
        f.write("="*90 + "\n")
        f.write("EXTENSION 6.1 - ADAPTIVE INFERENCE RESULTS\n")
        f.write("="*90 + "\n")
        
        for matcher, datasets_results in all_results.items():
            f.write(f"\n{'='*90}\n")
            f.write(f"MATCHER: {matcher.upper()}\n")
            f.write(f"Threshold: {thresholds[matcher]['threshold']:.2f}\n")
            f.write(f"Expected easy%: {thresholds[matcher]['expected_easy_pct']:.1f}%\n")
            f.write(f"{'='*90}\n\n")
            
            for dataset, metrics in datasets_results.items():
                f.write(f"\nDataset: {dataset}\n")
                f.write(f"{'-'*90}\n")
                f.write(f"  Total queries: {metrics['total_queries']}\n")
                f.write(f"  Easy queries: {metrics['easy_queries']} ({metrics['easy_pct']:.1f}%)\n")
                f.write(f"  Hard queries: {metrics['hard_queries']} ({metrics['hard_pct']:.1f}%)\n")
                f.write(f"\n  Recall@1:  {metrics['recall@1']:.4f}\n")
                f.write(f"  Recall@5:  {metrics['recall@5']:.4f}\n")
                f.write(f"  Recall@10: {metrics['recall@10']:.4f}\n")
                f.write(f"\n  Timing:\n")
                f.write(f"    Avg time (easy): {metrics['avg_time_easy']:.4f}s\n")
                f.write(f"    Avg time (hard): {metrics['avg_time_hard']:.4f}s\n")
                f.write(f"    Total time (easy): {metrics['total_time_easy']:.2f}s\n")
                f.write(f"    Total time (hard): {metrics['total_time_hard']:.2f}s\n")
                f.write(f"    Total time saved: {metrics['total_time_saved']:.2f}s\n")
                f.write(f"    Baseline time/query: {metrics['baseline_time_per_query']:.4f}s\n\n")
    
    # Save as JSON for easy parsing
    json_file = INFERENCE_DIR / "adaptive_inference_results.json"
    with open(json_file, 'w') as f:
        json.dump(all_results, f, indent=2)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Adaptive inference with configurable path mapping')
    parser.add_argument('--old-prefix', help='Old path prefix (source path to convert from)')
    parser.add_argument('--new-prefix', help='New path prefix (target path to convert to)')
    args = parser.parse_args()
    
    # Detect or use provided path mapping
    if args.old_prefix and args.new_prefix:
        old_prefix = args.old_prefix
        new_prefix = args.new_prefix
        print(f"\n[PATH MAPPING]")
        print(f"  Old prefix: {old_prefix}")
        print(f"  New prefix: {new_prefix}")
    else:
        old_prefix, new_prefix = detect_path_mapping()
        print(f"\n[AUTO-DETECTED PATH MAPPING]")
        print(f"  Old prefix: {old_prefix}")
        print(f"  New prefix: {new_prefix}")
    
    # Store in globals for use in main
    globals()['OLD_PREFIX'] = old_prefix
    globals()['NEW_PREFIX'] = new_prefix
    
    main()

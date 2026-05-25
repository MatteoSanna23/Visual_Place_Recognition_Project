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
from tqdm import tqdm

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
TRAINING_LOGS_DIR = cfg['input'].get('training_logs_dir', 'training_logs')
TESTING_LOGS_DIR = cfg['input'].get('testing_logs_dir', 'testing_logs')
MATCHERS = cfg['matchers']
TEST_DATASETS = cfg['input']['test_datasets']
THRESHOLD_DIST = cfg['hyperparams']['threshold_dist']
TOP_K = cfg['hyperparams']['top_k']

# Paths
RESULTS_DIR = Path(BASE_PATH) / cfg['output']['base_dir']
MODELS_DIR = RESULTS_DIR / cfg['output']['lr_models']
THRESHOLD_DIR = RESULTS_DIR / cfg['output']['th_analysis']
INFERENCE_DIR = RESULTS_DIR / cfg['output']['inference']
INFERENCE_DIR.mkdir(parents=True, exist_ok=True)

VPR_MODELS = cfg['vpr_models']


def detect_path_mapping():
    """
    Auto-detect path mapping from config and system.
    Returns (old_prefix, new_prefix) for path conversion.
    """
    base_path_str = str(BASE_PATH)
    
    # Check if we're on Windows (local path)
    if "\\" in base_path_str or "C:" in base_path_str or "D:" in base_path_str:
        # We're on Windows, convert from TeamSpace to Windows
        old_prefix = "/teamspace/studios/this_studio/data"
        data_path = Path(BASE_PATH).parent.parent / "data"
        new_prefix = str(data_path)
        return old_prefix, new_prefix
    else:
        # We're on TeamSpace
        old_prefix = "C:\\Users\\leozi\\Desktop\\uni\\Magi\\AML\\Visual_Place_Recognition\\data"
        new_prefix = "/teamspace/studios/this_studio/data"
        return old_prefix, new_prefix


def validate_path_mapping(old_prefix, new_prefix, test_preds_dir):
    """
    Validate path mapping by checking a sample preds file.
    Returns (is_valid, sample_original, sample_converted, exists)
    """
    preds_files = sorted(test_preds_dir.glob("*.txt"))
    if not preds_files:
        return False, None, None, False
    
    try:
        sample_file = preds_files[0]
        
        # Read the file
        with open(sample_file, 'r') as f:
            lines = f.readlines()
        
        # Find a sample prediction path
        original_in_file = None
        in_predictions = False
        for line in lines:
            if "Predictions paths:" in line:
                in_predictions = True
                continue
            if in_predictions:
                if line.strip() and "Positives" not in line:
                    original_in_file = line.strip()
                    break
        
        if not original_in_file:
            return False, None, None, False
        
        # Convert the path
        converted = convert_path(original_in_file, old_prefix, new_prefix)
        exists = os.path.exists(converted)
        
        return True, original_in_file, converted, exists
    
    except Exception as e:
        return False, None, None, False


def convert_path(path, old_prefix, new_prefix):
    """Convert a path from old_prefix to new_prefix.
    If path is already in new_prefix format, return as-is.
    """
    if not path:
        return path
    
    # Normalize paths for comparison
    path_normalized = path.replace("\\", "/")
    old_normalized = old_prefix.replace("\\", "/")
    new_normalized = new_prefix.replace("\\", "/")
    
    # Check if path is already in the target format
    if path_normalized.startswith(new_normalized):
        return path
    
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


def run_image_matching(query_img_loaded, db_img_path, matcher, img_size=512):
    """Run image matching between query and database image.
    
    Args:
        query_img_loaded: Pre-loaded query image tensor (or path if not loaded)
        db_img_path: Path to database image
        matcher: Pre-instantiated matcher object
        img_size: Image size for resizing
    """
    try:
        # Load query image if it's a path string
        if isinstance(query_img_loaded, str):
            img0 = matcher.load_image(query_img_loaded, resize=img_size)
        else:
            img0 = query_img_loaded
        
        # Load database image
        img1 = matcher.load_image(db_img_path, resize=img_size)
        
        result = matcher(deepcopy(img0), img1)
        inliers = result.get('num_inliers', 0)
        return inliers
    
    except Exception as e:
        print(f"    [ERROR] Matching failed: {e}")
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


def calculate_recalls(preds_file_path, top_k_list=[1, 5, 10], threshold_dist=THRESHOLD_DIST, distances=None):
    """Calculate recall@k metrics"""
    # Use provided distances or extract from file
    if distances is None:
        distances = get_list_distances_from_preds(str(preds_file_path))
    
    recalls = {}
    for k in top_k_list:
        has_correct = any(distances[i] <= threshold_dist for i in range(min(k, len(distances))))
        recalls[f'recall@{k}'] = 1.0 if has_correct else 0.0
    return recalls


def process_matcher(matcher_name, lr_models, thresholds, old_prefix, new_prefix):
    """Process inference for a single matcher."""
    print(f"\n{'='*90}")
    print(f"Processing matcher: {matcher_name.upper()}")
    print(f"{'='*90}")
    
    results_per_dataset = {}
    
    threshold = thresholds[matcher_name]['threshold']
    print(f"  Optimal threshold: {threshold:.2f}")
    print(f"  Expected easy queries: {thresholds[matcher_name]['expected_easy_pct']:.1f}%")
    
    # ========== LOAD MATCHER ONCE ==========
    print(f"  Loading {matcher_name.upper()} matcher...", end=" ", flush=True)
    try:
        if matcher_name.lower() == 'loftr':
            matcher_instance = get_matcher('loftr', device=get_default_device())
        elif matcher_name.lower() == 'superglue':
            matcher_instance = get_matcher('superglue', device=get_default_device())
        else:
            raise ValueError(f"Unknown matcher: {matcher_name}")
        print("✓\n")
    except Exception as e:
        print(f"[FAILED: {e}]")
        return results_per_dataset
    
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
        
        skip_reasons = defaultdict(int)
        
        # Process queries with progress bar
        preds_subset = preds_files  # Process all queries, no limit
        
        for preds_file in tqdm(preds_subset, desc=f"    {dataset}", leave=False, unit=" query"):
            
            try:
                # Parse preds file
                predictions, positives = parse_preds_file(preds_file, old_prefix, new_prefix)
                
                if not predictions or not positives:
                    skip_reasons['no_predictions_or_positives'] += 1
                    continue
                
                metrics['total_queries'] += 1
                
                # === Load distances from predictions file ===
                try:
                    original_distances = get_list_distances_from_preds(str(preds_file))
                except:
                    original_distances = [float('inf')] * len(predictions)
                
                # === STEP 1: Run matching on top-1 ===
                query_path = None
                with open(preds_file, 'r') as f:
                    lines = f.readlines()
                    for i, line in enumerate(lines):
                        if "Query path:" in line:
                            # Query path might be on the same line or next line
                            path_part = line.split("Query path:")[1].strip()
                            if path_part:
                                query_path = path_part
                            elif i + 1 < len(lines):
                                # Try next line
                                query_path = lines[i + 1].strip()
                            break
                
                if query_path:
                    query_path = convert_path(query_path, old_prefix, new_prefix)
                
                if not query_path or not os.path.exists(query_path):
                    skip_reasons['query_path_missing'] += 1
                    continue
                
                top1_path = predictions[0]
                if not os.path.exists(top1_path):
                    skip_reasons['top1_path_missing'] += 1
                    continue
                
                # Run matching on top-1 (measure timing)
                match_start = time.time()
                inliers_top1 = run_image_matching(query_path, top1_path, matcher_instance)
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
                    total_match_time = match_time_top1
                    metrics['time_easy'] += total_match_time
                    ranked_distances = original_distances
                    
                else:
                    # HARD: run full matching on top-20, re-rank by inliers
                    metrics['hard_queries'] += 1
                    metrics['count_hard'] += 1
                    full_match_start = time.time()
                    
                    # Pre-load query image once for batch matching
                    try:
                        img_size = 512
                        query_img_loaded = matcher_instance.load_image(query_path, resize=img_size)
                    except:
                        query_img_loaded = query_path  # Fallback to path-based loading
                    
                    inliers_list = []
                    # Progress bar for hard query matching (top-20)
                    for pred_path in tqdm(predictions, desc="        Matching top-20", leave=False, unit=" match", disable=len(predictions)<5):
                        if not os.path.exists(pred_path):
                            inliers_list.append(0)
                        else:
                            # Pass pre-loaded query image and matcher instance
                            inliers = run_image_matching(query_img_loaded, pred_path, matcher_instance)
                            inliers_list.append(inliers)
                    
                    # Re-rank by inliers (descending)
                    ranked_indices = np.argsort(inliers_list)[::-1]
                    ranked_distances = [original_distances[i] for i in ranked_indices]
                    
                    total_match_time = time.time() - full_match_start
                    metrics['time_hard'] += total_match_time
                
                # === STEP 4: Calculate recalls using ranked distances ===
                recalls = calculate_recalls(
                    preds_file, 
                    threshold_dist=THRESHOLD_DIST, 
                    distances=ranked_distances
                )
                metrics['recall@1'] += recalls['recall@1']
                metrics['recall@5'] += recalls['recall@5']
                metrics['recall@10'] += recalls['recall@10']
                
            except Exception as e:
                skip_reasons['exception'] += 1
                continue
        
        # === Compute final metrics ===
        if metrics['total_queries'] > 0:
            metrics['recall@1'] /= metrics['total_queries']
            metrics['recall@5'] /= metrics['total_queries']
            metrics['recall@10'] /= metrics['total_queries']
            
            avg_time_easy = metrics['time_easy'] / metrics['count_easy'] if metrics['count_easy'] > 0 else 0
            avg_time_hard = metrics['time_hard'] / metrics['count_hard'] if metrics['count_hard'] > 0 else 0
            
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
            
            status_msg = f"✓ R@1={metrics['recall@1']:.4f} | Easy={easy_pct:.1f}% | Queries={metrics['total_queries']}"
            if skip_reasons and metrics['easy_queries'] == 0 and metrics['hard_queries'] == 0:
                status_msg += f" | Skipped: {dict(skip_reasons)}"
            
            print(status_msg)
        else:
            print(f"[SKIP - no valid queries]")
    
    return results_per_dataset


def main():
    print("\n" + "="*90)
    print("EXTENSION 6.1 - ADAPTIVE INFERENCE (MATCHER-WISE)")
    print("="*90)
    
    # Detect path mapping
    old_prefix, new_prefix = detect_path_mapping()
    print(f"\n[PATH MAPPING]")
    print(f"  Old: {old_prefix}")
    print(f"  New: {new_prefix}")
    
    # Validate path mapping with a sample file
    print(f"\n[VALIDATING PATH MAPPING]")
    test_preds_dir = Path(BASE_PATH) / TESTING_LOGS_DIR / f"{VPR_MODELS[0]}_prediction" / TEST_DATASETS[0] / "preds"
    
    if test_preds_dir.exists():
        is_valid, orig, converted, exists = validate_path_mapping(old_prefix, new_prefix, test_preds_dir)
        if is_valid:
            print(f"  Original path in file: {orig}")
            print(f"  Converted to:          {converted}")
            print(f"  File exists: {'✓ YES' if exists else '✗ NO'}")
            
            if not exists:
                print(f"\n  [INFO] Path fix attempted in convert_path() - will retry during processing")
        else:
            print(f"  [WARNING] Could not validate path mapping - will attempt conversion during processing")
    else:
        print(f"  [WARNING] Test preds directory not found: {test_preds_dir}")
    
    # Check if testing logs directory exists
    testing_logs_path = Path(BASE_PATH) / TESTING_LOGS_DIR
    if not testing_logs_path.exists():
        print(f"\n[ERROR] Testing logs not found: {testing_logs_path}")
        print(f"Waiting for test predictions to be available...")
        return
    
    # Load models and thresholds
    print(f"\n[Loading] LR models and optimal thresholds...")
    try:
        lr_models = load_lr_models()
        thresholds = load_optimal_thresholds()
    except FileNotFoundError as e:
        print(f"[ERROR] Missing files: {e}")
        print("Make sure Steps 2-3 completed successfully")
        return
    
    print(f"✓ Loaded {len(lr_models)} models")
    print(f"✓ Loaded {len(thresholds)} threshold configs")
    print(f"  Matchers: {MATCHERS}")
    print(f"  Test datasets: {TEST_DATASETS}")
    
    # ======== MATCHER-WISE PROCESSING ========
    print(f"\n\n{'='*90}")
    print("ADAPTIVE INFERENCE PROCESSING")
    print(f"{'='*90}")
    
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
    main()

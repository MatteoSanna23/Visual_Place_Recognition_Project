"""
EXTENSION 6.1 - STEP 4: ADAPTIVE INFERENCE WITH DATASET TRANSFER ANALYSIS
===========================================================================
Adaptive image matching with dataset-specific models and transfer analysis.

For each training dataset (svox_sun, svox_night):
  For each matcher (loftr, superglue):
    For each test dataset (tokyo, sf_xs_test, svox_sun_test, svox_night_test):
      1. Load LR model trained on training_dataset
      2. Load threshold optimized on training_dataset
      3. For each test query:
         - Run matching on top-1
         - Predict P(correct) using LR
         - If P >= threshold: EASY → use top-1, skip full matching
         - Else: HARD → run full matching on top-20, re-rank by inliers
      4. Calculate Recall@1,5,10 using GEOGRAPHIC DISTANCES
      5. Measure timing and cost savings

Transfer Analysis: Shows how models from one dataset transfer to others
(e.g., svox_sun model on svox_night_test reveals environmental mismatch)

Output:
  - adaptive_inference_results_dataset_specific.txt (per dataset)
  - transfer_analysis.txt (cross-dataset performance)
  - adaptive_inference_results.json
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

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))
from util import get_list_distances_from_preds

# Import matching if available
try:
    from matching import get_matcher, available_models
    from matching.utils import get_default_device
    HAS_MATCHING = True
except ImportError:
    HAS_MATCHING = False
    print("[WARNING] Matching module not available - using placeholder matching")

# Import config
config_path = Path(__file__).parent.parent / "config" / "paths_config.json"
with open(config_path, 'r') as f:
    cfg = json.load(f)

BASE_PATH = cfg['input']['base_path']
TRAINING_LOGS_DIR = cfg['input'].get('training_logs_dir', 'training_logs')
TESTING_LOGS_DIR = cfg['input'].get('testing_logs_dir', 'testing_logs')
MATCHERS = cfg['matchers']
TEST_DATASETS = cfg['input']['test_datasets']
TRAINING_DATASETS = cfg['input']['training_datasets']
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
        old_prefix = "/teamspace/studios/this_studio/Visual_Place_Recognition_Project/data"
        data_path = Path(BASE_PATH).parent.parent / "data"
        new_prefix = str(data_path)
        return old_prefix, new_prefix
    else:
        # We're on TeamSpace
        # Paths might be either Windows format (from old generation) or already TeamSpace format
        old_prefix = "C:\\Users\\leozi\\Desktop\\uni\\Magi\\AML\\Visual_Place_Recognition\\data"
        new_prefix = "/teamspace/studios/this_studio/Visual_Place_Recognition_Project/data"
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
    
    # Check if path has incorrect TeamSpace prefix (missing Visual_Place_Recognition_Project)
    if "/teamspace/studios/this_studio/data/" in path_normalized:
        # Fix the path by inserting the missing project folder
        corrected = path_normalized.replace(
            "/teamspace/studios/this_studio/data/",
            "/teamspace/studios/this_studio/Visual_Place_Recognition_Project/data/"
        )
        if os.path.exists(corrected):
            return corrected
    
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


def load_lr_models_dataset_specific():
    """Load dataset-specific LR models from Step 2."""
    models_file = MODELS_DIR / "lr_models_dataset_specific.pkl"
    with open(models_file, 'rb') as f:
        models = pickle.load(f)
    return models


def load_optimal_thresholds_dataset_specific():
    """Load dataset-specific optimal thresholds from Step 3."""
    thresholds_file = THRESHOLD_DIR / "optimal_thresholds_dataset_specific.json"
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
        if not HAS_MATCHING:
            return np.random.randint(10, 150)
        
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


def calculate_recalls(preds_file_path, top_k_list=[1, 5, 10], threshold_dist=THRESHOLD_DIST, distances=None):
    """
    Calculate recall@k metrics using GEOGRAPHIC DISTANCES from preds file.
    """
    # Use provided distances or extract from file
    if distances is None:
        try:
            distances = get_list_distances_from_preds(str(preds_file_path))
        except Exception as e:
            print(f"      [WARNING] Could not extract distances: {e}")
            distances = []
    
    recalls = {}
    for k in top_k_list:
        # Recall@k = 1 if any of top-k has correct match (distance <= threshold)
        has_correct = any(distances[i] <= threshold_dist 
                         for i in range(min(k, len(distances))))
        recalls[f'recall@{k}'] = 1.0 if has_correct else 0.0
    
    return recalls


def process_inference(training_dataset, matcher_name, lr_models, thresholds, old_prefix, new_prefix):
    """
    Process inference for one training_dataset/matcher combination.
    Tests on all test_datasets to show transfer.
    """
    print(f"\n  {'─'*88}")
    print(f"  Dataset-Specific: {training_dataset.upper()} | Matcher: {matcher_name.upper()}")
    print(f"  {'─'*88}")
    
    model_key = f"{matcher_name}_{training_dataset}"
    threshold_key = f"{matcher_name}_{training_dataset}"
    
    if model_key not in lr_models:
        print(f"    [ERROR] Model not found: {model_key}")
        return None
    
    if threshold_key not in thresholds:
        print(f"    [ERROR] Threshold not found: {threshold_key}")
        return None
    
    lr_model = lr_models[model_key]
    threshold = thresholds[threshold_key]['threshold']
    
    print(f"    Threshold: {threshold:.2f}")
    
    # ========== LOAD MATCHER ONCE ==========
    print(f"    Loading {matcher_name.upper()} matcher...", end=" ", flush=True)
    try:
        if matcher_name.lower() == 'loftr':
            matcher_instance = get_matcher('loftr', device=get_default_device())
        elif matcher_name.lower() == 'superglue':
            matcher_instance = get_matcher('superglue', device=get_default_device())
        else:
            raise ValueError(f"Unknown matcher: {matcher_name}")
        print("✓")
    except Exception as e:
        print(f"[FAILED: {e}]")
        return None
    
    transfer_results = {}
    
    # Test on each test dataset
    for test_dataset in TEST_DATASETS:
        print(f"    Testing on {test_dataset.upper()}...", end=" ", flush=True)
        
        preds_dir = Path(BASE_PATH) / TESTING_LOGS_DIR / f"{VPR_MODELS[0]}_prediction" / test_dataset / "preds"
        
        if not preds_dir.exists():
            print(f"[SKIP - no data]")
            continue
        
        preds_files = sorted(preds_dir.glob("*.txt"))
        if not preds_files:
            print(f"[SKIP - empty]")
            continue
        
        # Metrics
        total_queries = 0
        easy_queries = 0
        hard_queries = 0
        recall_1 = 0.0
        recall_5 = 0.0
        recall_10 = 0.0
        time_easy = 0.0
        time_hard = 0.0
        skip_reasons = defaultdict(int)
        
        # Progress bar for query processing
        preds_subset = preds_files[:min(500, len(preds_files))]
        
        for preds_file in tqdm(preds_subset, desc=f"      {test_dataset}", leave=False, unit=" query"):
            try:
                predictions, positives = parse_preds_file(preds_file, old_prefix, new_prefix)
                if not predictions or not positives:
                    skip_reasons['no_predictions_or_positives'] += 1
                    continue
                
                total_queries += 1
                
                # Load query path
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
                
                # Get distances from predictions
                try:
                    original_distances = get_list_distances_from_preds(str(preds_file))
                except:
                    original_distances = [float('inf')] * len(predictions)
                
                # === STEP 1: Run matching on top-1 ===
                top1_path = predictions[0]
                if not os.path.exists(top1_path):
                    skip_reasons['top1_path_missing'] += 1
                    continue
                
                match_start = time.time()
                inliers_top1 = run_image_matching(query_path, top1_path, matcher_instance)
                match_time_top1 = time.time() - match_start
                
                # === STEP 2: Predict with LR ===
                X_test = np.array([[inliers_top1]])
                prob_correct = lr_model.predict_proba(X_test)[0][1]
                
                # === STEP 3: Decide easy vs hard ===
                if prob_correct >= threshold:
                    # EASY: use top-1, skip full matching
                    easy_queries += 1
                    time_easy += match_time_top1
                    ranked_distances = original_distances
                    
                else:
                    # HARD: run full matching on top-20, re-rank
                    hard_queries += 1
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
                    
                    time_hard += time.time() - full_match_start
                
                # === STEP 4: Calculate recalls ===
                recalls = calculate_recalls(
                    preds_file, 
                    threshold_dist=THRESHOLD_DIST, 
                    distances=ranked_distances
                )
                recall_1 += recalls['recall@1']
                recall_5 += recalls['recall@5']
                recall_10 += recalls['recall@10']
                
            except Exception as e:
                skip_reasons['exception'] += 1
                continue
        
        if total_queries > 0:
            recall_1 /= total_queries
            recall_5 /= total_queries
            recall_10 /= total_queries
            easy_pct = 100 * easy_queries / total_queries
            
            avg_time_easy = time_easy / easy_queries if easy_queries > 0 else 0
            avg_time_hard = time_hard / hard_queries if hard_queries > 0 else 0
            
            transfer_results[test_dataset] = {
                'total_queries': total_queries,
                'easy_queries': easy_queries,
                'hard_queries': hard_queries,
                'easy_pct': easy_pct,
                'recall@1': recall_1,
                'recall@5': recall_5,
                'recall@10': recall_10,
                'avg_time_easy': avg_time_easy,
                'avg_time_hard': avg_time_hard,
            }
            
            status_msg = f"✓ R@1={recall_1:.4f} | Easy={easy_pct:.1f}% | Queries={total_queries}"
            if skip_reasons and easy_queries == 0 and hard_queries == 0:
                status_msg += f" | Skipped: {dict(skip_reasons)}"
            
            print(status_msg)
        else:
            print(f"[SKIP - no valid queries]")
    
    return transfer_results


def main():
    print("\n" + "="*90)
    print("EXTENSION 6.1 - STEP 4: ADAPTIVE INFERENCE (DATASET-SPECIFIC)")
    print("="*90)
    
    # Check if testing logs exist
    testing_logs_path = Path(BASE_PATH) / TESTING_LOGS_DIR
    if not testing_logs_path.exists():
        print(f"\n[ERROR] Testing logs not found: {testing_logs_path}")
        print(f"Waiting for test predictions to be available...")
        return
    
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
    
    # Load models and thresholds
    print(f"\n[Loading] Dataset-specific models and thresholds...")
    try:
        lr_models = load_lr_models_dataset_specific()
        thresholds = load_optimal_thresholds_dataset_specific()
    except FileNotFoundError as e:
        print(f"[ERROR] Missing files: {e}")
        print("Make sure Steps 1-3 completed successfully")
        return
    
    print(f"✓ Loaded {len(lr_models)} models")
    print(f"✓ Loaded {len(thresholds)} threshold configs")
    print(f"  Training datasets: {TRAINING_DATASETS}")
    print(f"  Matchers: {MATCHERS}")
    print(f"  Test datasets: {TEST_DATASETS}")
    
    # ======== TRANSFER ANALYSIS ========
    print(f"\n\n{'='*90}")
    print("DATASET TRANSFER ANALYSIS")
    print(f"{'='*90}")
    print("Shows how models from each training dataset transfer to test datasets")
    
    all_results = {}
    
    for training_dataset in TRAINING_DATASETS:
        print(f"\n{'='*90}")
        print(f"Training Dataset: {training_dataset.upper()}")
        print(f"{'='*90}")
        
        for matcher_name in MATCHERS:
            results = process_inference(
                training_dataset, matcher_name, lr_models, thresholds, 
                old_prefix, new_prefix
            )
            if results:
                key = f"{matcher_name}_{training_dataset}"
                all_results[key] = results
    
    # ======== SAVE RESULTS ========
    print(f"\n\n{'='*90}")
    print("SAVING RESULTS")
    print(f"{'='*90}")
    
    # Create summary
    summary_lines = []
    summary_lines.append("="*100)
    summary_lines.append("EXTENSION 6.1 - STEP 4: ADAPTIVE INFERENCE (DATASET-SPECIFIC TRANSFER ANALYSIS)")
    summary_lines.append("="*100)
    summary_lines.append("")
    summary_lines.append("Analysis: How models from each training dataset perform on all test datasets")
    summary_lines.append("Reveals environmental variation impact on model transferability")
    summary_lines.append("")
    
    for training_dataset in TRAINING_DATASETS:
        summary_lines.append(f"\n{'='*100}")
        summary_lines.append(f"MODELS TRAINED ON: {training_dataset.upper()}")
        summary_lines.append(f"{'='*100}")
        
        for matcher in MATCHERS:
            key = f"{matcher}_{training_dataset}"
            if key not in all_results:
                continue
            
            threshold = thresholds[key]['threshold']
            summary_lines.append(f"\nMatcher: {matcher.upper()} | Threshold: {threshold:.2f}")
            summary_lines.append(f"{'Test Dataset':<20} {'Queries':<12} {'Easy%':<10} {'R@1':<10} {'R@5':<10} {'R@10':<10}")
            summary_lines.append(f"{'-'*80}")
            
            results = all_results[key]
            for test_dataset in sorted(results.keys()):
                metrics = results[test_dataset]
                summary_lines.append(
                    f"{test_dataset:<20} {metrics['total_queries']:<12} "
                    f"{metrics['easy_pct']:<10.1f} "
                    f"{metrics['recall@1']:<10.4f} "
                    f"{metrics['recall@5']:<10.4f} "
                    f"{metrics['recall@10']:<10.4f}"
                )
    
    summary_lines.append(f"\n{'='*100}")
    summary_lines.append("INTERPRETATION:")
    summary_lines.append("- Compare Recall@1 for same matcher across different training sources")
    summary_lines.append("- If Recall drops on certain test datasets, environmental mismatch exists")
    summary_lines.append("- This justifies dataset-specific model training approach")
    summary_lines.append(f"{'='*100}\n")
    
    summary_file = INFERENCE_DIR / "adaptive_inference_dataset_specific.txt"
    with open(summary_file, 'w') as f:
        f.write('\n'.join(summary_lines))
    
    print(f"\n✓ Transfer analysis saved: {summary_file}")
    
    # Save JSON
    json_file = INFERENCE_DIR / "adaptive_inference_dataset_specific.json"
    with open(json_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print(f"✓ JSON results saved: {json_file}")
    
    print(f"\n{'='*90}")
    print("STEP 4 COMPLETE")
    print(f"{'='*90}\n")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Adaptive inference with dataset-specific models')
    parser.add_argument('--old-prefix', help='Old path prefix (source)')
    parser.add_argument('--new-prefix', help='New path prefix (target)')
    args = parser.parse_args()
    
    main()

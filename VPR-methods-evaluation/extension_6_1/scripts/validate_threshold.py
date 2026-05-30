"""
Input:
  - lr_models.pkl (from train_lr.py)
  - Validation dataset (SF-XS val)

Output:
  - threshold_analysis.txt: Results for each dataset/matcher
  - threshold_curves.png: Plot of trade-off curves
  - optimal_thresholds.json: Best threshold for each dataset/matcher
"""

import json
import pickle
import numpy as np
from pathlib import Path
import sys
import matplotlib.pyplot as plt

# Add utils to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.data_loader import load_inliers_and_labels


def main():
    # Load configuration parameters from external configuration file
    config_path = Path(__file__).parent.parent / "config" / "paths_config.json"
    with open(config_path, encoding='utf-8') as f:
        cfg = json.load(f)
    
    base_path = cfg['input']['base_path']
    matchers = cfg['matchers']
    vpr_models = cfg['vpr_models']
    training_datasets = cfg['input']['training_datasets']
    val_dataset = cfg['input']['val_dataset']
    threshold_dist = cfg['hyperparams']['threshold_dist']
    score_weight_accuracy = cfg['hyperparams'].get('score_weight_accuracy', 0.5)
    
    # Define input directory for pre-trained logistic regression model artifacts
    models_dir = Path(base_path) / cfg['output']['base_dir'] / "lr_models"
    models_path = models_dir / "lr_models.pkl"
    
    if not models_path.exists():
        print(f"Models file not found: {models_path}")
        return
    
    # Deserialize pre-trained classifier ensemble from persistent storage
    with open(models_path, 'rb') as f:
        models = pickle.load(f)
    
    print(f"Loaded {len(models)} models: {list(models.keys())}")
    
    # Initialize output directory structure for threshold optimization results
    output_dir = Path(base_path) / cfg['output']['base_dir'] / "threshold_analysis"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Define probability threshold candidates for systematic sensitivity analysis
    thresholds = [0.50, 0.60, 0.70, 0.75, 0.80, 0.85, 0.90]
    
    summary_lines = []
    summary_lines.append("=" * 90)
    summary_lines.append("EXTENSION 6.1 - THRESHOLD VALIDATION (DATASET-SPECIFIC ANALYSIS)")
    summary_lines.append("=" * 90)
    summary_lines.append(f"Score formula: alpha × accuracy_on_easy + (1-alpha) × time_saved_pct")
    summary_lines.append(f"where alpha (score_weight_accuracy) = {score_weight_accuracy}")
    summary_lines.append("")
    
    results_by_dataset_matcher = {}
    optimal_thresholds = {}
    
    for dataset in training_datasets:
        
        print(f"Processing dataset: {dataset}")
        
        for matcher in matchers:
            print(f"\n  Matcher: {matcher.upper()}")
            
            # Retrieve pre-trained classifier associated with current dataset-matcher combination
            model_key = f"{matcher}_{dataset}"
            if model_key not in models:
                print(f"    Model not found for key: {model_key}")
                continue
            
            # Extract execution timing statistics from historical training logs
            timing_data = {'total_time': None, 'avg_time_per_query': None}
            for vpr_model in vpr_models:
                timing_file = Path(base_path) / "training_logs" / f"{vpr_model}_image_matching" / matcher / val_dataset / "timing_report.txt"
                if timing_file.exists():
                    with open(timing_file, 'r') as f:
                        for line in f:
                            if 'Total time:' in line:
                                timing_data['total_time'] = float(line.split(':')[1].split('seconds')[0].strip())
                            elif 'Average time per query:' in line:
                                timing_data['avg_time_per_query'] = float(line.split(':')[1].split('seconds')[0].strip())
                    print(f"    [Timing] avg={timing_data['avg_time_per_query']:.4f}s")
                    break
            
            if timing_data['total_time'] is None:
                print(f"    Warning: timing data not found, using defaults")
                timing_data['total_time'] = 1.0
                timing_data['avg_time_per_query'] = 1.0
            
            # Aggregate validation correspondence data from all available VPR models
            X_all = []
            y_all = []
            
            for vpr_model in vpr_models:
                try:
                    X, y = load_inliers_and_labels(
                        base_path=base_path,
                        vpr_model=vpr_model,
                        matcher=matcher,
                        dataset=val_dataset,
                        threshold_dist=threshold_dist
                    )
                    
                    X_all.extend(X)
                    y_all.extend(y)
                    
                except Exception as e:
                    print(f"    Error loading {vpr_model}: {e}")
                    continue
            
            if len(X_all) == 0:
                print(f"    No validation data for {matcher}")
                continue
            
            X_all = np.array(X_all)
            y_all = np.array(y_all)
            
            print(f"    Validation data: {len(X_all)} queries")
            
            # Retrieve trained logistic regression classifier for inference on validation set
            lr_model = models[model_key]
            
            # Generate probabilistic predictions on validation feature vectors
            X_reshaped = X_all.reshape(-1, 1)
            y_pred_proba = lr_model.predict_proba(X_reshaped)[:, 1]
                   
            threshold_results = []
            best_score = -np.inf
            best_threshold = 0.5
            best_threshold_idx = 0
            
            for idx, thresh in enumerate(thresholds):
                easy_mask = y_pred_proba >= thresh
                num_easy = np.sum(easy_mask)
                pct_easy = 100 * num_easy / len(X_all)
                
                if num_easy > 0:
                    accuracy_on_easy = np.mean(y_all[easy_mask] == 1)
                else:
                    accuracy_on_easy = 0.0
                
                time_saved_sec = num_easy * timing_data['avg_time_per_query']
                time_saved_pct = time_saved_sec / timing_data['total_time']
                
                score = score_weight_accuracy * accuracy_on_easy + (1 - score_weight_accuracy) * (19/20) * time_saved_pct
                
                if score > best_score:
                    best_score = score
                    best_threshold = thresh
                    best_threshold_idx = idx
                
                threshold_results.append({
                    'threshold': thresh,
                    'pct_easy': pct_easy,
                    'num_easy': num_easy,
                    'accuracy_on_easy': accuracy_on_easy,
                    'time_saved_pct': time_saved_pct,
                    'score': score
                })
            
            # Mark only the best threshold in results for plotting
            for i, result in enumerate(threshold_results):
                result['is_best'] = (i == best_threshold_idx)
            
            # Get the best result
            best_result = threshold_results[best_threshold_idx]
            
            # Store optimal threshold
            threshold_key = f"{matcher}_{dataset}"
            optimal_thresholds[threshold_key] = {
                'threshold': best_threshold,
                'score': best_score,
                'expected_easy_pct': best_result['pct_easy']
            }
            
            results_by_dataset_matcher[threshold_key] = {
                'threshold_results': threshold_results,
                'total_queries': len(X_all),
                'y_pred_proba': y_pred_proba,
                'best_threshold': best_threshold,
                'best_score': best_score
            }
            
            # Add to summary
            summary_lines.append(f"\n{'─'*90}")
            summary_lines.append(f"DATASET: {dataset} | MATCHER: {matcher.upper()}")
            summary_lines.append(f"{'─'*90}")
            summary_lines.append(f"Validation queries: {len(X_all)}")
            summary_lines.append(f"Correct: {sum(y_all)} ({100*sum(y_all)/len(y_all):.1f}%)")
            summary_lines.append(f"\nThreshold Analysis:")
            summary_lines.append(f"{'Thr':<6} {'Easy %':<10} {'Acc(Easy)':<12} {'TimeSave%':<10} {'Score':<10} {'Best?':<10}")
            summary_lines.append(f"{'-'*60}")
            for res in threshold_results:
                best_marker = "✓ YES" if res['is_best'] else ""
                summary_lines.append(
                    f"{res['threshold']:<6.2f} {res['pct_easy']:<10.1f} "
                    f"{res['accuracy_on_easy']:<12.4f} {100*res['time_saved_pct']:<10.2f} "
                    f"{res['score']:<10.6f} {best_marker:<10}"
                )
            summary_lines.append(f"\n>>> RECOMMENDED THRESHOLD: {best_threshold:.2f}")
            summary_lines.append(f"Expected easy queries: {best_result['pct_easy']:.1f}%")
            summary_lines.append(f"Accuracy on easy queries: {best_result['accuracy_on_easy']:.4f}")
            summary_lines.append(f"Optimal score: {best_result['score']:.6f}")
    
    # Generate comparative threshold trade-off visualization across dataset-model pairs
    plt.figure(figsize=(16, 10))
    
    styles = {'loftr': '-', 'superglue': '--'}
    
    for dataset in training_datasets:
        for matcher in matchers:
            key = f"{matcher}_{dataset}"
            if key not in results_by_dataset_matcher:
                continue
            
            results = results_by_dataset_matcher[key]['threshold_results']
            best_threshold_val = results_by_dataset_matcher[key]['best_threshold']
            thresholds_plot = [r['threshold'] for r in results]
            scores_plot = [r['score'] for r in results]
            best_idx = next(i for i, r in enumerate(results) if r['threshold'] == best_threshold_val)
            
            label = f"{dataset}_{matcher.upper()}"
            plt.plot(thresholds_plot, scores_plot, 
                    marker='o', linewidth=2, markersize=6,
                    label=label, linestyle=styles.get(matcher, '-'))
            
            # Highlight optimal threshold selection with distinct visual marker
            plt.plot(thresholds_plot[best_idx], scores_plot[best_idx], 
                    marker='*', markersize=15, markeredgecolor='black', markeredgewidth=1)
    
    plt.xlabel('Probability Threshold', fontsize=12)
    plt.ylabel(f'Score ({score_weight_accuracy:.1%}×acc + {1-score_weight_accuracy:.1%}×time)', fontsize=12)
    plt.title(f'Threshold Optimization \nalpha={score_weight_accuracy}, Shows Transfer Analysis', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=10, loc='best')
    plt.xticks(thresholds)
    plt.tight_layout()

    plot_path = output_dir / f"threshold_curves_alpha{score_weight_accuracy:.1%}.png"
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"\n✓ Plot saved: {plot_path}")
    plt.close()
    
    # Serialize comprehensive threshold optimization results to persistent text format
    summary_lines.append(f"\n{'='*90}")
    summary_lines.append(f"\nOPTIMAL THRESHOLDS:")
    summary_lines.append(f"{'-'*90}")
    for threshold_key, opt in optimal_thresholds.items():
        dataset, matcher = threshold_key.rsplit('_', 1)
        summary_lines.append(f"\nTrained on {dataset.upper()} | {matcher.upper()}:")
        summary_lines.append(f"  Optimal threshold: {opt['threshold']:.2f}")
        summary_lines.append(f"  Expected easy queries: {opt['expected_easy_pct']:.1f}%")
        summary_lines.append(f"  Automated score: {opt['score']:.6f}")
    
    summary_path = output_dir / "threshold_analysis.txt"
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(summary_lines))
    
    print(f"✓ Summary saved: {summary_path}")
    
    # Persist optimal probability thresholds in structured JSON format for downstream applications
    optimal_thresholds_path = output_dir / "optimal_thresholds.json"
    with open(optimal_thresholds_path, 'w', encoding='utf-8') as f:
        json.dump(optimal_thresholds, f, indent=2)
    
    print(f"Optimal thresholds saved: {optimal_thresholds_path}")

if __name__ == "__main__":
    main()

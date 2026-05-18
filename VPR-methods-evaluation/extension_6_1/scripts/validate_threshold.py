"""
Find optimal probability threshold for easy/hard query decision.
Sweep thresholds and analyze trade-off between % easy queries and quality.

Input:
  - lr_models.pkl
  - Validation dataset (SF-XS val)

Output:
  - threshold_analysis.txt: Results for each threshold
  - threshold_curves.png: Plot of trade-off curves
"""

import json
import pickle
import numpy as np
from pathlib import Path
import sys
import matplotlib.pyplot as plt

# Add utils to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.data_loader import load_inliers_val_set


def compute_recall_at_k(distances: np.ndarray, threshold: float = 25.0, k: int = 1) -> float:
    """
    Compute Recall@K: percentage of queries with correct match in top-k.
    
    Args:
        distances: Array of geo distances for top-k predictions
        threshold: Distance threshold for correctness (default 25m)
        k: Consider only first k predictions
    
    Returns:
        Recall@K percentage (0-1)
    """
    top_k_distances = distances[:k]
    correct = np.min(top_k_distances) <= threshold
    return 1.0 if correct else 0.0


def main():
    # Load config
    config_path = Path(__file__).parent.parent / "config" / "paths_config.json"
    with open(config_path, encoding='utf-8') as f:
        cfg = json.load(f)
    
    base_path = cfg['input']['base_path']
    matchers = cfg['matchers']
    vpr_models = cfg['vpr_models']
    val_dataset = cfg['input']['val_dataset']
    threshold_dist = cfg['hyperparams']['threshold_dist']
    top_k = cfg['hyperparams']['top_k']
    
    # Input: Logistic Regression models
    models_dir = Path(base_path) / cfg['output']['base_dir'] / "lr_models"
    models_path = models_dir / "lr_models.pkl"
    
    if not models_path.exists():
        print(f"Models file not found: {models_path}")
        return
    
    # Load models
    with open(models_path, 'rb') as f:
        models = pickle.load(f)
    
    print(f"Loaded models: {list(models.keys())}")
    
    # Output directory
    output_dir = Path(base_path) / cfg['output']['base_dir'] / "threshold_analysis"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Thresholds to sweep
    thresholds = [0.50, 0.60, 0.70, 0.75, 0.80, 0.85, 0.90]
    
    summary_lines = []
    summary_lines.append("=" * 80)
    summary_lines.append("EXTENSION 6.1 - THRESHOLD VALIDATION")
    summary_lines.append("=" * 80)
    summary_lines.append("")
    
    results_by_matcher = {}
    
    # Process each matcher
    for matcher in matchers:
        print(f"\n{'='*80}")
        print(f"Processing matcher: {matcher}")
        print(f"{'='*80}")
        
        # Collect validation data from ALL VPR models
        X_all = []
        y_all = []
        
        for vpr_model in vpr_models:
            print(f"\n[Loading] {vpr_model}...")
            
            try:
                X, y, dists = load_inliers_val_set(
                    base_path=base_path,
                    vpr_model=vpr_model,
                    matcher=matcher,
                    val_dataset=val_dataset,
                    threshold_dist=threshold_dist
                )
                
                X_all.extend(X)
                y_all.extend(y)
                
            except Exception as e:
                print(f"Error: {e}")
                continue
        
        if len(X_all) == 0:
            print(f"No validation data for {matcher}")
            continue
        
        X_all = np.array(X_all)
        y_all = np.array(y_all)
        
        print(f"\nValidation data loaded: {len(X_all)} queries")
        print(f"Correct: {sum(y_all)} ({100*sum(y_all)/len(y_all):.1f}%)")
        print(f"Wrong: {len(y_all)-sum(y_all)} ({100*(len(y_all)-sum(y_all))/len(y_all):.1f}%)")
        
        # Get model for this matcher
        lr_model = models[matcher]
        
        # Predict probabilities
        X_reshaped = X_all.reshape(-1, 1)
        y_pred_proba = lr_model.predict_proba(X_reshaped)[:, 1] #   [0.3, 0.8, ...]
        
        # Sweep thresholds
        print(f"\nThreshold sweep:")
        print(f"{'Threshold':<12} {'Easy Queries %':<20} {'Easy Count':<15} {'Accuracy':<15}")
        print(f"{'-'*62}")
        
        threshold_results = []
        
        for thresh in thresholds:
            # Classify as easy/hard
            easy_mask = y_pred_proba >= thresh  # Easy if predicted probability of being correct is above threshold
            num_easy = np.sum(easy_mask)
            pct_easy = 100 * num_easy / len(X_all)
            
            # Accuracy on validation set
            y_pred_binary = (y_pred_proba >= 0.5).astype(int)   # Final decision based on 0.5 threshold for correctness, not the easy/hard threshold but the actual correctness prediction
            accuracy = np.mean(y_pred_binary == y_all)
            
            print(f"{thresh:<12.2f} {pct_easy:<20.1f} {num_easy:<15} {accuracy:<15.4f}")
            
            threshold_results.append({
                'threshold': thresh,
                'pct_easy': pct_easy,
                'num_easy': num_easy,
                'accuracy': accuracy
            })
        
        results_by_matcher[matcher] = {
            'threshold_results': threshold_results,
            'total_queries': len(X_all),
            'y_pred_proba': y_pred_proba
        }
        
        # Add to summary
        summary_lines.append(f"\n{'─'*80}")
        summary_lines.append(f"MATCHER: {matcher.upper()}")
        summary_lines.append(f"{'─'*80}")
        summary_lines.append(f"Validation queries: {len(X_all)}")
        summary_lines.append(f"Correct: {sum(y_all)} ({100*sum(y_all)/len(y_all):.1f}%)")
        summary_lines.append(f"Wrong: {len(y_all)-sum(y_all)} ({100*(len(y_all)-sum(y_all))/len(y_all):.1f}%)")
        summary_lines.append(f"\nThreshold Analysis:")
        summary_lines.append(f"{'Threshold':<12} {'Easy Queries %':<20} {'Easy Count':<15} {'Accuracy':<15}")
        summary_lines.append(f"{'-'*62}")
        for res in threshold_results:
            summary_lines.append(
                f"{res['threshold']:<12.2f} {res['pct_easy']:<20.1f} "
                f"{res['num_easy']:<15} {res['accuracy']:<15.4f}"
            )
    
    # Plot threshold curves
    plt.figure(figsize=(14, 8))
    
    colors = {'loftr': 'blue', 'superglue': 'red'}
    
    for matcher in matchers:
        if matcher not in results_by_matcher:
            continue
        
        results = results_by_matcher[matcher]['threshold_results']
        thresholds_plot = [r['threshold'] for r in results]
        pct_easy_plot = [r['pct_easy'] for r in results]
        
        plt.plot(thresholds_plot, pct_easy_plot, 
                marker='o', linewidth=2, markersize=8,
                label=f"{matcher.upper()}", color=colors.get(matcher, 'green'))
    
    plt.xlabel('Probability Threshold', fontsize=12)
    plt.ylabel('Percentage of Easy Queries (%)', fontsize=12)
    plt.title('Threshold Analysis: Easy Queries vs Probability Threshold\n(Higher = More Queries Skip Expensive Matching)', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=11)
    plt.xticks(thresholds)
    plt.tight_layout()
    
    plot_path = output_dir / "threshold_curves.png"
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"\nPlot saved: {plot_path}")
    plt.close()
    
    # Save summary
    summary_lines.append(f"\n{'='*80}")
    summary_path = output_dir / "threshold_analysis.txt"
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(summary_lines))
    
    print(f"Summary saved: {summary_path}")
    print(f"\n{'='*80}")

if __name__ == "__main__":
    main()

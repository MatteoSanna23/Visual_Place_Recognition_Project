"""
Find optimal probability threshold for easy/hard query decision using cost-sensitive scoring.
Sweep thresholds and find the threshold that maximizes: score = accuracy * pct_easy

This balances:
  - accuracy: don't lose quality
  - pct_easy: maximize computational savings

Input:
  - lr_models.pkl (from Step 2)
  - Validation dataset (SF-XS val)

Output:
  - threshold_analysis.txt: Results for each threshold + BEST THRESHOLD
  - threshold_curves.png: Plot of trade-off curves
  - optimal_thresholds.json: Best threshold for each matcher
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
    
    # Input: models from Step 2
    models_dir = Path(base_path) / cfg['output']['base_dir'] / "lr_models"
    models_path = models_dir / "lr_models.pkl"
    
    if not models_path.exists():
        print(f"⚠️  Models file not found: {models_path}")
        print("    Did you run Step 2?")
        return
    
    # Load models
    with open(models_path, 'rb') as f:
        models = pickle.load(f)
    
    print(f"✓ Loaded models: {list(models.keys())}")
    
    # Output directory
    output_dir = Path(base_path) / cfg['output']['base_dir'] / "threshold_analysis"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Thresholds to sweep
    thresholds = [0.50, 0.60, 0.70, 0.75, 0.80, 0.85, 0.90]
    
    summary_lines = []
    summary_lines.append("=" * 90)
    summary_lines.append("EXTENSION 6.1 - STEP 3: THRESHOLD VALIDATION (COST-SENSITIVE)")
    summary_lines.append("=" * 90)
    summary_lines.append("")
    
    results_by_matcher = {}
    optimal_thresholds = {}
    
    # Process each matcher
    for matcher in matchers:
        print(f"\n{'='*90}")
        print(f"Processing matcher: {matcher.upper()}")
        print(f"{'='*90}")
        
        # Collect validation data from ALL VPR models
        X_all = []
        y_all = []
        
        for vpr_model in vpr_models:
            print(f"\n  [Loading] {vpr_model}...")
            
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
                print(f"    ⚠️  Error: {e}")
                continue
        
        if len(X_all) == 0:
            print(f"  ⚠️  No validation data for {matcher}")
            continue
        
        X_all = np.array(X_all)
        y_all = np.array(y_all)
        
        print(f"\n  ✓ Validation data: {len(X_all)} queries")
        print(f"    Correct: {sum(y_all)} ({100*sum(y_all)/len(y_all):.1f}%)")
        print(f"    Wrong: {len(y_all)-sum(y_all)} ({100*(len(y_all)-sum(y_all))/len(y_all):.1f}%)")
        
        # Get model for this matcher
        lr_model = models[matcher]
        
        # Predict probabilities
        X_reshaped = X_all.reshape(-1, 1)
        y_pred_proba = lr_model.predict_proba(X_reshaped)[:, 1]
        
        # Sweep thresholds with cost-sensitive scoring
        print(f"\n  Threshold sweep (Cost-Sensitive Score):")
        print(f"  {'Threshold':<12} {'Easy %':<12} {'Accuracy':<12} {'Score':<12} {'Status':<15}")
        print(f"  {'-'*63}")
        
        threshold_results = []
        best_score = -np.inf
        best_threshold = 0.5
        
        for thresh in thresholds:
            # Classify as easy/hard AND correct/wrong using the same threshold
            # Higher threshold = more stringent = fewer predicted as correct
            y_pred_binary = (y_pred_proba >= thresh).astype(int)
            num_easy = np.sum(y_pred_binary)
            pct_easy_frac = num_easy / len(X_all)
            pct_easy = 100 * pct_easy_frac
            
            # Accuracy on validation set (using same threshold for consistency)
            accuracy = np.mean(y_pred_binary == y_all)
            
            # Cost-sensitive score: balance accuracy × ease
            score = accuracy * pct_easy_frac
            
            # Track best threshold
            is_best = False
            if score > best_score:
                best_score = score
                best_threshold = thresh
                is_best = True
            
            status = "BEST" if is_best else ""
            print(f"  {thresh:<12.2f} {pct_easy:<12.1f} {accuracy:<12.4f} {score:<12.4f} {status:<15}")
            
            threshold_results.append({
                'threshold': thresh,
                'pct_easy': pct_easy,
                'num_easy': num_easy,
                'accuracy': accuracy,
                'score': score,
                'is_best': is_best
            })
        
        optimal_thresholds[matcher] = {
            'threshold': best_threshold,
            'score': best_score,
            'expected_easy_pct': threshold_results[[r['threshold'] for r in threshold_results].index(best_threshold)]['pct_easy']
        }
        
        results_by_matcher[matcher] = {
            'threshold_results': threshold_results,
            'total_queries': len(X_all),
            'y_pred_proba': y_pred_proba,
            'best_threshold': best_threshold,
            'best_score': best_score
        }
        
        # Add to summary
        summary_lines.append(f"\n{'─'*90}")
        summary_lines.append(f"MATCHER: {matcher.upper()}")
        summary_lines.append(f"{'─'*90}")
        summary_lines.append(f"Validation queries: {len(X_all)}")
        summary_lines.append(f"Correct: {sum(y_all)} ({100*sum(y_all)/len(y_all):.1f}%)")
        summary_lines.append(f"Wrong: {len(y_all)-sum(y_all)} ({100*(len(y_all)-sum(y_all))/len(y_all):.1f}%)")
        summary_lines.append(f"\nThreshold Analysis (Cost-Sensitive):")
        summary_lines.append(f"{'Threshold':<12} {'Easy %':<12} {'Accuracy':<12} {'Score':<12} {'Best?':<10}")
        summary_lines.append(f"{'-'*58}")
        for res in threshold_results:
            best_marker = "✓ YES" if res['is_best'] else ""
            summary_lines.append(
                f"{res['threshold']:<12.2f} {res['pct_easy']:<12.1f} "
                f"{res['accuracy']:<12.4f} {res['score']:<12.4f} {best_marker:<10}"
            )
        summary_lines.append(f"\n>>> RECOMMENDED THRESHOLD: {best_threshold:.2f}")
        summary_lines.append(f"    Expected easy queries: {optimal_thresholds[matcher]['expected_easy_pct']:.1f}%")
        summary_lines.append(f"    Optimal score: {best_score:.4f}")
    
    # Plot threshold curves (score)
    plt.figure(figsize=(14, 8))
    
    colors = {'loftr': 'blue', 'superglue': 'red', 'lightglue': 'green'}
    
    for matcher in matchers:
        if matcher not in results_by_matcher:
            continue
        
        results = results_by_matcher[matcher]['threshold_results']
        thresholds_plot = [r['threshold'] for r in results]
        scores_plot = [r['score'] for r in results]
        best_idx = [r['is_best'] for r in results].index(True)
        
        plt.plot(thresholds_plot, scores_plot, 
                marker='o', linewidth=2, markersize=8,
                label=f"{matcher.upper()}", color=colors.get(matcher, 'gray'))
        
        # Mark best threshold
        plt.plot(thresholds_plot[best_idx], scores_plot[best_idx], 
                marker='*', markersize=20, color=colors.get(matcher, 'gray'),
                markeredgecolor='black', markeredgewidth=1.5)
    
    plt.xlabel('Probability Threshold', fontsize=12)
    plt.ylabel('Cost-Sensitive Score (accuracy × pct_easy)', fontsize=12)
    plt.title('Threshold Optimization: Cost-Sensitive Score\n(★ = Best threshold per matcher)', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=11)
    plt.xticks(thresholds)
    plt.tight_layout()
    
    plot_path = output_dir / "threshold_curves_cost_sensitive.png"
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"\n✓ Plot saved: {plot_path}")
    plt.close()
    
    # Save summary
    summary_lines.append(f"\n{'='*90}")
    summary_lines.append("\nOPTIMAL THRESHOLDS (MATCHER-SPECIFIC):")
    summary_lines.append(f"{'-'*90}")
    for matcher, opt in optimal_thresholds.items():
        summary_lines.append(f"\n{matcher.upper()}:")
        summary_lines.append(f"  Optimal threshold: {opt['threshold']:.2f}")
        summary_lines.append(f"  Expected easy queries: {opt['expected_easy_pct']:.1f}%")
        summary_lines.append(f"  Cost-sensitive score: {opt['score']:.4f}")
    
    summary_path = output_dir / "threshold_analysis.txt"
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(summary_lines))
    
    print(f"✓ Summary saved: {summary_path}")
    
    # Save optimal thresholds as JSON for Step 4
    optimal_thresholds_path = output_dir / "optimal_thresholds.json"
    with open(optimal_thresholds_path, 'w', encoding='utf-8') as f:
        json.dump(optimal_thresholds, f, indent=2)
    
    print(f"✓ Optimal thresholds saved: {optimal_thresholds_path}")
    
    print(f"\n{'='*90}")
    print("SUMMARY:")
    for matcher, opt in optimal_thresholds.items():
        print(f"  {matcher}: threshold={opt['threshold']:.2f}, easy={opt['expected_easy_pct']:.1f}%")
    print(f"{'='*90}")


if __name__ == "__main__":
    main()

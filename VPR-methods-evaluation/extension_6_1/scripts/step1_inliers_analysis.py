"""
EXTENSION 6.1 - STEP 1: Analyze Inliers Distribution
========================================================

Analyze the distribution of inliers_top1 for each matcher across all training datasets.
Extract features (X) and labels (y) for logistic regression training.

Output:
  - inliers_{matcher}.pkl: Training data (X, y)
  - distribution_{matcher}.png: Histogram plots
  - step1_summary.txt: Statistics and summary
"""

import json
import pickle
import numpy as np
from pathlib import Path
import sys

# Add utils to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.data_loader import load_inliers_and_labels, get_inliers_statistics
from utils.visualization import plot_inliers_distribution


def main():
    # Load config
    config_path = Path(__file__).parent.parent / "config" / "paths_config.json"
    with open(config_path) as f:
        cfg = json.load(f)
    
    base_path = cfg['input']['base_path']
    matchers = cfg['matchers']
    vpr_models = cfg['vpr_models']
    training_datasets = cfg['input']['training_datasets']
    threshold_dist = cfg['hyperparams']['threshold_dist']
    
    # Output directories
    output_dir = Path(base_path) / cfg['output']['base_dir'] / cfg['output']['step1']
    output_dir.mkdir(parents=True, exist_ok=True)
    
    summary_lines = []
    summary_lines.append("=" * 80)
    summary_lines.append("EXTENSION 6.1 - STEP 1: INLIERS ANALYSIS SUMMARY")
    summary_lines.append("=" * 80)
    summary_lines.append("")
    
    # Process each matcher
    for matcher in matchers:
        print(f"\n{'='*80}")
        print(f"Processing matcher: {matcher.upper()}")
        print(f"{'='*80}")
        
        # We collect data from ALL VPR models for this matcher
        X_all = []
        y_all = []
        
        for vpr_model in vpr_models:
            print(f"\n[{vpr_model}] Loading training data...")
            
            try:
                X, y = load_inliers_and_labels(
                    base_path=base_path,
                    vpr_model=vpr_model,
                    matcher=matcher,
                    datasets=training_datasets,
                    threshold_dist=threshold_dist
                )
                
                X_all.extend(X)
                y_all.extend(y)
                
            except Exception as e:
                print(f"  ⚠️  Error loading data for {vpr_model}: {e}")
                continue
        
        if len(X_all) == 0:
            print(f"  ✗ No data loaded for {matcher}")
            continue
        
        X_all = np.array(X_all)
        y_all = np.array(y_all)
        
        # Save as pickle
        output_pkl = output_dir / f"inliers_{matcher}.pkl"
        data_dict = {'X': X_all, 'y': y_all}
        with open(output_pkl, 'wb') as f:
            pickle.dump(data_dict, f)
        print(f"\n✓ Data saved: {output_pkl}")
        
        # Compute statistics
        stats = get_inliers_statistics(X_all, y_all)
        
        # Plot distribution
        X_correct = X_all[y_all == 1]
        X_wrong = X_all[y_all == 0]
        
        output_png = output_dir / f"distribution_{matcher}.png"
        plot_inliers_distribution(X_correct, X_wrong, matcher, output_png)
        
        # Print summary
        print(f"\n[Statistics] {matcher.upper()}")
        print(f"  Total samples: {len(X_all)}")
        print(f"  Correct queries: {stats['correct']['count']} ({100*stats['correct']['count']/len(X_all):.1f}%)")
        print(f"  Wrong queries: {stats['wrong']['count']} ({100*stats['wrong']['count']/len(X_all):.1f}%)")
        print(f"\n  Inliers - Correct queries:")
        print(f"    Mean: {stats['correct']['mean']:.2f}")
        print(f"    Std: {stats['correct']['std']:.2f}")
        print(f"    Median (p50): {stats['correct']['p50']:.2f}")
        print(f"    Range: [{stats['correct']['min']:.0f}, {stats['correct']['max']:.0f}]")
        print(f"    Percentiles: p25={stats['correct']['p25']:.2f}, p75={stats['correct']['p75']:.2f}")
        
        print(f"\n  Inliers - Wrong queries:")
        print(f"    Mean: {stats['wrong']['mean']:.2f}")
        print(f"    Std: {stats['wrong']['std']:.2f}")
        print(f"    Median (p50): {stats['wrong']['p50']:.2f}")
        print(f"    Range: [{stats['wrong']['min']:.0f}, {stats['wrong']['max']:.0f}]")
        print(f"    Percentiles: p25={stats['wrong']['p25']:.2f}, p75={stats['wrong']['p75']:.2f}")
        
        # Separation quality
        overlap_min = max(stats['wrong']['max'], stats['correct']['min'])
        overlap_max = min(stats['correct']['max'], stats['wrong']['min'])
        
        if overlap_max >= overlap_min:
            print(f"\n  ⚠️  Overlap region: [{overlap_min:.0f}, {overlap_max:.0f}]")
            print(f"      Queries in overlap: Poor separability")
        else:
            print(f"\n  ✓ Good separation! Gap: {overlap_min - overlap_max:.0f}")
        
        # Add to summary file
        summary_lines.append(f"\n{'─'*80}")
        summary_lines.append(f"MATCHER: {matcher.upper()}")
        summary_lines.append(f"{'─'*80}")
        summary_lines.append(f"Total samples: {len(X_all)}")
        summary_lines.append(f"Correct queries: {stats['correct']['count']} ({100*stats['correct']['count']/len(X_all):.1f}%)")
        summary_lines.append(f"Wrong queries: {stats['wrong']['count']} ({100*stats['wrong']['count']/len(X_all):.1f}%)")
        summary_lines.append(f"\nCorrect queries - Inliers:")
        summary_lines.append(f"  Mean: {stats['correct']['mean']:.2f} ± {stats['correct']['std']:.2f}")
        summary_lines.append(f"  Median: {stats['correct']['p50']:.2f}")
        summary_lines.append(f"  Range: [{stats['correct']['min']:.0f}, {stats['correct']['max']:.0f}]")
        summary_lines.append(f"\nWrong queries - Inliers:")
        summary_lines.append(f"  Mean: {stats['wrong']['mean']:.2f} ± {stats['wrong']['std']:.2f}")
        summary_lines.append(f"  Median: {stats['wrong']['p50']:.2f}")
        summary_lines.append(f"  Range: [{stats['wrong']['min']:.0f}, {stats['wrong']['max']:.0f}]")
    
    # Save summary
    summary_lines.append(f"\n{'='*80}")
    summary_path = output_dir / "step1_summary.txt"
    with open(summary_path, 'w') as f:
        f.write('\n'.join(summary_lines))
    
    print(f"\n{'='*80}")
    print(f"✓ Step 1 completed!")
    print(f"✓ Summary saved: {summary_path}")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()

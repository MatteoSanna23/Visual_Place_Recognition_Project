"""
Inliers Analysis

Output:
  - inliers_{matcher}_{dataset}.pkl: Training data (X, y) per dataset
  - distribution_{matcher}_{dataset}.png: Histogram plots per dataset
  - inliers_analysis_summary.txt: Statistics and summary
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
    # Load configuration parameters from external configuration file
    config_path = Path(__file__).parent.parent / "config" / "paths_config.json"
    with open(config_path) as f:
        cfg = json.load(f)
    
    base_path = cfg['input']['base_path']
    matchers = cfg['matchers']
    vpr_models = cfg['vpr_models']
    training_datasets = cfg['input']['training_datasets']
    threshold_dist = cfg['hyperparams']['threshold_dist']
    
    # Initialize output directory structure for results storage
    output_dir = Path(base_path) / cfg['output']['base_dir'] / "inliers_analysis"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    summary_lines = []
    summary_lines.append("=" * 80)
    summary_lines.append("EXTENSION 6.1 - INLIERS ANALYSIS")
    summary_lines.append("=" * 80)
    
    for dataset in training_datasets:
        
        print(f"Processing dataset: {dataset}")
        
        for matcher in matchers:
            print(f"\n  Matcher: {matcher}")
            
            # Aggregate inlier correspondence counts from all VPR models for comparative analysis
            X_all = []
            y_all = []
            
            for vpr_model in vpr_models:
                print(f"[{vpr_model}] Loading data...")
                
                try:
                    X, y = load_inliers_and_labels(
                        base_path=base_path,
                        vpr_model=vpr_model,
                        matcher=matcher,
                        dataset=dataset,
                        threshold_dist=threshold_dist
                    )
                    
                    X_all.extend(X)
                    y_all.extend(y)
                    
                except Exception as e:
                    print(f"Error: {e}")
                    continue
            
            if len(X_all) == 0:
                print(f"No data loaded for {matcher} in {dataset}")
                continue
            
            X_all = np.array(X_all)
            y_all = np.array(y_all)
            
            # Persist aggregated inlier data and correspondence labels
            output_pkl = output_dir / f"inliers_{matcher}_{dataset}.pkl"
            data_dict = {'X': X_all, 'y': y_all}
            with open(output_pkl, 'wb') as f:
                pickle.dump(data_dict, f)
            print(f"Saved: {output_pkl}")
            
            # Calculate comprehensive statistical descriptors for inlier distributions
            stats = get_inliers_statistics(X_all, y_all)
            
            # Generate comparative distribution analysis visualization
            X_correct = X_all[y_all == 1]
            X_wrong = X_all[y_all == 0]
            
            output_png = output_dir / f"distribution_{matcher}_{dataset}.png"
            plot_inliers_distribution(X_correct, X_wrong, f"{matcher}_{dataset}", output_png)
            
            # Append detailed statistical summary to comprehensive analysis report
            summary_lines.append(f"\n{'─'*80}")
            summary_lines.append(f"DATASET: {dataset} | MATCHER: {matcher}")
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
    
    # Serialize comprehensive analysis summary to persistent text format
    summary_lines.append(f"\n{'='*80}")
    summary_path = output_dir / "inliers_analysis_summary.txt"
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(summary_lines))

if __name__ == "__main__":
    main()
    
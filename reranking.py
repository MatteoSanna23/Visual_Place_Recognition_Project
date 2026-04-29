import numpy as np
from tqdm import tqdm
import os, argparse, sys
from glob import glob
from pathlib import Path
import torch

from util import get_list_distances_from_preds

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--preds-dir", type=str, help="directory with predictions of a VPR model")
    parser.add_argument("--inliers-dir", type=str, help="directory with image matching results")
    parser.add_argument("--num-preds", type=int, default=100, help="number of predictions to re-rank")
    parser.add_argument("--positive-dist-threshold", type=int, default=25, help="distance (in meters)")
    parser.add_argument("--recall-values", type=int, nargs="+", default=[1, 5, 10, 20], help="values for recall")
    return parser.parse_args()

def main(args):
    preds_folder = args.preds_dir
    inliers_folder = Path(args.inliers_dir)
    num_preds = args.num_preds
    threshold = args.positive_dist_threshold
    recall_values = args.recall_values

    txt_files = glob(os.path.join(preds_folder, "*.txt"))
    txt_files.sort(key=lambda x: int(Path(x).stem))

    total_queries = len(txt_files)
    
    # --- Printing some informations ---
    # Extract the method name from the path
    vpr_method = Path(preds_folder).parents[3].name.replace("_predictions", "")
    matcher_method = inliers_folder.parent.name
    
    print("-" * 50)
    print(f"Distance Metric: L2 (Standard)")
    print(f"Predictions Dir: {preds_folder}")
    print(f"Inliers Dir:     {args.inliers_dir}")
    print(f"Method:          {vpr_method} + {matcher_method}")
    print(f"Number of Queries: {total_queries}")
    print(f"Threshold:       {threshold} meters")
    print("-" * 50)

    recalls = np.zeros(len(recall_values))

    # tqdm with 'leave=False' to disappear after completion
    for txt_file_query in tqdm(txt_files, desc="Calculating Recall", leave=False, disable=not sys.stdout.isatty()):
        geo_dists = torch.tensor(get_list_distances_from_preds(txt_file_query))[:num_preds]
        torch_file_query = inliers_folder.joinpath(Path(txt_file_query).name.replace('txt', 'torch'))
        
        if not torch_file_query.exists():
            continue
            
        query_results = torch.load(torch_file_query, weights_only=False)
        query_db_inliers = torch.zeros(num_preds, dtype=torch.float32)
        
        for i in range(min(len(query_results), num_preds)):
            query_db_inliers[i] = query_results[i]['num_inliers']
            
        query_db_inliers, indices = torch.sort(query_db_inliers, descending=True)
        geo_dists = geo_dists[indices]
        
        for i, n in enumerate(recall_values):
            if torch.any(geo_dists[:n] <= threshold):
                recalls[i:] += 1
                break

    recalls = recalls / total_queries * 100
    
    # --- Print formatted results ---
    print("\nFINAL RESULTS:")

    for val, rec in zip(recall_values, recalls):
        print(f"R@{val}: {rec:.2f}%")
    print("-" * 50)

if __name__ == "__main__":
    args = parse_arguments()
    main(args)
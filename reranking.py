import numpy as np
import os, argparse, re
import torch
import matplotlib.pyplot as plt
from glob import glob
from pathlib import Path
from util import get_list_distances_from_preds

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--preds-dir", type=str, help="directory with predictions of a VPR model")
    parser.add_argument("--inliers-dir", type=str, help="directory with image matching results")
    parser.add_argument("--num-preds", type=int, default=20, help="number of predictions to re-rank")
    parser.add_argument("--positive-dist-threshold", type=int, default=25, help="distance in meters")
    parser.add_argument("--recall-values", type=int, nargs="+", default=[1, 5, 10, 20], help="recall values")
    parser.add_argument("--vpr-model", type=str, default="unknown", help="VPR model name (e.g., netvlad, cosplace)")
    parser.add_argument("--dataset", type=str, default="unknown", help="dataset name (e.g., tokyo, sf_xs)")
    parser.add_argument("--matcher", type=str, default="unknown", help="image matcher name (e.g., loftr, superpoint)")
    return parser.parse_args()

def parse_original_results(preds_dir):
    """
    Reads the original retrieval performance from results.txt 
    located in the parent directory of 'preds'.
    """
    results_path = Path(preds_dir).parent / "results.txt"
    original_recalls = {}
    if results_path.exists():
        with open(results_path, 'r') as f:
            content = f.read()
            # Find patterns like R@1: 51.50%
            matches = re.findall(r"R@(\d+):\s+([\d.]+)%", content)
            for val, rec in matches:
                original_recalls[int(val)] = float(rec)
    return original_recalls

def main(args):
    preds_folder = args.preds_dir
    inliers_folder = Path(args.inliers_dir)
    num_preds = args.num_preds
    threshold = args.positive_dist_threshold
    recall_values = args.recall_values
    vpr_model = args.vpr_model
    dataset = args.dataset
    matcher = args.matcher

    # Load original retrieval recalls for comparison [cite: 10, 14, 15]
    original_stats = parse_original_results(preds_folder)

    txt_files = glob(os.path.join(preds_folder, "*.txt"))
    txt_files.sort(key=lambda x: int(Path(x).stem))

    total_queries = len(txt_files)
    recalls_reranked = np.zeros(len(recall_values))
    
    # Lists for inlier correlation analysis [cite: 82, 83, 103, 106]
    inliers_correct_queries = []
    inliers_wrong_queries = []

    # Iterate through queries (tqdm removed to keep .txt output clean)
    for txt_file_query in txt_files:
        # Get original distances from retrieval part [cite: 6, 7]
        geo_dists_orig = torch.tensor(get_list_distances_from_preds(txt_file_query))[:num_preds]
        
        # Load inliers from the matching step [cite: 11]
        torch_file_query = inliers_folder.joinpath(Path(txt_file_query).name.replace('txt', 'torch'))
        if not torch_file_query.exists(): 
            continue
        
        query_results = torch.load(torch_file_query, weights_only=False)
        query_db_inliers = torch.zeros(num_preds)
        for i in range(min(len(query_results), num_preds)):
            query_db_inliers[i] = query_results[i]['num_inliers']

        # Analysis: correlation between original R@1 correctness and inliers [cite: 82, 103, 106]
        if geo_dists_orig[0] <= threshold:
            inliers_correct_queries.append(query_db_inliers[0].item())
        else:
            inliers_wrong_queries.append(query_db_inliers[0].item())

        # Geometric Re-ranking based on inlier count 
        _, indices = torch.sort(query_db_inliers, descending=True)
        geo_dists_reranked = geo_dists_orig[indices]
        
        # Calculate Reranked Recall@N [cite: 14, 15]
        for i, n in enumerate(recall_values):
            if torch.any(geo_dists_reranked[:n] <= threshold):
                recalls_reranked[i:] += 1
                break

    # Final recall calculation
    recalls_reranked = recalls_reranked / total_queries * 100

    # --- PRINT COMPARISON TABLE ---
    print("\n" + "="*65)
    print(f"VPR RE-RANKING EVALUATION ({total_queries} queries)")
    print("-" * 65)
    print(f"Dataset: {dataset} | Model: {vpr_model} | Matcher: {matcher}")
    print("-" * 65)
    print(f"{'Metric':<10} | {'Original Retrieval':<20} | {'With Re-ranking':<15}")
    print("-" * 65)
    for i, val in enumerate(recall_values):
        orig = f"{original_stats.get(val, 0.0):>18.2f}%" if val in original_stats else "N/A"
        print(f"R@{val:<8} | {orig} | {recalls_reranked[i]:>13.2f}%")
    print("="*65)

    # --- GENERATE INLIER HISTOGRAM ---
    # This addresses the goal of finding correlation between correctness and inliers [cite: 82, 83, 84]
    if inliers_correct_queries or inliers_wrong_queries:
        plt.figure(figsize=(11, 7))
        plt.hist(inliers_correct_queries, bins=30, alpha=0.5, label='Correct Queries (Original R@1)', color='green')
        plt.hist(inliers_wrong_queries, bins=30, alpha=0.5, label='Wrong Queries (Original R@1)', color='red')
        plt.title(f"Inliers Distribution: Correct vs Wrong\nDataset: {dataset} | Model: {vpr_model} | Matcher: {matcher} (Threshold {threshold}m)", fontsize=12)
        plt.xlabel("Number of Inliers")
        plt.ylabel("Number of Queries")
        plt.legend()
        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        
        # Save histogram in the same folder as .torch files
        plot_path = inliers_folder / "analysis_histogram.png"
        plt.savefig(plot_path)
        print(f"Histogram saved to: {plot_path}")

if __name__ == "__main__":
    args = parse_arguments()
    main(args)
import time
import os
import sys
import argparse
import torch
from glob import glob
from tqdm import tqdm
from pathlib import Path
from copy import deepcopy

from util import read_file_preds

sys.path.append(str(Path(__file__).parent.joinpath("image-matching-models")))

from matching import get_matcher, available_models
from matching.utils import get_default_device

def parse_arguments():
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--preds-dir", type=str, help="directory with predictions of a VPR model")
    parser.add_argument("--out-dir", type=str, default=None, help="output directory of image matching results")
    # Choose matcher
    parser.add_argument(
        "--matcher",
        type=str,
        default="sift-lg",
        choices=available_models,
        help="choose your matcher",
    )
    parser.add_argument("--device", type=str, default=get_default_device(), choices=["cpu", "cuda"])
    parser.add_argument("--im-size", type=int, default=512, help="resize img to im_size x im_size")
    parser.add_argument("--num-preds", type=int, default=100, help="number of predictions to match")
    parser.add_argument("--start-query", type=int, default=-1, help="query to start from")
    parser.add_argument("--num-queries", type=int, default=-1, help="number of queries")

    return parser.parse_args()

def main(args):
    device = args.device
    matcher_name = args.matcher
    img_size = args.im_size
    num_preds = args.num_preds
    matcher = get_matcher(matcher_name, device=device)
    preds_folder = args.preds_dir
    start_query = args.start_query
    num_queries = args.num_queries

    output_folder = Path(preds_folder + f"_{matcher_name}") if args.out_dir is None else Path(args.out_dir)
    output_folder.mkdir(exist_ok=True)
    
    txt_files = glob(os.path.join(preds_folder, "*.txt"))
    txt_files.sort(key=lambda x: int(Path(x).stem))

    start_query = start_query if start_query >= 0 else 0
    num_queries = num_queries if num_queries >= 0 else len(txt_files)

    # --- Start monitoring time ---
    start_time_total = time.time()
    count_processed = 0
    
    for txt_file in tqdm(txt_files[start_query : start_query + num_queries]):
        q_num = Path(txt_file).stem
        out_file = output_folder.joinpath(f"{q_num}.torch")
        if out_file.exists():
            continue
        results = []
        q_path, pred_paths = read_file_preds(txt_file)
        img0 = matcher.load_image(q_path, resize=img_size)
        for pred_path in pred_paths[:num_preds]:
            img1 = matcher.load_image(pred_path, resize=img_size)
            result = matcher(deepcopy(img0), img1)
            result["all_desc0"] = result["all_desc1"] = None
            results.append(result)
        torch.save(results, out_file)
        count_processed += 1
        
    # --- COMPUTE TIMING STATISTICS and SAVE ---
    end_time_total = time.time()
    total_duration = end_time_total - start_time_total
    
    if count_processed > 0:
        avg_time = total_duration / count_processed
        
        stats_file = output_folder.joinpath("timing_report.txt")
        with open(stats_file, "w") as f:
            f.write(f"--- Timing Report for {matcher_name} ---\n")
            f.write(f"Total queries processed: {count_processed}\n")
            f.write(f"Total time: {total_duration:.2f} seconds ({total_duration/60:.2f} minutes)\n")
            f.write(f"Average time per query: {avg_time:.4f} seconds\n")
            f.write(f"Device used: {device}\n")
        
        print(f"\n[INFO] Report saved in: {stats_file}")
        print(f"[INFO] Average time per query: {avg_time:.4f}s")
    else:
        print("\n[INFO] No new queries processed (all files already existed).")

if __name__ == "__main__":
    args = parse_arguments()
    main(args)
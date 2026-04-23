import parser
import sys
from datetime import datetime
from pathlib import Path

import faiss
import numpy as np
import torch
from loguru import logger
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Subset
from tqdm import tqdm

import visualizations
import vpr_models
from test_dataset import TestDataset


def main(args):
    # Store the run start time to build a unique logging folder.
    start_time = datetime.now()

    # Clear previously configured loguru handlers to avoid duplicated logs.
    logger.remove()
    # Build the output path: logs/<user_subdir>/<timestamp>/.
    log_dir = Path("logs") / args.log_dir / start_time.strftime("%Y-%m-%d_%H-%M-%S")

    # Log INFO messages to stdout with colors and a compact timestamped format.
    logger.add(sys.stdout, colorize=True, format="<green>{time:%Y-%m-%d %H:%M:%S}</green> {message}", level="INFO")
    # Log INFO messages to a persistent info file inside the run directory.
    logger.add(log_dir / "info.log", format="<green>{time:%Y-%m-%d %H:%M:%S}</green> {message}", level="INFO")
    # Log DEBUG and above messages to a dedicated debug file.
    logger.add(log_dir / "debug.log", level="DEBUG")

    # Save the executed CLI command for reproducibility.
    logger.info(" ".join(sys.argv))
    # Save the full parsed arguments object.
    logger.info(f"Arguments: {args}")
    # Log the selected VPR setup.
    logger.info(
        f"Testing with {args.method} with a {args.backbone} backbone and descriptors dimension {args.descriptors_dimension}"
    )
    # Show where all artifacts from this run are stored.
    logger.info(f"The outputs are being saved in {log_dir}")

    # Instantiate the requested VPR model.
    model = vpr_models.get_model(args.method, args.backbone, args.descriptors_dimension)
    # Switch to evaluation mode and move the model to the configured device.
    model = model.eval().to(args.device)

    # Create the benchmark dataset with database and query folders.
    test_ds = TestDataset(
        args.database_folder,
        args.queries_folder,
        positive_dist_threshold=args.positive_dist_threshold,
        image_size=args.image_size,
        use_labels=args.use_labels,
    )
    # Print dataset summary.
    logger.info(f"Testing on {test_ds}")

    # Disable gradient computation for faster and lighter inference.
    with torch.inference_mode():
        logger.debug("Extracting database descriptors for evaluation/testing")

        # Build a subset containing only database samples (first num_database items).
        database_subset_ds = Subset(test_ds, list(range(test_ds.num_database)))
        # Use batch inference for the database split.
        database_dataloader = DataLoader(
            dataset=database_subset_ds, num_workers=args.num_workers, batch_size=args.batch_size
        )

        # Pre-allocate one descriptor matrix for both database and query items.
        all_descriptors = np.empty((len(test_ds), args.descriptors_dimension), dtype="float32")

        # Extract and store descriptors for each database batch.
        for images, indices in tqdm(database_dataloader):
            # Forward pass on the selected device.
            descriptors = model(images.to(args.device))
            # Move back to CPU and convert to NumPy.
            descriptors = descriptors.cpu().numpy()
            # Write descriptors at their original dataset indices.
            all_descriptors[indices.numpy(), :] = descriptors

        logger.debug("Extracting queries descriptors for evaluation/testing using batch size 1")

        # Build a subset containing only query samples.
        queries_subset_ds = Subset(
            test_ds, list(range(test_ds.num_database, test_ds.num_database + test_ds.num_queries))
        )
        # Use batch size 1 for queries, as required by this evaluation setup.
        queries_dataloader = DataLoader(dataset=queries_subset_ds, num_workers=args.num_workers, batch_size=1)

        # Extract and store descriptors for each query sample.
        for images, indices in tqdm(queries_dataloader):
            # Forward pass on the selected device.
            descriptors = model(images.to(args.device))
            # Move back to CPU and convert to NumPy.
            descriptors = descriptors.cpu().numpy()
            # Write descriptors at their original dataset indices.
            all_descriptors[indices.numpy(), :] = descriptors

    # Slice the query descriptors from the combined descriptor matrix.
    queries_descriptors = all_descriptors[test_ds.num_database :]
    # Slice the database descriptors from the combined descriptor matrix.
    database_descriptors = all_descriptors[: test_ds.num_database]

    # Optionally persist descriptors for external analysis or debugging.
    if args.save_descriptors:
        logger.info(f"Saving the descriptors in {log_dir}")
        np.save(log_dir / "queries_descriptors.npy", queries_descriptors)
        np.save(log_dir / "database_descriptors.npy", database_descriptors)

    # Loop over each distance metric.
    for distance_metric in args.distance_metric:
        logger.info(f"\n=== Evaluating with {distance_metric} distance metric ===")
        
        # Create a subdirectory for this metric's results
        metric_log_dir = log_dir / distance_metric
        metric_log_dir.mkdir(parents=True, exist_ok=True)
        
        # Create copies of descriptors to avoid modifying originals
        db_desc = database_descriptors.copy()
        q_desc = queries_descriptors.copy()
        
        # Build a FAISS index based on the selected distance metric.
        if distance_metric == "dot_product":
            # Normalize descriptors for dot product (equivalent to cosine similarity)
            faiss.normalize_L2(db_desc)
            faiss.normalize_L2(q_desc)
            # Build a FAISS flat Inner Product index for exact nearest-neighbor search.
            faiss_index = faiss.IndexFlatIP(args.descriptors_dimension)
        else:  # L2 distance (default)
            # Build a FAISS flat L2 index for exact nearest-neighbor search.
            faiss_index = faiss.IndexFlatL2(args.descriptors_dimension)
        
        # Add all database descriptors to the search index.
        faiss_index.add(db_desc)

        logger.debug(f"Calculating recalls with {distance_metric} metric")
        # Retrieve top-k nearest database candidates for each query.
        distances, predictions = faiss_index.search(q_desc, max(args.recall_values))

        # Evaluate recall only when ground-truth labels are available.
        if args.use_labels:
            # positives_per_query contains valid positive database indices for each query.
            positives_per_query = test_ds.get_positives()
            # Initialize counters for each recall cutoff (R@k).
            recalls = np.zeros(len(args.recall_values))

            # Check each query prediction list.
            for query_index, preds in enumerate(predictions):
                # Test every configured k in ascending order.
                for i, n in enumerate(args.recall_values):
                    # If any positive appears in top-n predictions, count this query as correct for this and larger k.
                    if np.any(np.in1d(preds[:n], positives_per_query[query_index])):
                        recalls[i:] += 1
                        break

            # Convert recall counts to percentages.
            recalls = recalls / test_ds.num_queries * 100
            # Build a compact printable string: "R@1: x, R@5: y, ...".
            recalls_str = ", ".join([f"R@{val}: {rec:.1f}" for val, rec in zip(args.recall_values, recalls)])
            # Log final recall metrics.
            logger.info(f"{distance_metric}: {recalls_str}")

        # Optionally save qualitative prediction visualizations.
        if args.num_preds_to_save != 0:
            logger.info(f"Saving {distance_metric} predictions")
            # Save up to num_preds_to_save predictions for each query.
            visualizations.save_preds(
                predictions[:, : args.num_preds_to_save], test_ds, metric_log_dir, args.save_only_wrong_preds, args.use_labels
            )

        # Optionally store raw retrieval outputs for downstream uncertainty analysis.
        if args.save_for_uncertainty:
            z_data = {}
            # Store database UTM coordinates.
            z_data["database_utms"] = test_ds.database_utms
            # Store positives per query (available when labels are used).
            if args.use_labels:
                z_data["positives_per_query"] = positives_per_query
            # Store ranked prediction indices returned by FAISS.
            z_data["predictions"] = predictions
            # Store distances associated with each prediction.
            z_data["distances"] = distances

            # Save everything in a single Torch file.
            torch.save(z_data, metric_log_dir / "z_data.torch")
    
    # Release memory
    del database_descriptors, queries_descriptors, all_descriptors


if __name__ == "__main__":
    # Parse command-line arguments.
    args = parser.parse_arguments()
    # Execute the evaluation pipeline.
    main(args)

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
    # Initialize logging and output directory
    start_time = datetime.now()
    logger.remove()
    log_dir = Path("testing_logs") / args.log_dir
    
    logger.info(" ".join(sys.argv))
    logger.info(f"Arguments: {args}")
    logger.info(
        f"Testing with {args.method} with a {args.backbone} backbone and descriptors dimension {args.descriptors_dimension}"
    )
    logger.info(f"The outputs are being saved in {log_dir}")

    # Instantiate and prepare model
    model = vpr_models.get_model(args.method, args.backbone, args.descriptors_dimension)
    model = model.eval().to(args.device)

    # Load test dataset
    test_ds = TestDataset(
        args.database_folder,
        args.queries_folder,
        positive_dist_threshold=args.positive_dist_threshold,
        image_size=args.image_size,
        use_labels=args.use_labels,
    )
    logger.info(f"Testing on {test_ds}")

    # Execute descriptor extraction in inference mode with gradient computation disabled for efficiency and memory optimization
    with torch.inference_mode():
        logger.debug("Extracting database descriptors for evaluation/testing")

        database_subset_ds = Subset(test_ds, list(range(test_ds.num_database)))
        database_dataloader = DataLoader(
            dataset=database_subset_ds, num_workers=args.num_workers, batch_size=args.batch_size
        )

        all_descriptors = np.empty((len(test_ds), args.descriptors_dimension), dtype="float32")

        for images, indices in tqdm(database_dataloader):
            descriptors = model(images.to(args.device))
            descriptors = descriptors.cpu().numpy()
            all_descriptors[indices.numpy(), :] = descriptors

        logger.debug("Extracting queries descriptors for evaluation/testing using batch size 1")

        queries_subset_ds = Subset(
            test_ds, list(range(test_ds.num_database, test_ds.num_database + test_ds.num_queries))
        )
        queries_dataloader = DataLoader(dataset=queries_subset_ds, num_workers=args.num_workers, batch_size=1)

        for images, indices in tqdm(queries_dataloader):
            descriptors = model(images.to(args.device))
            descriptors = descriptors.cpu().numpy()
            all_descriptors[indices.numpy(), :] = descriptors

    # Partition and optionally save descriptors
    queries_descriptors = all_descriptors[test_ds.num_database :]
    database_descriptors = all_descriptors[: test_ds.num_database]

    if args.save_descriptors:
        logger.info(f"Saving the descriptors in {log_dir}")
        np.save(log_dir / "queries_descriptors.npy", queries_descriptors)
        np.save(log_dir / "database_descriptors.npy", database_descriptors)

    # Evaluate across different distance metrics
    for distance_metric in args.distance_metric:
        logger.info(f"\n=== Evaluating with {distance_metric} distance metric ===")
        
        metric_log_dir = log_dir / distance_metric
        metric_log_dir.mkdir(parents=True, exist_ok=True)
        db_desc = database_descriptors.copy()
        q_desc = queries_descriptors.copy()
        
        # Build FAISS index
        if distance_metric == "dot_product":
            faiss_index = faiss.IndexFlatIP(args.descriptors_dimension)
        else:
            faiss_index = faiss.IndexFlatL2(args.descriptors_dimension)
        
        faiss_index.add(db_desc)

        logger.debug(f"Calculating recalls with {distance_metric} metric")
        distances, predictions = faiss_index.search(q_desc, max(args.recall_values))

        # Compute recall metrics
        if args.use_labels:
            positives_per_query = test_ds.get_positives()
            recalls = np.zeros(len(args.recall_values))

            for query_index, preds in enumerate(predictions):
                for i, n in enumerate(args.recall_values):
                    if np.any(np.isin(preds[:n], positives_per_query[query_index])):
                        recalls[i:] += 1
                        break

            recalls = recalls / test_ds.num_queries * 100
            recalls_str = ", ".join([f"R@{val}: {rec:.1f}" for val, rec in zip(args.recall_values, recalls)])
            logger.info(f"{distance_metric}: {recalls_str}")
            
            # Save results summary
            results_file = metric_log_dir / "results.txt"
            with open(results_file, "w") as f:
                f.write(f"Distance Metric: {distance_metric}\n")
                f.write(f"Database: {args.database_folder}\n")
                f.write(f"Queries: {args.queries_folder}\n")
                f.write(f"Method: {args.method}\n")
                f.write(f"Backbone: {args.backbone}\n")
                f.write(f"Descriptors Dimension: {args.descriptors_dimension}\n")
                f.write(f"Number of Queries: {test_ds.num_queries}\n")
                f.write(f"Number of Database Images: {test_ds.num_database}\n")
                f.write(f"\n{recalls_str}\n")
                for val, rec in zip(args.recall_values, recalls):
                    f.write(f"R@{val}: {rec:.2f}%\n")

        # Save visualizations
        if args.num_preds_to_save != 0:
            logger.info(f"Saving {distance_metric} predictions")
            visualizations.save_preds(
                predictions[:, : args.num_preds_to_save], test_ds, metric_log_dir, args.save_only_wrong_preds, args.use_labels
            )

        # Save raw retrieval data
        if args.save_for_uncertainty:
            z_data = {}
            z_data["database_utms"] = test_ds.database_utms
            if args.use_labels:
                z_data["positives_per_query"] = positives_per_query
            z_data["predictions"] = predictions
            z_data["distances"] = distances
            torch.save(z_data, metric_log_dir / "z_data.torch")
    
    del database_descriptors, queries_descriptors, all_descriptors


if __name__ == "__main__":
    args = parser.parse_arguments()
    main(args)

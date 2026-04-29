#!/bin/bash

# --- CONFIGURATION ---
# Base directory for logs
BASE_LOGS="/teamspace/studios/this_studio/Visual_Place_Recognition_Project/VPR-methods-evaluation/logs"

VPR_MODEL=$1  # netvlad, cosplace, mixvpr, megaloc
DATASET=$2    # tokyo, sf_xs, svox_sun, svox_night

if [ -z "$VPR_MODEL" ] || [ -z "$DATASET" ]; then
    echo "Error! Usage: ./run_reranking.sh <model> <dataset>"
    exit 1
fi

# Find predictions automatically
# With */*/preds covers [TIMESTAMP]/L2/preds
PREDS_DIR=$(ls -d ${BASE_LOGS}/${VPR_MODEL}_predictions/${DATASET}/*/*/preds 2>/dev/null | tail -n 1)

if [ -z "$PREDS_DIR" ]; then
    echo "ERRORE: Non trovo le predizioni in ${BASE_LOGS}/${VPR_MODEL}_predictions/${DATASET}"
    exit 1
fi

# Root path for matching
MATCHING_ROOT="${BASE_LOGS}/${VPR_MODEL}_image_matching"

echo "-------------------------------------------------------"
echo "Analysis: $VPR_MODEL on $DATASET"
echo "-------------------------------------------------------"

for matcher_path in "${MATCHING_ROOT}"/*; do
    # Verify that it's a directory (ignore .txt timing files)
    if [ -d "$matcher_path" ]; then
        MATCHER_NAME=$(basename "$matcher_path")
        INLIERS_DIR="${matcher_path}/${DATASET}"
        
        if [ -d "$INLIERS_DIR" ]; then
            echo "Computing Recall for Matcher: $MATCHER_NAME..."

        # Define the output file at the SAME LEVEL as the .torch files
        # The txt file will be named "reranking_results_${MATCHER_NAME}.txt"
        OUTPUT_FILE="${INLIERS_DIR}/reranking_results_${MATCHER_NAME}.txt"
            
        python reranking.py \
            --preds-dir "$PREDS_DIR" \
            --inliers-dir "$INLIERS_DIR" \
            --num-preds 20 \
            --positive-dist-threshold 25 \
            --recall-values 1 5 10 20 \
            --vpr-model "$VPR_MODEL" \
            --dataset "$DATASET" \
            --matcher "$MATCHER_NAME" \
            > "$OUTPUT_FILE" 2>&1
            
            echo "=> Results saved in: $OUTPUT_FILE"
        fi
    fi
done
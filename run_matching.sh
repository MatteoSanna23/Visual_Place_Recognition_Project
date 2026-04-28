#!/bin/bash

# --- PATH CONFIGURATION ---
# Set the base path where VPR model prediction folders are located
BASE_PREDS_DIR="/teamspace/studios/this_studio/Visual_Place_Recognition_Project/VPR-methods-evaluation/logs/netvlad_predictions/tokyo/2026-04-27_14-07-39/L2/preds"
# Set the base path where you want to save matching results
BASE_OUT_DIR="/teamspace/studios/this_studio/Visual_Place_Recognition_Project/VPR-methods-evaluation/logs/netvlad_image_matching"
DATABASE_NAME="tokyo"
# --- FIXED PARAMETERS ---
# The project-recommended size is 512
IM_SIZE=512
# Number of predictions to refine (K = 20)
NUM_PREDS=20
DEVICE="cuda"

# --- MODEL LISTS ---
# Names of folders containing VPR prediction .txt files
AVAILABLE_VPR_MODELS=("netvlad" "cosplace" "mixvpr" "megaloc")

# Names of available matchers (ensure they match those in 'available_models')
# "superglue", "loftr", "superpoint-lg"
MATCHERS=("superglue" "loftr" "superpoint-lg")

# --- COMMAND LINE ARGUMENT HANDLING ---
if [ $# -eq 0 ]; then
    echo "Usage: $0 <vpr_model>"
    echo "Available VPR models: ${AVAILABLE_VPR_MODELS[@]}"
    exit 1
fi

VPR_MODEL=$1

# Validate VPR model
if [[ ! " ${AVAILABLE_VPR_MODELS[@]} " =~ " ${VPR_MODEL} " ]]; then
    echo "Error: VPR model '${VPR_MODEL}' not recognized."
    echo "Available VPR models: ${AVAILABLE_VPR_MODELS[@]}"
    exit 1
fi

# --- EXECUTION ---
for matcher in "${MATCHERS[@]}"; do
        
        INPUT_DIR="${BASE_PREDS_DIR}"
        OUTPUT_DIR="${BASE_OUT_DIR}/${matcher}/${DATABASE_NAME}"
        
        echo "-------------------------------------------------------"
        echo "Starting Matching: VPR=${VPR_MODEL} | Matcher=${matcher}"
        echo "Input: ${INPUT_DIR}"
        echo "Output: ${OUTPUT_DIR}"
        echo "-------------------------------------------------------"

        mkdir -p "$OUTPUT_DIR"

        # Check if input folder exists before running the script
        if [ -d "$INPUT_DIR" ]; then
            python match_queries_preds.py \
                --preds-dir "$INPUT_DIR" \
                --out-dir "$OUTPUT_DIR" \
                --matcher "$matcher" \
                --device "$DEVICE" \
                --im-size "$IM_SIZE" \
                --num-preds "$NUM_PREDS"
        else
            echo "WARNING: Folder ${INPUT_DIR} not found. Skipping..."
        fi
        
    done

echo "All processes completed."
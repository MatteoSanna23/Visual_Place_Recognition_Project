# Extension 6.1: Adaptive Re-ranking

## Overview

Implementation of **Adaptive Phase 2** (image matching) for Visual Place Recognition.

**Goal**: Reduce computational cost by selectively applying expensive image matching only to "hard" queries.

### Pipeline

```
Standard (Non-Adaptive):
  1. Retrieval (fast)
  2. Image Matching on top-20 (expensive, ~45 sec/query)
  3. Re-ranking

Adaptive (Extension 6.1):
  1. Retrieval (fast)
  2. Image Matching ONLY on top-1 (fast, ~2 sec/query)
  3. Logistic Regression: P(correct | inliers_top1)
  4. Decision:
     - If P > threshold → SKIP remaining matching (EASY query)
     - If P ≤ threshold → Complete matching on top-20 (HARD query)
  5. Re-ranking

Expected Savings: ~70% queries are easy → 68% total cost reduction ⚡
```

---

## Project Structure

```
extension_6_1/
├── README.md                           # This file
│
├── config/
│   └── paths_config.json               # Configuration (paths, hyperparams)
│
├── utils/
│   ├── __init__.py
│   ├── data_loader.py                  # Load .torch and .txt files
│   ├── metrics.py                      # AUPRC, accuracy, etc.
│   └── visualization.py                # Plot histograms
│
├── scripts/
│   ├── step1_inliers_analysis.py       # Analyze inliers distribution
│   ├── step2_train_lr.py               # Train logistic regression models
│   ├── step3_validate_threshold.py     # Find optimal threshold
│   └── step4_adaptive_inference.py     # Run adaptive inference
│
└── results/
    ├── step1_analysis/                 # Output from Step 1
    ├── models/                         # Output from Step 2
    ├── validation/                     # Output from Step 3
    └── inference/                      # Output from Step 4-5
```

---

## Steps

### Step 1: Analyze Inliers Distribution

**Goal**: Understand how inliers_top1 correlate with query correctness.

```bash
cd extension_6_1
python scripts/step1_inliers_analysis.py
```

**Outputs**:
- `results/step1_analysis/inliers_{matcher}.pkl` — Training data (X, y)
- `results/step1_analysis/distribution_{matcher}.png` — Histograms
- `results/step1_analysis/step1_summary.txt` — Statistics

**What it does**:
1. For each matcher (loftr, superglue, lightglue)
2. Load ALL training sets (svox_sun, svox_night, gsv_xs) × ALL VPR models
3. For each query:
   - Extract `inliers_top1` from .torch file
   - Extract `geo_dist_top1` from .txt file
   - Label: `y = 1` if `geo_dist_top1 ≤ 25m`, else `0`
4. Save (X, y) pairs as pickle files
5. Compute statistics and plot distributions

---

### Step 2: Train Logistic Regression Models

**Goal**: Train matcher-specific classifiers: `inliers_top1 → P(correct)`.

```bash
python scripts/step2_train_lr.py
```

**Outputs**:
- `results/models/lr_models.pkl` — Trained LR models (1 per matcher)
- `results/models/validation_metrics.txt` — AUPRC, accuracy scores

**What it does**:
1. Load data from Step 1 (`inliers_*.pkl`)
2. For each matcher:
   - Split: 80% train, 20% validation
   - Train LogisticRegression on (inliers_top1 → is_correct)
   - Evaluate: AUPRC, AUC-ROC, Accuracy on validation set
3. Save models in dictionary: `{'loftr': lr_loftr, 'superglue': lr_superglue, ...}`

---

### Step 3: Validate and Select Threshold

**Goal**: Find optimal probability threshold for easy/hard decision.

```bash
python scripts/step3_validate_threshold.py
```

**Outputs**:
- `results/validation/threshold_analysis.txt` — Threshold sweep results
- `results/validation/threshold_curves.png` — Cost vs quality trade-off plots

**What it does**:
1. Load LR models from Step 2
2. On SF-XS validation set:
   - For each query: compute P(correct) using LR
   - Try thresholds: [0.5, 0.6, 0.7, 0.8, 0.9]
   - For each threshold: compute percentage of EASY queries
3. Plot: threshold vs percentage_easy vs quality_metrics
4. Recommend optimal threshold (e.g., 0.75 = 70% easy queries)

---

### Step 4-5: Adaptive Inference on Test Sets

**Goal**: Run adaptive matching on test sets and measure performance.

```bash
python scripts/step4_adaptive_inference.py --threshold 0.75 --dataset tokyo
```

**Outputs**:
- `results/inference/results_{dataset}.txt` — Recall@N with adaptive approach
- `results/inference/cost_analysis.txt` — Time saved, queries skipped
- `results/inference/timing_report.txt` — Execution time breakdown

**What it does**:
1. For each test query:
   - Retrieval (top-20)
   - Image Matching on top-1 ONLY
   - LR prediction: P(correct)
   - **Decision**:
     - If P > threshold: Use retrieval-only results (SKIP top-19 matching)
     - If P ≤ threshold: Run full matching on top-20, re-rank
2. Compute: Recall@1, @5, @10, @20
3. Compare with baseline (no adaptive)

---

## Configuration

Edit `config/paths_config.json` to customize:

```json
{
  "input": {
    "base_path": "...",              // Path to VPR-methods-evaluation
    "training_datasets": [...]       // Training datasets
  },
  "vpr_models": ["netvlad", ...],   // VPR models to use
  "matchers": ["loftr", ...],        // Image matchers
  "hyperparams": {
    "threshold_dist": 25,            // Distance threshold (meters)
    "top_k": 20,                     // Top-K for re-ranking
    "train_val_split": 0.8           // Train/validation split
  }
}
```

---

## Usage Example

```bash
# Step 1: Analyze
python scripts/step1_inliers_analysis.py

# Step 2: Train models
python scripts/step2_train_lr.py

# Step 3: Find threshold
python scripts/step3_validate_threshold.py

# Step 4: Evaluate on test sets
python scripts/step4_adaptive_inference.py --threshold 0.75 --dataset tokyo
python scripts/step4_adaptive_inference.py --threshold 0.75 --dataset sf_xs_test
```

---

## Expected Results

**Cost Savings**:
```
Assuming:
- 70% queries classified as EASY
- 30% queries classified as HARD

Per-query time:
  EASY:  1 sec (retrieval) + 2 sec (LoFTR top-1) + 0.1 sec (LR) = 3.1 sec
  HARD: 1 sec (retrieval) + 45 sec (LoFTR 20) = 46 sec
  
Average: 0.7 × 3.1 + 0.3 × 46 ≈ 15 sec/query

Vs. baseline (no adaptive): 47 sec/query

SAVINGS: 32/47 ≈ 68% ⚡
```

**Performance**:
- Recall@N should stay similar or improve slightly
- Trade-off: small performance loss for massive cost savings

---

## Notes

- **Matcher-specific models**: One LR per matcher (loftr, superglue, lightglue)
- **Dataset-agnostic**: Trained on all training datasets together
- **Threshold selection**: Validate on SF-XS val set, not on training set
- **Cost analysis**: Include time for top-1 matching (not just decision overhead)

---

## References

- Extension 6.1 specification: `../../../Are_You_Sure_You_Are_in_the_Right_Place.pdf` (Section 6.1)
- Base methods: Sferrazza et al. "To match or not to match" (CVPR 2025)


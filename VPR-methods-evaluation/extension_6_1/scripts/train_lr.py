"""
Train matcher-specific logistic regression models to predict P(correct | inliers_top1).
Models are trained on data from inliers_analysis and validated on a held-out validation set.

Input:
  - inliers_{matcher}.pkl

Output:
  - lr_models.pkl: Dictionary of trained LogisticRegression models
  - validation_metrics.txt: Performance metrics (AUPRC, AUC-ROC, Accuracy)
"""

import json
import pickle
import numpy as np
from pathlib import Path
import sys
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

# Add utils to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.metrics import compute_auprc_and_accuracy


def main():
    # Load config
    config_path = Path(__file__).parent.parent / "config" / "paths_config.json"
    with open(config_path) as f:
        cfg = json.load(f)
    
    base_path = cfg['input']['base_path']
    matchers = cfg['matchers']
    train_val_split = cfg['hyperparams']['train_val_split']
    
    # Input directory
    inliers_analysis_dir = Path(base_path) / cfg['output']['base_dir'] / "inliers_analysis"
    
    # Output directory
    output_dir = Path(base_path) / cfg['output']['base_dir'] / "lr_models"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    summary_lines = []
    summary_lines.append("=" * 80)
    summary_lines.append("EXTENSION 6.1 - LOGISTIC REGRESSION TRAINING")
    summary_lines.append("=" * 80)
    summary_lines.append("")
    
    # Dictionary to store all trained models
    models = {}
    
    # Process each matcher
    for matcher in matchers:
        print(f"\n{'='*80}")
        print(f"Training LR model for matcher: {matcher}")
        print(f"{'='*80}")
        
        # Load data from inliers_analysis
        pkl_path = inliers_analysis_dir / f"inliers_{matcher}.pkl"
        
        if not pkl_path.exists():
            print(f"File not found: {pkl_path}")
            continue
        
        try:
            with open(pkl_path, 'rb') as f:
                data = pickle.load(f)
            X = data['X']
            y = data['y']
        except Exception as e:
            print(f"Error loading {pkl_path}: {e}")
            continue
        
        print(f"\nLoaded {len(X)} samples from {pkl_path}")
        print(f"Correct: {sum(y)} ({100*sum(y)/len(y):.1f}%)")
        print(f"Wrong: {len(y)-sum(y)} ({100*(len(y)-sum(y))/len(y):.1f}%)")
        
        # Split into train/validation
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, 
            test_size=(1 - train_val_split),
            random_state=42,
            stratify=y  # Stratified split to keep class balance
        )
        
        print(f"\nTrain/Val split ({train_val_split:.0%}/{1-train_val_split:.0%}):")
        print(f"Train: {len(X_train)} samples")
        print(f"Val:   {len(X_val)} samples")
        
        # ----------- TRAIN LOGISTIC REGRESSION ------------
        print(f"\nTraining LogisticRegression...")
        lr = LogisticRegression(random_state=42, max_iter=1000)
        
        # Reshape for sklearn (needs 2D input)
        X_train_reshaped = X_train.reshape(-1, 1)
        X_val_reshaped = X_val.reshape(-1, 1)
        
        lr.fit(X_train_reshaped, y_train)
        print(f"Model trained")
        print(f"Coefficient: {lr.coef_[0][0]:.6f}")
        print(f"Intercept: {lr.intercept_[0]:.6f}")
        
        # Evaluate on validation set
        y_pred_proba = lr.predict_proba(X_val_reshaped)[:, 1]
        metrics = compute_auprc_and_accuracy(y_val, y_pred_proba)
        
        print(f"\nValidation Metrics:")
        print(f"AUPRC: {metrics['auprc']:.4f}")
        print(f"AUC-ROC: {metrics['auc_roc']:.4f}")
        print(f"Accuracy: {metrics['accuracy']:.4f}")
        
        # Store model
        models[matcher] = lr
        
        # Add to summary
        summary_lines.append(f"\n{'─'*80}")
        summary_lines.append(f"MATCHER: {matcher}")
        summary_lines.append(f"{'─'*80}")
        summary_lines.append(f"Training samples: {len(X_train)}")
        summary_lines.append(f"Validation samples: {len(X_val)}")
        summary_lines.append(f"\nModel Parameters:")
        summary_lines.append(f"Coefficient: {lr.coef_[0][0]:.6f}")
        summary_lines.append(f"Intercept: {lr.intercept_[0]:.6f}")
        summary_lines.append(f"\nValidation Performance:")
        summary_lines.append(f"AUPRC: {metrics['auprc']:.4f}")
        summary_lines.append(f"AUC-ROC: {metrics['auc_roc']:.4f}")
        summary_lines.append(f"Accuracy: {metrics['accuracy']:.4f}")
    
    # Save all models
    models_path = output_dir / "lr_models.pkl"
    with open(models_path, 'wb') as f:
        pickle.dump(models, f)
    print(f"\nAll models saved: {models_path}")
    
    # Save summary
    summary_lines.append(f"\n{'='*80}")
    summary_path = output_dir / "validation_metrics.txt"
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(summary_lines))

if __name__ == "__main__":
    main()

import numpy as np
from sklearn.metrics import average_precision_score, accuracy_score, roc_auc_score
from typing import Dict


def compute_auprc_and_accuracy(y_true: np.ndarray, y_pred_proba: np.ndarray) -> Dict:
    """
    Compute AUPRC, AUC-ROC, and Accuracy metrics.
    
    Args:
        y_true: True labels (0 or 1)
        y_pred_proba: Predicted probabilities for class 1
    
    Returns:
        Dictionary with metrics
    """
    
    y_pred_binary = (y_pred_proba >= 0.5).astype(int)
    
    metrics = {
        'auprc': average_precision_score(y_true, y_pred_proba),
        'auc_roc': roc_auc_score(y_true, y_pred_proba),
        'accuracy': accuracy_score(y_true, y_pred_binary),
    }
    
    return metrics

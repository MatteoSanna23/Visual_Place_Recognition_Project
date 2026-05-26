import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path


def plot_inliers_distribution(
    X_correct: np.ndarray,
    X_wrong: np.ndarray,
    matcher: str,
    output_path: Path,
    bins: int = 30
) -> None:
    """
    Plot histogram of inliers distribution for correct vs wrong queries.

    Args:
        X_correct: Inliers for correct queries
        X_wrong: Inliers for wrong queries
        matcher: Matcher name (for title)
        output_path: Path to save figure
        bins: Number of histogram bins
    """
    
    plt.figure(figsize=(12, 7))
    plt.hist(X_correct, bins=bins, alpha=0.6, label='Correct Queries', color='green', edgecolor='black')
    plt.hist(X_wrong, bins=bins, alpha=0.6, label='Wrong Queries', color='red', edgecolor='black')
    
    plt.xlabel('Number of Inliers (top-1)', fontsize=12)
    plt.ylabel('Count', fontsize=12)
    plt.title(f'Inliers Distribution: {matcher.upper()}\n(Correct vs Wrong Queries)', fontsize=14)
    plt.legend(fontsize=11)
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Plot saved: {output_path}")
    plt.close()

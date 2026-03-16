import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, average_precision_score
from sklearn.inspection import permutation_importance
import pandas as pd

def plot_precision_recall(y_true, y_prob, save_dir):
    """
    Plot Precision–Recall curve.
    """
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    ap = average_precision_score(y_true, y_prob)

    plt.figure()
    plt.plot(recall, precision)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(f"Precision–Recall curve (AP={ap:.3f})")
    plt.grid(True)
    plt.savefig(os.path.join(save_dir, "precision_recall_curve.png"))
    plt.close()

def plot_feature_importance(model, X_df: pd.DataFrame, y, save_dir: str):
    """
    Compute and plot permutation feature importance.

    Parameters:
    -----------
    model : trained estimator
        Model used for predictions.
    X : pandas.DataFrame
        Feature matrix with column names as feature labels.
    y : array-like
        True labels corresponding to X.
    save_dir : str
        Directory path to save the plot.
    """
        
    # Use DataFrame column names for plotting
    feature_labels = X_df.columns
    X = X_df.values

    # Compute permutation importance
    result = permutation_importance(
        model,
        X,
        y,
        n_repeats=10,
        random_state=42,
        n_jobs=-1
    )

    importance_df = pd.DataFrame({
        "feature": feature_labels,
        "importance": result.importances_mean
    }).sort_values("importance", ascending=True)

    # Plot horizontal bar chart
    plt.figure()
    plt.barh(importance_df["feature"], importance_df["importance"])
    plt.xlabel("Permutation Importance")
    plt.title("Feature Importance")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "feature_importance.png"))
    plt.close()

def plot_threshold_tradeoff(results, save_dir):
    """
    Plot precision and recall as a function of threshold.
    """
    plt.figure()
    plt.plot(results["threshold"], results["incident_recall"], label="indicent recall")
    plt.plot(results["threshold"], results["alert_precision"], label="alert precision")
    plt.xlabel("Threshold")
    plt.ylabel("Score")
    plt.title("Threshold tradeoff")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(save_dir, "threshold_tradeoff.png"))
    plt.close()
import os
import argparse
import numpy as np
import pandas as pd
import joblib

from plots import plot_precision_recall, plot_feature_importance, plot_threshold_tradeoff


def find_incidents(labels):
    """
    Find indices where incidents actually start.

    In this dataset:
        - label = 1 represents prediction horizon (early-warning window) before an incident
        - Actual incident occurs right after 1-block ends (transition 1→0)
        - Incident rows removed from dataset

    Returns list of integers: indices where each incident starts
    """
    incidents = []
    start = False

    for i, val in enumerate(labels):
        # incidents entered the end of the predicting window
        if val == 1 and not start:
            start = True
        # incident just occurred (it is no longer in the predicting window)
        if val == 0 and start:
            incidents.append(i)
            start = False

    return incidents


import numpy as np
import pandas as pd

import numpy as np
import pandas as pd

def evaluate_model(model, X, y, threshold=0.5):
    """
    Incident-based evaluation of model predictions.

    Returns:
        - results_df: DataFrame with metrics
        - y_prob: predicted probabilities for PR analysis

    Metrics:
        - alert_precision: TP_alert / (TP_alert + FP_alert)
        - incident_recall: detected_incidents / total_incidents
        - avg_detection_distance: mean lead time per detected incident
        - detected_incidents: number of detected incidents
        - total_incidents: total number of incidents
        - alerts_count: total number of alerts (TP + FP)
        - valid_alerts: true positives
        - invalid_alerts: false positives
        - false_alert_rate: FP / non-incident timesteps
    """
    # predict probabilities and binarize with threshold
    y_prob = model.predict_proba(X)[:, 1]
    y_pred = (y_prob >= threshold).astype(int)

    # identify incidents indices
    incidents = find_incidents(y)
    incident_dict = {start: None for start in incidents}

    # initialize counters
    true_positives = 0
    false_positives = 0
    current_incident_idx = 0
    current_incident = incidents[current_incident_idx] if incidents else None

    # iterate over predictions to track TP, FP, and first alert per incident
    for i, pred in enumerate(y_pred):
        # update current incident if past
        if current_incident is not None and i == current_incident:
            current_incident_idx += 1
            current_incident = incidents[current_incident_idx] if current_incident_idx < len(incidents) else None

        if pred == 1:
            if y[i] == 0:
                false_positives += 1
            else:
                true_positives += 1
                if current_incident and incident_dict[current_incident] is None:
                    incident_dict[current_incident] = current_incident - i

    # aggregate metrics
    total_incidents = len(incidents)
    detected_incidents = sum(1 for v in incident_dict.values() if v is not None)
    lead_times = [v for v in incident_dict.values() if v is not None]
    avg_distance = np.mean(lead_times) if lead_times else np.nan

    valid_alerts = true_positives
    invalid_alerts = false_positives
    alerts_count = valid_alerts + invalid_alerts

    alert_precision = valid_alerts / alerts_count if alerts_count > 0 else 0
    incident_recall = detected_incidents / total_incidents if total_incidents > 0 else 0
    non_incident_steps = (y == 0).sum()
    false_alert_rate = invalid_alerts / non_incident_steps if non_incident_steps > 0 else np.nan

    # create results DataFrame
    results_df = pd.DataFrame([{
        "threshold": threshold,
        "alert_precision": alert_precision,
        "incident_recall": incident_recall,
        "avg_detection_distance": avg_distance,
        "detected_incidents": detected_incidents,
        "total_incidents": total_incidents,
        "alerts_count": alerts_count,
        "valid_alerts": valid_alerts,
        "invalid_alerts": invalid_alerts,
        "false_alert_rate": false_alert_rate
    }])

    return results_df, y_prob


def sweep_thresholds(model, X, y, thresholds):
    """
    Evaluate model over multiple thresholds and combine results.

    Returns:
        - results: DataFrame with metrics for each threshold
        - y_prob: predicted probabilities for PR analysis
    """
    rows = []
    y_prob = model.predict_proba(X)[:, 1]

    for t in thresholds:
        # evaluate each threshold
        res, _ = evaluate_model(model, X, y, t)
        rows.append(res)

    # concatenate results into single DataFrame
    results = pd.concat(rows, ignore_index=True)
    return results, y_prob


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate trained Gradient Boosting model.")

    parser.add_argument("--dataset", type=str, default="../data/processed/test.csv",
                        help="Test dataset CSV (features + label).")
    parser.add_argument("--model", type=str, default="../models/model.pkl",
                        help="Path to trained model.")
    parser.add_argument("--threshold_sweep", action="store_true",
                        help="Run threshold sweep")
    parser.add_argument("--threshold", type=float, default=0.5,
                        help="Single threshold for evaluation")
    parser.add_argument("--thresholds", type=float, nargs=3,
                        metavar=('start', 'stop', 'num'),
                        help="Threshold sweep parameters for linspace")
    parser.add_argument("--save_dir", type=str, default="../results",
                        help="Directory to save plots and results")
    args = parser.parse_args()

    if not os.path.exists(args.dataset):
        raise FileNotFoundError(f"Dataset not found: {args.dataset}")

    # Load test dataset
    dataset = pd.read_csv(args.dataset)
    X_df = dataset.drop(columns=["label"])
    X = X_df.values
    y = dataset["label"].values

    # Load pre-trained model
    model = joblib.load(args.model)
    os.makedirs(args.save_dir, exist_ok=True)

    # Evaluate either single threshold or sweep thresholds
    if args.threshold_sweep:
        if args.thresholds:
            thresholds = np.linspace(*args.thresholds)
        else:
            thresholds = np.linspace(0.05, 0.95, 19)

        results, y_prob = sweep_thresholds(model, X, y, thresholds)
        print("\nExperiment results:")
        print(results)
        plot_threshold_tradeoff(results, args.save_dir)

    else:
        results, y_prob = evaluate_model(model, X, y, threshold=args.threshold)
        print("\nEvaluation results:")
        print(results)

    # always generate plots for PR curve and feature importance
    plot_precision_recall(y, y_prob, args.save_dir)
    plot_feature_importance(model, X_df, y, args.save_dir)

    print(f"\nPlots saved to {args.save_dir}")
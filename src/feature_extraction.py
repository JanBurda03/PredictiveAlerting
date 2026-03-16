import pandas as pd
import numpy as np
import argparse
import os
from sklearn.model_selection import train_test_split


def extract_features(df, W=150, H=60):
    """
    Generate statistical features from sliding windows of a time series and
    assign a binary label based on whether an incident occurs within the next H steps.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing 'TimeStamp', 'Value', and 'Label'.
    W : int
        Window size used to compute features.
    H : int
        Prediction horizon. If any label == 1 appears within the next H samples,
        the output label for the window is 1.

    Returns
    -------
    X : pandas.DataFrame
        Feature matrix with labels included.
    """

    # Chronological order ensured
    df = df.sort_values('TimeStamp').reset_index(drop=True)

    # Convert columns to numpy arrays for fast numerical operations
    values = df['Value'].astype(float).to_numpy()
    labels = df['Label'].astype(int).to_numpy()
    
    # If there is not enough data to form a window and prediction horizon, empty is returned
    n = len(values)
    if n < W + H:
        cols = [
            'mean','std','min','max','median','perc25','perc75','val_range',
            'slope','trend_strength','mean_diff','std_diff','max_diff',
            'last_val','z_score','spike_ratio','volatility','label'
        ]
        return pd.DataFrame(columns=cols)

    # Rolling statistics on raw values
    s = pd.Series(values)
    mean = s.rolling(W).mean().to_numpy()           # rolling mean
    std = s.rolling(W).std(ddof=0).to_numpy()       # rolling std
    min_ = s.rolling(W).min().to_numpy()            # rolling min
    max_ = s.rolling(W).max().to_numpy()            # rolling max
    median = s.rolling(W).median().to_numpy()       # rolling median
    perc25 = s.rolling(W).quantile(0.25).to_numpy() # 25th percentile
    perc75 = s.rolling(W).quantile(0.75).to_numpy() # 75th percentile
    val_range = max_ - min_                         # values range within window
    last_val_full = values.copy()                   # last value in window

    # Differences
    diffs = pd.Series(values).diff()                       
    mean_diff_full = diffs.rolling(W-1).mean().to_numpy()       # mean of differences
    std_diff_full = diffs.rolling(W-1).std(ddof=0).to_numpy()   # std of differences
    max_diff_full = diffs.abs().rolling(W-1).max().to_numpy()   # max absolute difference

    # Linear trend
    t = np.arange(W)
    k = t - t.mean()                                    # centered time vector
    denom = (k**2).sum()                                # denominator for slope
    conv = np.convolve(values, k[::-1], mode='valid')   # convolution to compute slope
    slope_full = np.full(n, np.nan)
    slope_full[W-1: W-1 + len(conv)] = conv / (denom + 1e-12)  # slope aligned to window end

    # Derived normalized features
    eps = 1e-12                                             # eps to avoid division by zero
    trend_strength_full = (slope_full * W) / (mean + eps)   # slope relative to window mean
    z_score_full = (last_val_full - mean) / (std + eps)     # standardized last value
    spike_ratio_full = last_val_full / (median + eps)       # last value / median ratio
    volatility_full = std / (mean + eps)                    # relative variability

    # Valid window end indices, skips first W-1 and last H (not enough future for label)
    end_idx = np.arange(W - 1, n - H)

    # Construct feature DataFrame
    features = {
        'mean': mean[end_idx],
        'std': std[end_idx],
        'min': min_[end_idx],
        'max': max_[end_idx],
        'median': median[end_idx],
        'perc25': perc25[end_idx],
        'perc75': perc75[end_idx],
        'val_range': val_range[end_idx],
        'slope': slope_full[end_idx],
        'trend_strength': trend_strength_full[end_idx],
        'mean_diff': mean_diff_full[end_idx],
        'std_diff': std_diff_full[end_idx],
        'max_diff': max_diff_full[end_idx],
        'last_val': last_val_full[end_idx],
        'z_score': z_score_full[end_idx],
        'spike_ratio': spike_ratio_full[end_idx],
        'volatility': volatility_full[end_idx],
    }

    X = pd.DataFrame(features)

    # Compute horizon labels
    conv_labels = np.convolve(labels, np.ones(H, dtype=int), mode='valid')
    label_positions = end_idx + 1
    label_sums = conv_labels[label_positions]
    X['label'] = (label_sums > 0).astype(int)

    # Drop rows ending during incident
    incident_mask = labels.astype(bool)
    valid_rows = ~incident_mask[end_idx]
    X = X[valid_rows].reset_index(drop=True)

    return X

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare sliding windows for MongoDB metrics.")
    parser.add_argument("--input_file", type=str, default="../data/raw/dataset.csv",
                        help="Path to the original CSV file (default='../data/raw/dataset.csv').")
    parser.add_argument("--W", type=int, default=150, help="Window size (number of past steps).")
    parser.add_argument("--H", type=int, default=60, help="Prediction horizon (number of future steps).")
    parser.add_argument("--output_dir", type=str, default="../data/processed",
                        help="Directory to save the dataset.")
    parser.add_argument("--split", action="store_true",
                        help="Whether to split dataset into train/test.")
    parser.add_argument("--test_size", type=float, default=0.5,
                        help="Fraction of data for testing if --split is True.")

    args = parser.parse_args()

    if not os.path.exists(args.input_file):
        raise FileNotFoundError(f"Input file not found: {args.input_file}")

    df = pd.read_csv(args.input_file, parse_dates=['TimeStamp'])

    dataset = extract_features(df, W=args.W, H=args.H)
    print(f"Created sliding windows, shape = {dataset.shape}")

    os.makedirs(args.output_dir, exist_ok=True)

    if args.split:
        # Split dataset into train/test
        train_set, test_set = train_test_split(
            dataset,
            test_size=args.test_size,
            shuffle=False)

        train_set.to_csv(os.path.join(args.output_dir, "train.csv"), index=False)
        test_set.to_csv(os.path.join(args.output_dir, "test.csv"), index=False)
        print(f"Saved train.csv ({train_set.shape}) and test.csv ({test_set.shape})")
        
    else:
        dataset.to_csv(os.path.join(args.output_dir, "dataset.csv"), index=False)
        print(f"Saved dataset.csv ({dataset.shape})")

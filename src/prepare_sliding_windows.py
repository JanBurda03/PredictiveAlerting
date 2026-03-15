import pandas as pd
import numpy as np
import argparse
import os

def create_sliding_windows(file_path, W=12, H=3):
    """
    Create sliding windows from a timestamped time series CSV with labels.

    Args:
        file_path (str): Path to the CSV file.
        W (int): Number of previous steps for input features.
        H (int): Number of future steps to look ahead for label.

    Returns:
        X (np.ndarray): Array of shape (num_windows, W) with input sequences.
        y (np.ndarray): Array of shape (num_windows,) with binary labels.
    """
    df = pd.read_csv(file_path, parse_dates=['TimeStamp'])
    df = df.sort_values('TimeStamp').reset_index(drop=True)

    values = df['Value'].values
    labels = df['Label'].values

    X, y_windows = [], []
    num_samples = len(df) - W - H + 1

    for i in range(num_samples):
        X_window = values[i:i+W]
        y_window = labels[i+W:i+W+H]
        y_label = 1 if np.any(y_window == 1) else 0
        X.append(X_window)
        y_windows.append(y_label)

    return np.array(X), np.array(y_windows)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare sliding windows for MongoDB metrics.")
    parser.add_argument("--input_file", type=str, default="../data/raw/dataset.csv",
                        help="Path to the original CSV file (default='../data/raw/dataset.csv').")
    parser.add_argument("--W", type=int, default=12, help="Window size (number of past steps).")
    parser.add_argument("--H", type=int, default=3, help="Prediction horizon (number of future steps).")
    parser.add_argument("--save", action="store_true", help="Whether to save the dataset.")
    parser.add_argument("--output_dir", type=str, default="../data/processed",
                        help="Directory to save the dataset.")

    args = parser.parse_args()

    if not os.path.exists(args.input_file):
        raise FileNotFoundError(f"Input file not found: {args.input_file}")

    X, y = create_sliding_windows(args.input_file, W=args.W, H=args.H)
    print(f"Created sliding windows: X shape = {X.shape}, y shape = {y.shape}")

    if args.save:
        os.makedirs(args.output_dir, exist_ok=True)
        np.savez(os.path.join(args.output_dir, "dataset.npz"), X=X, y=y)
        print(f"Saved dataset.npz to {args.output_dir}")
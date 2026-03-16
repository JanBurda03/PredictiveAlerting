import os
import argparse
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
import joblib


def train_model(X_train, y_train):
    """Train HistGradientBoostingClassifier."""
    model = HistGradientBoostingClassifier(
        max_iter=400,
        learning_rate=0.03,
        max_depth=2,
        random_state=42
    )
    model.fit(X_train, y_train)
    return model


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Train Gradient Boosting model.")
    parser.add_argument("--dataset", type=str, default="../data/processed/train.csv")
    parser.add_argument("--save_model", action="store_true")
    parser.add_argument("--save_dir", type=str, default="../models")

    args = parser.parse_args()

    if not os.path.exists(args.dataset):
        raise FileNotFoundError(f"Dataset not found: {args.dataset}")

    dataset = pd.read_csv(args.dataset)

    X = dataset.drop(columns=["label"])
    y = dataset["label"]

    X = X.values
    y = y.values

    print("Training model...")
    model = train_model(X, y)

    if args.save_model:
        os.makedirs(args.save_dir, exist_ok=True)
        joblib.dump(model, os.path.join(args.save_dir, "model.pkl"))
        print(f"Model saved to {args.save_dir}")
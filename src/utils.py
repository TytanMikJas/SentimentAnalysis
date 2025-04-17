import os
import json
import logging
from typing import Tuple, List, Optional

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import wandb
import yaml

from sklearn.metrics import confusion_matrix

# ------------------------------
# Logging Configuration
# ------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

# ------------------------------
# Constants
# ------------------------------
SEPHORA_DATASET = "sephora"
RT_POLARITY_DATASET = "rt-polarity"
TRAIN_FILENAME = "train.pkl"
TEST_FILENAME = "test.pkl"

# ------------------------------
# Utility Functions
# ------------------------------


def read_reviews_txt_file(path: str) -> pd.DataFrame:
    """
    Reads a rt-polarity dataset where each line is a review.

    Args:
        path (str): Path to the review text file.

    Returns:
        pd.DataFrame: DataFrame with a 'review_text' column.
    """
    try:
        with open(path, "r", encoding="ISO-8859-1") as f:
            lines = f.readlines()
        lines = [line.strip() for line in lines if line.strip()]
        return pd.DataFrame({"review_text": lines})
    except FileNotFoundError as e:
        logging.error(f"File not found: {path}")
        return pd.DataFrame()


def load_raw_data(path_to_raw_data: str, dataset: str) -> pd.DataFrame:
    """
    Loads raw data for a given dataset.

    Args:
        path_to_raw_data (str): Path to the raw dataset.
        dataset (str): Dataset identifier.

    Returns:
        pd.DataFrame: Loaded dataset.
    """
    if dataset == SEPHORA_DATASET:
        csv_files = [
            f
            for f in os.listdir(path_to_raw_data)
            if f.endswith(".csv") and f.startswith("reviews")
        ]
        chunks = []
        logging.info("Loading Sephora raw data...")
        # due to slow computer and running code in docker env I had to divide the process into chunks :)
        for f in csv_files:
            file_path = os.path.join(path_to_raw_data, f)
            for chunk in pd.read_csv(file_path, chunksize=1000):
                chunks.append(chunk)
        reviews_df = pd.concat(chunks, ignore_index=True)
        product_info_df = pd.read_csv(
            os.path.join(path_to_raw_data, "product_info.csv")
        )
        df = pd.merge(
            reviews_df, product_info_df, on="product_id", suffixes=("", "_drop")
        )
        df.drop(
            columns=[col for col in df.columns if col.endswith("_drop")], inplace=True
        )
    elif dataset == RT_POLARITY_DATASET:
        logging.info("Loading RT Polarity raw data...")
        pos_reviews = read_reviews_txt_file(
            os.path.join(path_to_raw_data, "rt-polarity.pos")
        )
        neg_reviews = read_reviews_txt_file(
            os.path.join(path_to_raw_data, "rt-polarity.neg")
        )
        pos_reviews["label"] = 1
        neg_reviews["label"] = 0
        df = pd.concat([pos_reviews, neg_reviews], ignore_index=True)
    else:
        raise ValueError(f"Unknown dataset: {dataset}")

    return df


def save_dataset(df: pd.DataFrame, path: str) -> None:
    """Saves a DataFrame to a pickle file."""
    try:
        logging.info(f"Saving dataset to {path}")
        df.to_pickle(path)
    except Exception as e:
        logging.error(f"Failed to save dataset: {e} to path {path}")


def load_dataset(path: str) -> pd.DataFrame:
    """Loads a DataFrame from a pickle file."""
    try:
        logging.info(f"Loading dataset from {path}")
        return pd.read_pickle(path)
    except FileNotFoundError:
        logging.error(f"Dataset not found at: {path}")
        return pd.DataFrame()


def save_train_test_data(
    train: pd.DataFrame, test: pd.DataFrame, path: str, dataset: str
) -> None:
    """Saves train and test sets as pickle files."""
    dataset_path = os.path.join(path, dataset)
    os.makedirs(dataset_path, exist_ok=True)
    save_dataset(train, os.path.join(dataset_path, TRAIN_FILENAME))
    save_dataset(test, os.path.join(dataset_path, TEST_FILENAME))


def load_train_test_data(path: str, dataset: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Loads train and test datasets from disk."""
    dataset_path = os.path.join(path, dataset)
    train = load_dataset(os.path.join(dataset_path, TRAIN_FILENAME))
    test = load_dataset(os.path.join(dataset_path, TEST_FILENAME))
    return train, test


def plot_confusion_matrix(
    y_train: Optional[pd.Series] = None,
    y_hat_train: Optional[pd.Series] = None,
    y_test: Optional[pd.Series] = None,
    y_hat_test: Optional[pd.Series] = None,
) -> None:
    """
    Plots and logs confusion matrices to Weights & Biases.
    """

    def _plot_and_log(cm, title, cmap):
        plt.figure(figsize=(6, 4))
        sns.heatmap(cm, annot=True, fmt="d", cmap=cmap)
        plt.title(title)
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        if wandb.run is not None:
            wandb.log({title: wandb.Image(plt)})
        plt.close()

    if y_train is not None and y_hat_train is not None:
        cm_train = confusion_matrix(y_train, y_hat_train)
        _plot_and_log(cm_train, "Confusion Matrix - Train", "Blues")

    if y_test is not None and y_hat_test is not None:
        cm_test = confusion_matrix(y_test, y_hat_test)
        _plot_and_log(cm_test, "Confusion Matrix - Test", "Reds")


def save_f1_score(
    f1_scores: List[float], experiment_name: str, metrics_file: str
) -> None:
    """Saves F1 scores to a JSON file."""
    try:
        if os.path.exists(metrics_file):
            with open(metrics_file, "r") as f:
                existing_data = json.load(f)
        else:
            existing_data = {}

        existing_data[f"avg weighted f1_score of {experiment_name}"] = f1_scores

        with open(metrics_file, "w") as f:
            json.dump(existing_data, f, indent=4)
    except Exception as e:
        logging.error(f"Failed to save F1 scores: {e}")


def log_info(text: str) -> None:
    """Logs a simple info message."""
    logging.info(text)


def get_params(dataset: str) -> Tuple[dict, dict]:
    """Loads parameters from common and dataset-specific YAML config files."""
    try:
        with open("params.yaml", "r") as file:
            common_params = yaml.safe_load(file)

        with open(os.path.join("configs", f"{dataset}.yaml"), "r") as file:
            custom_params = yaml.safe_load(file)

        return common_params, custom_params
    except FileNotFoundError as e:
        logging.error(f"Configuration file not found: {e}")
        return {}, {}

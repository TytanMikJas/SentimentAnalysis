import os
import pandas as pd
import json
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)


def load_raw_data(path_to_raw_data: str):
    csv_files = [
        f
        for f in os.listdir(path_to_raw_data)
        if f.endswith(".csv") and f.startswith("reviews")
    ]
    chunks = []
    logging.info("Beginning to load raw data")
    for f in csv_files:
        file_path = os.path.join(path_to_raw_data, f)
        for chunk in pd.read_csv(file_path, chunksize=1000):
            chunks.append(chunk)
    logging.info("Creating dataframes from raw data")
    reviews_df = pd.concat(chunks, ignore_index=True)
    product_info_df = pd.read_csv(os.path.join(path_to_raw_data, "product_info.csv"))

    return reviews_df, product_info_df


def save_dataset(df: pd.DataFrame, path_to_processed_data: str):
    logging.info("Saving dataset")
    df.to_pickle(os.path.join(path_to_processed_data))


def load_dataset(path_to_processed_data: str):
    logging.info("Loading dataset")
    return pd.read_pickle(os.path.join(path_to_processed_data))


def save_train_test_data(
    train: pd.DataFrame, test: pd.DataFrame, path_to_split_data: str
):
    logging.info("Saving train and test data")
    train.to_pickle(os.path.join(path_to_split_data, "train.pkl"))
    test.to_pickle(os.path.join(path_to_split_data, "test.pkl"))


def load_train_test_data(path_to_split_data: str):
    logging.info("Loading train and test data")
    os.makedirs(path_to_split_data, exist_ok=True)
    training_data = pd.read_pickle(os.path.join(path_to_split_data, "train.pkl"))
    test_data = pd.read_pickle(os.path.join(path_to_split_data, "test.pkl"))
    return training_data, test_data


def save_f1_score(f1_scores: list, metrics_file: str):
    logging.info("Saving f1 scores")
    with open(os.path.join(metrics_file), "w") as f:
        json.dump({"f1_scores": f1_scores}, f)

import pandas as pd
import sys
import yaml
from sklearn.model_selection import train_test_split
from src.utils import load_dataset, save_train_test_data


def split_data(
    path_to_processed_data, path_to_split_data, test_size, stratify, label_column
):
    dataset = load_dataset(path_to_processed_data)

    X = dataset.drop(columns=[label_column])
    y = dataset[label_column]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y if stratify else None
    )

    train = pd.concat([X_train, y_train], axis=1)
    test = pd.concat([X_test, y_test], axis=1)

    save_train_test_data(train, test, path_to_split_data)


if __name__ == "__main__":
    path_to_processed_data = sys.argv[1]
    path_to_split_data = sys.argv[2]

    with open("params.yaml", "r") as file:
        params = yaml.safe_load(file)

    test_size = params["split_data"]["test_size"]
    stratify = params["split_data"]["stratify"]
    label_column = params["split_data"]["label_column"]
    split_data(
        path_to_processed_data, path_to_split_data, test_size, stratify, label_column
    )

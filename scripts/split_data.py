import sys
from typing import Tuple

import pandas as pd
from sklearn.model_selection import train_test_split

from src.utils import load_dataset, save_train_test_data, get_params


def split_data(
    path_to_preprocessed_data: str,
    path_to_split_data: str,
    test_size: float,
    stratify: bool,
    label_column: str,
    dataset_name: str,
) -> None:
    """
    Splits the dataset into train and test sets and saves them.

    Args:
        path_to_preprocessed_data (str): Path to the preprocessed dataset.
        path_to_split_data (str): Directory where split datasets will be saved.
        test_size (float): Proportion of the dataset to include in the test split.
        stratify (bool): Whether to stratify the split based on the label.
        label_column (str): Name of the label column.
        dataset_name (str): Name of the dataset being processed.
    """
    dataset = load_dataset(path_to_preprocessed_data)

    if label_column not in dataset.columns:
        raise ValueError(f"Label column '{label_column}' not found in dataset.")

    X = dataset.drop(columns=[label_column])
    y = dataset[label_column]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=1,
        stratify=y if stratify else None,
    )

    train = pd.concat([X_train, y_train], axis=1)
    test = pd.concat([X_test, y_test], axis=1)

    save_train_test_data(train, test, path_to_split_data, dataset_name)


if __name__ == "__main__":
    if len(sys.argv) != 4:
        raise ValueError(
            "Expected 3 arguments: <path_to_preprocessed_data> <path_to_split_data> <dataset_name>"
        )

    path_to_preprocessed_data = sys.argv[1]
    path_to_split_data = sys.argv[2]
    dataset_name = sys.argv[3]

    common_params, custom_params = get_params(dataset_name)

    test_size: float = common_params["split_data"].get("test_size", 0.2)
    stratify: bool = common_params["split_data"].get("stratify", True)
    label_column: str = custom_params["features"].get("label", "label")

    split_data(
        path_to_preprocessed_data,
        path_to_split_data,
        test_size,
        stratify,
        label_column,
        dataset_name,
    )

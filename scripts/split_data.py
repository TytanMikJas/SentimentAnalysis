import pandas as pd
import sys
import yaml
from sklearn.model_selection import train_test_split
from src.utils import load_dataset, save_train_test_data, get_params


def split_data(
    path_to_preprocessed_data,
    path_to_split_data,
    test_size,
    stratify,
    label_column,
    dataset_name,
):
    dataset = load_dataset(path_to_preprocessed_data)

    X = dataset.drop(columns=[label_column])
    y = dataset[label_column]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=1, stratify=y if stratify else None
    )

    train = pd.concat([X_train, y_train], axis=1)
    test = pd.concat([X_test, y_test], axis=1)

    save_train_test_data(train, test, path_to_split_data, dataset_name)


if __name__ == "__main__":
    path_to_preprocessed_data = sys.argv[1]
    path_to_split_data = sys.argv[2]
    dataset_name = sys.argv[3]

    common_params, custom_params = get_params(dataset_name)

    test_size = common_params["split_data"]["test_size"]
    stratify = common_params["split_data"]["stratify"]
    label_column = custom_params["features"]["label"]
    split_data(
        path_to_preprocessed_data,
        path_to_split_data,
        test_size,
        stratify,
        label_column,
        dataset_name,
    )

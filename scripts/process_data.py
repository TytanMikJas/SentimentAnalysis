import pandas as pd
import sys
from src.utils import (
    load_raw_data,
    save_dataset,
    get_params,
    SEPHORA_DATASET,
    RT_POLARITY_DATASET,
)
import yaml


def create_new_attributes(df, review_text_column, dataset_name):
    df["review_length"] = df[review_text_column].astype(str).apply(len)

    if dataset_name == SEPHORA_DATASET:
        df["contains_refund"] = (
            df[review_text_column]
            .astype(str)
            .apply(lambda x: 1 if "refund" in x.lower() else 0)
        )
    elif dataset_name == RT_POLARITY_DATASET:
        df["contains_bad"] = (
            df[review_text_column]
            .astype(str)
            .apply(lambda x: 1 if "bad" in x.lower() else 0)
        )

    df["exclamation_count"] = (
        df[review_text_column].fillna("").astype(str).apply(lambda x: x.count("!"))
    )

    df["unique_word_count"] = (
        df[review_text_column].astype(str).apply(lambda x: len(set(x.split())))
    )

    return df


def fix_column_types_to_numeric(df, cols_to_fix):
    for col in cols_to_fix:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int)
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int)

    return df


def process_data(
    path_to_raw_data,
    path_to_processed_data,
    dataset_name,
    common_params,
    custom_params,
):
    cols_to_fix = custom_params["process_data"].get("cols_to_fix", []) or []
    cols_to_delete = custom_params["process_data"].get("cols_to_delete", []) or []

    df = load_raw_data(path_to_raw_data, dataset_name)
    df = df.drop(columns=cols_to_delete, errors="ignore")
    df = fix_column_types_to_numeric(df, cols_to_fix)

    df = create_new_attributes(
        df, common_params["process_data"]["review_text_column"], dataset_name
    )

    save_dataset(df, path_to_processed_data)


if __name__ == "__main__":
    path_to_raw_data = sys.argv[1]
    path_to_processed_data = sys.argv[2]
    dataset_name = sys.argv[3]

    common_params, custom_params = get_params(dataset_name)

    process_data(
        path_to_raw_data,
        path_to_processed_data,
        dataset_name,
        common_params,
        custom_params,
    )

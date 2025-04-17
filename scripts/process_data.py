import sys
from typing import List, Dict

import pandas as pd

from src.utils import (
    load_raw_data,
    save_dataset,
    get_params,
    SEPHORA_DATASET,
    RT_POLARITY_DATASET,
)


def create_new_attributes(
    df: pd.DataFrame, review_text_column: str, dataset_name: str
) -> pd.DataFrame:
    """
    Generate new attributes based on review text.
    """
    df["review_length"] = df[review_text_column].astype(str).apply(len)

    if dataset_name == SEPHORA_DATASET:
        df["contains_refund"] = (
            df[review_text_column]
            .astype(str)
            .apply(lambda x: int("refund" in x.lower()))
        )
    elif dataset_name == RT_POLARITY_DATASET:
        df["contains_bad"] = (
            df[review_text_column].astype(str).apply(lambda x: int("bad" in x.lower()))
        )

    df["exclamation_count"] = (
        df[review_text_column].fillna("").astype(str).apply(lambda x: x.count("!"))
    )
    df["unique_word_count"] = (
        df[review_text_column].astype(str).apply(lambda x: len(set(x.split())))
    )

    return df


def fix_column_types_to_numeric(
    df: pd.DataFrame, cols_to_fix: List[str]
) -> pd.DataFrame:
    """
    Convert specified columns to numeric values, replacing errors with 0.
    """
    for col in cols_to_fix:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int)
    return df


def process_data(
    path_to_raw_data: str,
    path_to_processed_data: str,
    dataset_name: str,
    common_params: Dict,
    custom_params: Dict,
):
    """
    Process raw data by cleaning, type-fixing, and creating features.
    """
    cols_to_fix = custom_params.get("process_data", {}).get("cols_to_fix", [])
    cols_to_delete = custom_params.get("process_data", {}).get("cols_to_delete", [])

    df = load_raw_data(path_to_raw_data, dataset_name)
    df.drop(columns=cols_to_delete, inplace=True, errors="ignore")
    df = fix_column_types_to_numeric(df, cols_to_fix)

    review_text_column = common_params.get("process_data", {}).get("review_text_column")
    if review_text_column:
        df = create_new_attributes(df, review_text_column, dataset_name)

    save_dataset(df, path_to_processed_data)


if __name__ == "__main__":
    if len(sys.argv) != 4:
        print(
            "Usage: python process_data.py <raw_data_path> <processed_data_path> <dataset_name>"
        )
        sys.exit(1)

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

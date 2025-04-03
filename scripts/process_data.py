import pandas as pd
import sys
from src.utils import load_raw_data, save_dataset
import yaml


def create_new_attributes(df, review_text_column):
    df["review_length"] = df[review_text_column].astype(str).apply(len)

    df["contains_refund"] = (
        df[review_text_column]
        .astype(str)
        .apply(lambda x: 1 if "refund" in x.lower() else 0)
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
    cols_to_delete,
    review_text_column,
    cols_to_fix,
):
    reviews_df, product_info_df = load_raw_data(path_to_raw_data)

    df = pd.merge(reviews_df, product_info_df, on="product_id", suffixes=("", "_drop"))
    df = df.drop(columns=[col for col in df.columns if col.endswith("_drop")])

    df = df.drop(columns=cols_to_delete, errors="ignore")

    df = fix_column_types_to_numeric(df, cols_to_fix)

    df = create_new_attributes(df, review_text_column)

    save_dataset(df, path_to_processed_data)


if __name__ == "__main__":
    path_to_raw_data = sys.argv[1]
    path_to_processed_data = sys.argv[2]

    with open("params.yaml", "r") as file:
        params = yaml.safe_load(file)

    process_data(
        path_to_raw_data,
        path_to_processed_data,
        params["process_data"]["cols_to_delete"],
        params["process_data"]["review_text_column"],
        params["process_data"]["cols_to_fix"],
    )

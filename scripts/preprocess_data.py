import re
import sys
import yaml
import ast
import pandas as pd
import spacy.cli
import spacy

from typing import Any
from pandas import DataFrame
from spacy.language import Language

from src.utils import load_dataset, save_dataset, log_info, SEPHORA_DATASET


def remove_missing_values(df: DataFrame) -> DataFrame:
    """Remove rows with missing review text."""
    return df.dropna(subset=["review_text"])


def fill_missing_data(df: DataFrame) -> DataFrame:
    """Fill missing values in 'helpfulness' and 'review_title' columns."""
    df["helpfulness"] = df.apply(
        lambda x: x["total_pos_feedback_count"] / x["total_feedback_count"]
        if x["total_feedback_count"] != 0
        else 0.5,
        axis=1,
    )
    df["review_title"] = df["review_title"].fillna("NaN")
    return df


def transform_highlights(df: DataFrame) -> DataFrame:
    """Convert highlight strings to Python lists."""
    df["highlights"] = df["highlights"].apply(
        lambda x: ast.literal_eval(x) if pd.notnull(x) else []
    )
    return df


def clean_text_data(df: DataFrame, nlp: Language, dataset_name: str) -> DataFrame:
    """Clean text data using SpaCy NLP pipeline."""

    def clean_column(texts):
        cleaned = []
        for doc in nlp.pipe(texts, batch_size=64, disable=["parser", "ner"]):
            tokens = [
                token.lemma_
                for token in doc
                if not token.is_stop
                and not token.is_punct
                and not token.like_num
                and token.pos_ in ["ADJ", "VERB", "ADV"]
            ]
            cleaned.append(" ".join(tokens))
        return cleaned

    log_info("Processing review_text with SpaCy...")
    df["review_text"] = clean_column(df["review_text"].astype(str).tolist())

    if dataset_name == SEPHORA_DATASET:
        log_info("Processing review_title with SpaCy...")
        df["review_title"] = clean_column(df["review_title"].astype(str).tolist())

    return df


def preprocess_data(
    path_to_data: str,
    path_to_preprocessed_data: str,
    dataset_name: str,
    nlp: Language,
) -> None:
    """Main preprocessing function that applies transformations."""
    df = load_dataset(path_to_data)
    df = remove_missing_values(df)

    if dataset_name == SEPHORA_DATASET:
        df = transform_highlights(df)
        df = fill_missing_data(df)

    df = clean_text_data(df, nlp, dataset_name)
    save_dataset(df, path_to_preprocessed_data)


if __name__ == "__main__":
    if len(sys.argv) != 4:
        raise ValueError(
            "Usage: python preprocess.py <path_to_data> <path_to_preprocessed_data> <dataset_name>"
        )

    spacy.cli.download("en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")

    path_to_data = sys.argv[1]
    path_to_preprocessed_data = sys.argv[2]
    dataset_name = sys.argv[3]

    preprocess_data(
        path_to_data,
        path_to_preprocessed_data,
        dataset_name,
        nlp,
    )

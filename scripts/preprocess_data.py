import re
import sys
import yaml
import ast
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from src.utils import load_dataset, save_dataset


def remove_missing_values(df, cols_to_delete):
    df = df.dropna(subset=["review_text"])
    df = df.drop(columns=cols_to_delete, errors="ignore")
    return df


def fill_missing_data(df):
    df["helpfulness"] = df.apply(
        lambda x: x["total_pos_feedback_count"] / x["total_feedback_count"]
        if x["total_feedback_count"] != 0
        else 0.5,
        axis=1,
    )
    df["review_title"] = df["review_title"].fillna("NaN")
    return df


def transform_highlights(df):
    df["highlights"] = df["highlights"].apply(
        lambda x: ast.literal_eval(x) if pd.notnull(x) else []
    )
    return df


def clean_text(text, stop_words, lemmatizer):
    text = re.sub(r"[^a-zA-Z\s]", "", str(text))
    text = text.lower()
    tokens = text.split()
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    return " ".join(tokens)


def clean_text_data(df, stop_words, lemmatizer):
    df["review_text"] = df["review_text"].apply(
        lambda x: clean_text(x, stop_words, lemmatizer)
    )
    df["review_title"] = df["review_title"].apply(
        lambda x: clean_text(x, stop_words, lemmatizer)
    )
    return df


def preprocess_data(
    path_to_data, path_to_preprocessed_data, cols_to_delete, stop_words, lemmatizer
):
    df = load_dataset(path_to_data)
    df = remove_missing_values(df, cols_to_delete)
    df = transform_highlights(df)
    df = fill_missing_data(df)
    df = clean_text_data(df, stop_words, lemmatizer)
    save_dataset(df, path_to_preprocessed_data)


if __name__ == "__main__":
    nltk.download("stopwords")
    nltk.download("wordnet")
    stop_words = set(stopwords.words("english"))
    lemmatizer = WordNetLemmatizer()

    path_to_data = sys.argv[1]
    path_to_preprocessed_data = sys.argv[2]

    with open("params.yaml", "r") as file:
        params = yaml.safe_load(file)

    preprocess_data(
        path_to_data,
        path_to_preprocessed_data,
        params["preprocess_data"]["cols_to_delete"],
        stop_words,
        lemmatizer,
    )

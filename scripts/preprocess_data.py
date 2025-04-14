import re
import sys
import yaml
import ast
import pandas as pd
import spacy.cli
from src.utils import load_dataset, save_dataset, log_info, SEPHORA_DATASET
import spacy


def remove_missing_values(df):
    df = df.dropna(subset=["review_text"])
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


def clean_text_data(df, nlp, dataset_name):
    log_info("Processing review_text with SpaCy...")
    texts_review = df["review_text"].tolist()
    cleaned_review = []
    for doc in nlp.pipe(texts_review, batch_size=64, disable=["parser", "ner"]):
        tokens = [
            token.lemma_
            for token in doc
            if not token.is_stop
            and not token.is_punct
            and not token.like_num
            and token.pos_ in ["ADJ", "VERB", "ADV"]
        ]
        cleaned_review.append(" ".join(tokens))

    df["review_text"] = cleaned_review

    if dataset_name == SEPHORA_DATASET:
        log_info("Processing review_title with SpaCy...")
        texts_title = df["review_title"].tolist()
        cleaned_title = []
        for doc in nlp.pipe(texts_title, batch_size=64, disable=["parser", "ner"]):
            tokens = [
                token.lemma_
                for token in doc
                if not token.is_stop
                and not token.is_punct
                and not token.like_num
                and token.pos_ in ["ADJ", "VERB", "ADV"]
            ]
            cleaned_title.append(" ".join(tokens))
        df["review_title"] = cleaned_title

    return df


def preprocess_data(path_to_data, path_to_preprocessed_data, dataset_name, nlp):
    df = load_dataset(path_to_data)
    df = remove_missing_values(df)
    if dataset_name == SEPHORA_DATASET:
        df = df.sample(frac=0.1, random_state=42)
        df = transform_highlights(df)
        df = fill_missing_data(df)
    df = clean_text_data(df, nlp, dataset_name)
    save_dataset(df, path_to_preprocessed_data)


if __name__ == "__main__":
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

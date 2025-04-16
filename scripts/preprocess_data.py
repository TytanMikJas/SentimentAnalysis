import re
import sys
import yaml
import ast
import pandas as pd
import spacy.cli

# import nltk
# from nltk.corpus import stopwords
# from nltk.stem import WordNetLemmatizer
from src.utils import load_dataset, save_dataset, log_info
import spacy


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


# def clean_text_data(df, stop_words, lemmatizer):
#     df["review_text"] = df["review_text"].apply(
#         lambda x: clean_text(x, stop_words, lemmatizer)
#     )
#     df["review_title"] = df["review_title"].apply(
#         lambda x: clean_text(x, stop_words, lemmatizer)
#     )


def clean_text_data(df, nlp):
    texts_review = df["review_text"].tolist()
    texts_title = df["review_title"].tolist()

    log_info("Processing review_text with SpaCy...")
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

    log_info("Processing review_title with SpaCy...")
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

    df["review_text"] = cleaned_review
    df["review_title"] = cleaned_title

    return df


def preprocess_data(path_to_data, path_to_preprocessed_data, cols_to_delete, nlp):
    df = load_dataset(path_to_data)
    df = remove_missing_values(df, cols_to_delete)
    df = transform_highlights(df)
    df = fill_missing_data(df)
    df = clean_text_data(df, nlp)
    save_dataset(df, path_to_preprocessed_data)


if __name__ == "__main__":
    # nltk.download("stopwords")
    # nltk.download("wordnet")
    # stop_words = set(stopwords.words("english"))
    # lemmatizer = WordNetLemmatizer()
    spacy.cli.download("en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")

    path_to_data = sys.argv[1]
    path_to_preprocessed_data = sys.argv[2]

    with open("params.yaml", "r") as file:
        params = yaml.safe_load(file)

    preprocess_data(
        path_to_data,
        path_to_preprocessed_data,
        params["preprocess_data"]["cols_to_delete"],
        nlp,
    )

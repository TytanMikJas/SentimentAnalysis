import pandas as pd
import numpy as np
import spacy
import spacy.cli as spacy_cli

from typing import List, Optional, Tuple, Union
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, MultiLabelBinarizer
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import PCA, TruncatedSVD

from src.utils import SEPHORA_DATASET


class HighlightsBinarizer(BaseEstimator, TransformerMixin):
    """
    Binarizes top-k most frequent highlight terms in a multilabel column.
    """

    def __init__(self, top_k: int = 20):
        self.top_k = top_k
        self.mlb = MultiLabelBinarizer()
        self.top_items = set()

    def fit(self, X: pd.Series, y=None):
        X = X.squeeze()
        flat = [item for sublist in X for item in sublist]
        top_items = pd.Series(flat).value_counts().nlargest(self.top_k).index
        self.top_items = set(top_items)
        filtered = [[i for i in sub if i in self.top_items] for sub in X]
        self.mlb.fit(filtered)
        return self

    def transform(self, X: pd.Series):
        X = X.squeeze()
        filtered = [[i for i in sub if i in self.top_items] for sub in X]
        return self.mlb.transform(filtered)


class SpacyWord2VecVectorizer(BaseEstimator, TransformerMixin):
    """
    Vectorizes text using spaCy pre-trained word vectors.
    """

    def __init__(self, model_name: str = "en_core_web_md"):
        self.model_name = model_name
        self.nlp = None

    def _load_model(self):
        if self.nlp is None:
            try:
                self.nlp = spacy.load(self.model_name)
            except OSError:
                spacy_cli.download(self.model_name)
                self.nlp = spacy.load(self.model_name)

    def _get_doc_vector(self, doc) -> np.ndarray:
        vec = doc.vector
        return vec if vec.shape[0] > 0 else np.zeros(self.nlp.vocab.vectors_length)

    def fit(self, X: pd.Series, y=None):
        self._load_model()
        return self

    def transform(self, X: pd.Series):
        self._load_model()
        return np.array(
            [self._get_doc_vector(doc) for doc in self.nlp.pipe(X, batch_size=32)]
        )


def build_pipeline(
    custom_params: dict,
    dataset_name: str,
    use_data: str,
    classifier,
    vec_method: str = "bag-of-words",
) -> Pipeline:
    """
    Build a machine learning pipeline based on provided parameters.
    """
    selected_columns = custom_params["pipeline"]["selected"]
    cat_cols = [
        col
        for col in custom_params["features"]["categorical"]
        if col in selected_columns
    ]
    num_cols = [
        col for col in custom_params["features"]["numerical"] if col in selected_columns
    ]
    text_cols = [
        col for col in custom_params["features"]["text"] if col in selected_columns
    ]
    highlights_col = custom_params["features"].get("highlights")
    use_dim_red = custom_params["pipeline"].get("dim_reduction", False)

    transformers = []

    if use_data in ["all", "non-text"]:
        if cat_cols:
            transformers.append(
                ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols)
            )
        if num_cols:
            transformers.append(("num", StandardScaler(), num_cols))
        if highlights_col and highlights_col in selected_columns:
            transformers.append(("highlights", HighlightsBinarizer(), [highlights_col]))

    if use_data in ["all", "text"]:
        for text_col in text_cols:
            if vec_method == "bag-of-words":
                vec = CountVectorizer(max_features=200, min_df=2, max_df=0.90)
            elif vec_method == "tf-idf":
                vec = TfidfVectorizer(max_features=200, min_df=2, max_df=0.90)
            elif vec_method == "word2vec":
                vec = SpacyWord2VecVectorizer()
            else:
                raise ValueError(f"Unknown vectorization method: {vec_method}")
            transformers.append((f"text_{text_col}", vec, text_col))

    preprocessor = ColumnTransformer(
        transformers=transformers, remainder="drop", verbose_feature_names_out=False
    )

    steps = [("preprocessor", preprocessor)]

    if use_dim_red:
        dim_red_method = custom_params["pipeline"].get("dim_red_method", "pca")
        n_components = custom_params["pipeline"].get("n_components", 50)

        if dim_red_method == "pca":
            steps.append(("reduce_dim", PCA(n_components=n_components, random_state=1)))
        elif dim_red_method == "svd":
            steps.append(
                ("reduce_dim", TruncatedSVD(n_components=n_components, random_state=1))
            )
        else:
            raise ValueError(
                f"Unknown dimensionality reduction method: {dim_red_method}"
            )

    steps.append(("classifier", classifier))

    return Pipeline(steps=steps)

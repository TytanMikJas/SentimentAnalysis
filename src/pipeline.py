import pandas as pd
import spacy
import numpy as np
import spacy.cli
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, MultiLabelBinarizer
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.base import BaseEstimator, TransformerMixin
from src.utils import SEPHORA_DATASET


class HighlightsBinarizer(BaseEstimator, TransformerMixin):
    def __init__(self, top_k=20):
        self.top_k = top_k
        self.mlb = MultiLabelBinarizer()

    def fit(self, X, y=None):
        X = X.squeeze()
        flat = [item for sublist in X for item in sublist]
        top_items = pd.Series(flat).value_counts().nlargest(self.top_k).index
        self.top_items = set(top_items)
        filtered = [[i for i in sub if i in self.top_items] for sub in X]
        self.mlb.fit(filtered)
        return self

    def transform(self, X):
        X = X.squeeze()
        filtered = [[i for i in sub if i in self.top_items] for sub in X]
        return self.mlb.transform(filtered)


class ColumnSelector(BaseEstimator, TransformerMixin):
    def __init__(self, columns):
        self.columns = columns

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X[self.columns]


class SpacyWord2VecVectorizer(BaseEstimator, TransformerMixin):
    def __init__(self, model_name="en_core_web_md"):
        self.model_name = model_name
        self.nlp = None

    def _load_model(self):
        if self.nlp is None:
            try:
                self.nlp = spacy.load(self.model_name)
            except OSError:
                spacy.cli.download(self.model_name)
                self.nlp = spacy.load(self.model_name)

    def _get_doc_vector(self, doc):
        vec = doc.vector
        return vec if vec.shape[0] > 0 else np.zeros(self.nlp.vocab.vectors_length)

    def fit(self, X, y=None):
        self._load_model()
        return self

    def transform(self, X):
        self._load_model()
        return np.array(
            [self._get_doc_vector(doc) for doc in self.nlp.pipe(X, batch_size=32)]
        )


def build_pipeline(
    custom_params, dataset_name, use_data, classifier, vec_method="bag-of-words"
):
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
    highlights_col = custom_params["features"]["highlights"]
    use_dim_red = custom_params["pipeline"]["dim_reduction"]

    transformers = []

    if use_data in ["all", "non-text"]:
        transformers.append(("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols))
        transformers.append(("num", StandardScaler(), num_cols))
        if highlights_col != None:
            transformers.append(("highlights", HighlightsBinarizer(), [highlights_col]))

    if use_data in ["all", "text"]:
        for text_col in text_cols:
            if vec_method == "bag-of-words":
                transformers.append(
                    (
                        f"text_{text_col}",
                        CountVectorizer(max_features=200, min_df=2, max_df=0.90),
                        text_col,
                    )
                )
            elif vec_method == "tf-idf":
                transformers.append(
                    (
                        f"text_{text_col}",
                        TfidfVectorizer(max_features=200, min_df=2, max_df=0.90),
                        text_col,
                    )
                )
            elif vec_method == "word2vec":
                transformers.append(
                    (
                        f"text_{text_col}",
                        SpacyWord2VecVectorizer(),
                        text_col,
                    )
                )

    preprocessor = ColumnTransformer(
        transformers, remainder="drop", verbose_feature_names_out=False
    )

    steps = [("preprocessor", preprocessor)]

    if use_dim_red:
        dim_red_method = custom_params["pipeline"]["dim_red_method"]
        n_components = custom_params["pipeline"]["n_components"]

        if dim_red_method == "pca":
            steps.append(("reduce_dim", PCA(n_components=n_components, random_state=1)))
        else:
            steps.append(("reduce_dim", TruncatedSVD(n_components=n_components, random_state=1)))

    steps.append(("classifier", classifier))

    pipeline = Pipeline(steps=steps)

    return pipeline

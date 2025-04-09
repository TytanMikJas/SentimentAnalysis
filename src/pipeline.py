import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, MultiLabelBinarizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.decomposition import PCA, TruncatedSVD


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


from sklearn.base import BaseEstimator, TransformerMixin


class ColumnSelector(BaseEstimator, TransformerMixin):
    def __init__(self, columns):
        self.columns = columns

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X[self.columns]


def build_pipeline(params, use_data, classifier):
    selected_columns = params["features"]["selected"]
    cat_cols = [
        col for col in params["features"]["categorical"] if col in selected_columns
    ]
    num_cols = [
        col for col in params["features"]["numerical"] if col in selected_columns
    ]
    text_cols = [col for col in params["features"]["text"] if col in selected_columns]
    highlights_col = params["features"]["highlights"]
    use_dim_red = params["features"]["dim_reduction"]

    transformers = []

    if use_data in ["all", "non-text"]:
        transformers.append(("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols))
        transformers.append(("num", StandardScaler(), num_cols))
        transformers.append(("highlights", HighlightsBinarizer(), [highlights_col]))

    if use_data in ["all", "text"]:
        for text_col in text_cols:
            transformers.append(
                (
                    f"text_{text_col}",
                    CountVectorizer(max_features=200, min_df=2, max_df=0.95),
                    text_col,
                )
            )

    preprocessor = ColumnTransformer(
        transformers, remainder="drop", verbose_feature_names_out=False
    )

    steps = [("preprocessor", preprocessor)]

    if use_dim_red:
        dim_red_method = params["features"]["dim_red_method"]
        n_components = params["features"]["n_components"]

        if dim_red_method == "pca":
            steps.append(("reduce_dim", PCA(n_components=n_components)))
        else:
            steps.append(("reduce_dim", TruncatedSVD(n_components=n_components)))

    steps.append(("classifier", classifier))

    pipeline = Pipeline(steps=steps)

    return pipeline

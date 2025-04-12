import sys
import yaml
import wandb
from sklearn.dummy import DummyClassifier
from sklearn.metrics import f1_score
from src.utils import (
    load_train_test_data,
    save_f1_score,
    plot_confusion_matrix,
    log_info,
)
from src.pipeline import build_pipeline
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
import numpy as np
import os


def run_test_classifiers(path_to_split_data, metrics_file, params):
    classifiers = {
        "dummy": DummyClassifier(),
        "svm": SVC(),
        "random_forest": RandomForestClassifier(max_depth=100, n_jobs=-1),
    }

    settings = {
        "text": "Tylko dane tekstowe",
        "non-text": "Tylko dane nietekstowe",
        "all": "Wszystkie dane",
    }

    training_data, _ = load_train_test_data(path_to_split_data)
    training_data = training_data[params["features"]["selected"]]
    training_data = training_data[0:10_000]
    X = training_data.drop(columns=["LABEL-rating"])
    y = training_data["LABEL-rating"]

    for use_data in settings:
        for clf_name, clf in classifiers.items():
            log_info(f"TESTTING {use_data.upper()} FOR {clf_name.upper()} CLASSIFIER")
            wandb.init(
                project="pdiow-lab-5-exp", name=f"{use_data}_{clf_name}", reinit=True
            )

            skf = StratifiedKFold(n_splits=5, shuffle=True)
            f1_train_scores = []
            f1_test_scores = []
            y_train_true_all = []
            y_train_pred_all = []
            y_test_true_all = []
            y_test_pred_all = []

            for train_idx, test_idx in skf.split(X, y):
                log_info(
                    f"{use_data} for {clf_name} iteration {len(f1_train_scores) + 1}/5"
                )
                X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
                y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

                pipeline = build_pipeline(params, use_data=use_data, classifier=clf)
                pipeline.fit(X_train, y_train)
                y_hat_train = pipeline.predict(X_train)
                y_hat_test = pipeline.predict(X_test)

                f1_train = f1_score(y_train, y_hat_train, average="weighted")
                f1_test = f1_score(y_test, y_hat_test, average="weighted")

                f1_train_scores.append(f1_train)
                f1_test_scores.append(f1_test)

                y_train_true_all.extend(y_train)
                y_train_pred_all.extend(y_hat_train)
                y_test_true_all.extend(y_test)
                y_test_pred_all.extend(y_hat_test)

            wandb.log(
                {
                    "avg F1 Score Train": np.mean(f1_train),
                    "avg F1 Score Test": np.mean(f1_test),
                }
            )

            save_f1_score(
                np.mean(f1_test_scores), f"{clf_name} on {use_data}", metrics_file
            )

            print(
                f"{clf_name.upper()} ({use_data}): Avg F1 = {np.mean(f1_test_scores):.6f}"
            )

            plot_confusion_matrix(
                y_train_true_all, y_train_pred_all, y_test_true_all, y_test_pred_all
            )

            wandb.finish()


if __name__ == "__main__":
    path_to_split_data = sys.argv[1]
    metrics_file = sys.argv[2]
    os.environ["WANDB_SILENT"] = "true"

    with open("params.yaml", "r") as file:
        params = yaml.safe_load(file)

    run_test_classifiers(path_to_split_data, metrics_file, params)

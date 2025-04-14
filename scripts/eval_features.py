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
    get_params,
)
from src.pipeline import build_pipeline
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
import numpy as np
import os
import json


def run_test_classifiers(path_to_split_data, metrics_file, custom_params, dataset_name):
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

    training_data, _ = load_train_test_data(path_to_split_data, dataset_name)
    training_data = training_data[custom_params["pipeline"]["selected"]]
    training_data = training_data[0:5_000]
    X = training_data.drop(columns=custom_params["features"]["label"])
    y = training_data[custom_params["features"]["label"]]

    best_f1_score = -1
    best_feature_set = None

    for use_data in settings:
        for clf_name, clf in classifiers.items():
            log_info(
                f"TESTTING {use_data.upper()} FOR {clf_name.upper()} CLASSIFIER ON {dataset_name.upper()}"
            )
            wandb.init(
                project="pdiow-features-test",
                name=f"{use_data}_{clf_name}_{dataset_name}",
                reinit=True,
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
                    f"{use_data} for {clf_name} iteration {dataset_name} {len(f1_train_scores) + 1}/5"
                )
                X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
                y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

                pipeline = build_pipeline(
                    custom_params, dataset_name, use_data=use_data, classifier=clf
                )
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

            avg_f1 = np.mean(f1_test_scores)

            if avg_f1 > best_f1_score:
                best_f1_score = avg_f1
                best_feature_set = use_data

            wandb.log(
                {
                    "avg F1 Score Train": np.mean(f1_train_scores),
                    "avg F1 Score Test": np.mean(f1_test_scores),
                }
            )

            save_f1_score(
                np.mean(f1_test_scores),
                f"{clf_name} on {use_data} for {dataset_name}",
                metrics_file,
            )

            print(
                f"{clf_name.upper()} ({use_data}): Avg F1 = {np.mean(f1_test_scores):.6f}"
            )

            plot_confusion_matrix(
                y_train_true_all, y_train_pred_all, y_test_true_all, y_test_pred_all
            )

            wandb.finish()

    with open(f"data/models/{dataset_name}/best_features.json", "w") as f:
        json.dump({"best_features": best_feature_set}, f)


if __name__ == "__main__":
    path_to_split_data = sys.argv[1]
    metrics_file = sys.argv[2]
    dataset_name = sys.argv[3]
    os.environ["WANDB_SILENT"] = "true"

    common_params, custom_params = get_params(dataset_name)

    run_test_classifiers(path_to_split_data, metrics_file, custom_params, dataset_name)

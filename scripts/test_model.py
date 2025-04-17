import os
import sys
import json
import joblib
import wandb
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.dummy import DummyClassifier

from src.utils import (
    load_train_test_data,
    plot_confusion_matrix,
    log_info,
    get_params,
    SEPHORA_DATASET,
)
from src.pipeline import build_pipeline


WANDB_PROJECT = "pdiow-model-test"


def load_best_model_settings(dataset_name):
    def load_json(path):
        with open(path, "r") as f:
            return json.load(f)

    base_path = f"data/models/{dataset_name}"
    return (
        load_json(f"{base_path}/best_params.json"),
        load_json(f"{base_path}/best_features.json"),
        load_json(f"{base_path}/best_vec_method.json"),
    )


def prepare_data(path_to_split_data, dataset_name, selected_columns, label_col):
    train_data, test_data = load_train_test_data(path_to_split_data, dataset_name)

    train_data = train_data[selected_columns]
    test_data = test_data[selected_columns]

    X_train = train_data.drop(columns=[label_col])
    y_train = train_data[label_col]
    X_test = test_data.drop(columns=[label_col])
    y_test = test_data[label_col]

    return X_train, y_train, X_test, y_test


def run_test_classifiers(path_to_split_data, metrics_file, custom_params, dataset_name):
    best_params, best_features, best_vec = load_best_model_settings(dataset_name)
    best_vec_method = "bag-of-words"

    classifiers = {
        "dummy": DummyClassifier(random_state=1),
        "svm": SVC(
            C=best_params["SVM"]["classifier__C"],
            kernel=best_params["SVM"]["classifier__kernel"],
            degree=best_params["SVM"]["classifier__degree"],
            gamma=best_params["SVM"]["classifier__gamma"],
            random_state=1,
        ),
        "random_forest": RandomForestClassifier(
            max_depth=best_params["RandomForest"]["classifier__max_depth"],
            n_estimators=best_params["RandomForest"]["classifier__n_estimators"],
            criterion=best_params["RandomForest"]["classifier__criterion"],
            max_features=best_params["RandomForest"]["classifier__max_features"],
            min_samples_leaf=best_params["RandomForest"][
                "classifier__min_samples_leaf"
            ],
            n_jobs=-1,
            random_state=1,
        ),
    }

    label_col = custom_params["features"]["label"]
    selected_columns = custom_params["pipeline"]["selected"]

    X_train, y_train, X_test, y_test = prepare_data(
        path_to_split_data, dataset_name, selected_columns, label_col
    )

    for clf_name, clf in classifiers.items():
        log_info(
            f"Testing '{best_features['best_features']}' with {clf_name.upper()} on {best_vec_method}"
        )

        wandb.init(
            project=WANDB_PROJECT,
            name=f"{clf_name}_{dataset_name}",
            reinit=True,
            config={
                "classifier": clf_name,
                "use_data": best_features["best_features"],
                "params": clf.get_params(),
            },
        )

        pipeline = build_pipeline(
            custom_params,
            dataset_name,
            best_features["best_features"],
            clf,
            best_vec_method,
        )

        log_info("🔧 Fitting pipeline...")
        pipeline.fit(X_train, y_train)

        log_info("Saving trained model")
        joblib.dump(
            pipeline, f"data/models/{dataset_name}/{clf_name}_final_pipeline.pkl"
        )

        log_info("Predicting on test set")
        y_pred = pipeline.predict(X_test)

        log_info("Plotting confusion matrix")
        plot_confusion_matrix(y_test, y_pred)

        metrics = {
            "test_accuracy": accuracy_score(y_test, y_pred),
            "test_precision": precision_score(
                y_test, y_pred, average="weighted", zero_division=0
            ),
            "test_recall": recall_score(
                y_test, y_pred, average="weighted", zero_division=0
            ),
            "test_f1_score": f1_score(
                y_test, y_pred, average="weighted", zero_division=0
            ),
        }

        wandb.log(metrics)

        experiment_name = f"{clf_name}_{dataset_name}"
        metrics_key = f"{experiment_name}_metrics"

        if os.path.exists(metrics_file):
            with open(metrics_file, "r") as f:
                existing_metrics = json.load(f)
        else:
            existing_metrics = {}

        existing_metrics[metrics_key] = metrics

        with open(metrics_file, "w") as f:
            json.dump(existing_metrics, f, indent=4)

        wandb.finish()


if __name__ == "__main__":
    os.environ["WANDB_SILENT"] = "true"

    path_to_split_data = sys.argv[1]
    metrics_file = sys.argv[2]
    dataset_name = sys.argv[3]

    _, custom_params = get_params(dataset_name)

    run_test_classifiers(path_to_split_data, metrics_file, custom_params, dataset_name)

import sys
import yaml
import wandb
from sklearn.dummy import DummyClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)
from src.utils import (
    load_train_test_data,
    plot_confusion_matrix,
    log_info,
)
from src.pipeline import build_pipeline
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
import os
import json


def run_test_classifiers(path_to_split_data, metrics_file, params):
    classifiers = {
        "dummy": DummyClassifier(),
        "svm": SVC(max_iter=2000),
        "random_forest": RandomForestClassifier(max_depth=55, n_jobs=-1),
    }

    settings = {
        "all": "Wszystkie dane",
    }

    training_data, test_data = load_train_test_data(path_to_split_data)
    training_data = training_data[params["features"]["selected"]]
    test_data = test_data[params["features"]["selected"]]
    X_train = training_data.drop(columns=["LABEL-rating"])
    y_train = training_data["LABEL-rating"]
    X_test = test_data.drop(columns=["LABEL-rating"])
    y_test = test_data["LABEL-rating"]

    for use_data in settings:
        for clf_name, clf in classifiers.items():
            log_info(f"TESTING {use_data} FOR {clf_name} CLASSIFIER")
            wandb.init(
                project="pdiow-lab-5-test",
                name=f"{use_data}_{clf_name}",
                reinit=True,
                config={
                    "classifier": clf_name,
                    "use_data": use_data,
                    "params": clf.get_params(),
                },
            )
            log_info("Building pipeline")
            pipeline = build_pipeline(params, use_data, clf)
            log_info("Training model")
            pipeline.fit(X_train, y_train)
            log_info("Predict test")
            y_pred = pipeline.predict(X_test)
            log_info("Saving data")

            plot_confusion_matrix(y_test, y_pred)

            test_acc = accuracy_score(y_test, y_pred)
            test_prec = precision_score(
                y_test, y_pred, average="weighted", zero_division=0
            )
            test_rec = recall_score(y_test, y_pred, average="weighted", zero_division=0)
            test_f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)

            wandb.log(
                {
                    "test_accuracy": test_acc,
                    "test_precision": test_prec,
                    "test_recall": test_rec,
                    "test_f1_score": test_f1,
                }
            )

            experiment_name = f"{use_data}_{clf_name}"
            new_metrics = {
                f"{experiment_name}_metrics": {
                    "test accuracy": test_acc,
                    "test precision": test_prec,
                    "test recall": test_rec,
                    "test f1_score": test_f1,
                }
            }

            if os.path.exists(metrics_file):
                with open(metrics_file, "r") as f:
                    existing_data = json.load(f)
            else:
                existing_data = {}

            existing_data.update(new_metrics)

            with open(metrics_file, "w") as f:
                json.dump(existing_data, f, indent=4)

            wandb.finish()


if __name__ == "__main__":
    path_to_split_data = sys.argv[1]
    metrics_file = sys.argv[2]
    os.environ["WANDB_SILENT"] = "true"

    with open("params.yaml", "r") as file:
        params = yaml.safe_load(file)

    run_test_classifiers(path_to_split_data, metrics_file, params)

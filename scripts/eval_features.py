import sys
import wandb
import os
import json
from sklearn.dummy import DummyClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from src.utils import (
    load_train_test_data,
    save_f1_score,
    log_info,
    get_params,
)
from src.pipeline import build_pipeline
from src.evaluation_utils import evaluate_classifier


def run_test_classifiers(path_to_split_data, metrics_file, custom_params, dataset_name):
    classifiers = {
        "dummy": DummyClassifier(random_state=1),
        "svm": SVC(random_state=1),
        "random_forest": RandomForestClassifier(
            max_depth=100, n_jobs=-1, random_state=1
        ),
    }

    settings = {
        "text": "Tylko dane tekstowe",
        "non-text": "Tylko dane nietekstowe",
        "all": "Wszystkie dane",
    }

    training_data, _ = load_train_test_data(path_to_split_data, dataset_name)
    training_data = training_data[custom_params["pipeline"]["selected"]]
    X = training_data.drop(columns=[custom_params["features"]["label"]])
    y = training_data[custom_params["features"]["label"]]

    best_f1_score = -1
    best_feature_set = None

    for use_data in settings:
        for clf_name, clf in classifiers.items():
            log_info(f"Testing {use_data} features with {clf_name} on {dataset_name}")

            avg_f1 = evaluate_classifier(
                clf=clf,
                X=X,
                y=y,
                use_data=use_data,
                clf_name=clf_name,
                dataset_name=dataset_name,
                label=custom_params["features"]["label"],
                custom_params=custom_params,
                build_pipeline_fn=build_pipeline,
                wandb_project="pdiow-features-test",
                wandb_run_name_prefix=use_data,
            )

            if avg_f1 > best_f1_score:
                best_f1_score = avg_f1
                best_feature_set = use_data

            save_f1_score(
                avg_f1, f"{clf_name} on {use_data} for {dataset_name}", metrics_file
            )

    with open(f"data/models/{dataset_name}/best_features.json", "w") as f:
        json.dump({"best_features": best_feature_set}, f)


if __name__ == "__main__":
    os.environ["WANDB_SILENT"] = "true"
    path_to_split_data = sys.argv[1]
    metrics_file = sys.argv[2]
    dataset_name = sys.argv[3]

    _, custom_params = get_params(dataset_name)

    run_test_classifiers(path_to_split_data, metrics_file, custom_params, dataset_name)

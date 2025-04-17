import sys
import wandb
import os
import json
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
    clf_name = "random_forest"
    clf = RandomForestClassifier(max_depth=100, n_jobs=-1, random_state=1)

    vectorizers = {
        "word2vec": "word2vec",
        "tfidf": "tf-idf",
        "bow": "bag-of-words",
    }

    use_data = "text"

    training_data, _ = load_train_test_data(path_to_split_data, dataset_name)
    training_data = training_data[custom_params["pipeline"]["selected"]]
    X = training_data.drop(columns=[custom_params["features"]["label"]])
    y = training_data[custom_params["features"]["label"]]

    best_f1_score = -1
    best_vect = None

    for vect_name, vect in vectorizers.items():
        log_info(f"Testing {vect_name} vectorizer with {clf_name} on {dataset_name}")

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
            wandb_project="pdiow-lab-vect-test",
            wandb_run_name_prefix=vect_name,
        )

        if avg_f1 > best_f1_score:
            best_f1_score = avg_f1
            best_vect = vect

        save_f1_score(
            avg_f1, f"{clf_name} on {vect_name} for {dataset_name}", metrics_file
        )

    with open(f"data/models/{dataset_name}/best_vec_method.json", "w") as f:
        json.dump({"best_vec_method": best_vect}, f)


if __name__ == "__main__":
    os.environ["WANDB_SILENT"] = "true"
    path_to_split_data = sys.argv[1]
    metrics_file = sys.argv[2]
    dataset_name = sys.argv[3]

    common_params, custom_params = get_params(dataset_name)
    run_test_classifiers(path_to_split_data, metrics_file, custom_params, dataset_name)

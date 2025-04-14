import sys
import yaml
import wandb
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
    get_params,
    SEPHORA_DATASET,
)
from src.pipeline import build_pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.dummy import DummyClassifier
from sklearn.svm import SVC
import os
import json
import joblib


def run_test_classifiers(path_to_split_data, metrics_file, custom_params, dataset_name):
    with open(f"data/models/{dataset_name}/best_params.json", "r") as f:
        best_params = json.load(f)

    with open(f"data/models/{dataset_name}/best_features.json", "r") as f:
        best_features = json.load(f)

    with open(f"data/models/{dataset_name}/best_vec_method.json", "r") as f:
        best_vec_method = json.load(f)

    rf_clf = RandomForestClassifier(
        max_depth=best_params["RandomForest"]["classifier__max_depth"],
        n_estimators=best_params["RandomForest"]["classifier__n_estimators"],
        criterion=best_params["RandomForest"]["classifier__criterion"],
        max_features=best_params["RandomForest"]["classifier__max_features"],
        min_samples_leaf=best_params["RandomForest"]["classifier__min_samples_leaf"],
        n_jobs=-1,
        random_state=1
    )

    # przez bardzo długi czas pracowania word2vec zdefaultowano na bag-of-words
    best_vec_method =  "bag-of-words"
    svm_clf = SVC(
        C=best_params["SVM"]["classifier__C"],
        kernel=best_params["SVM"]["classifier__kernel"],
        degree=best_params["SVM"]["classifier__degree"],
        gamma=best_params["SVM"]["classifier__gamma"],
        random_state=1
    )

    classifiers = {"dummy": DummyClassifier(random_state=1), "svm": svm_clf, "random_forest": rf_clf}

    training_data, test_data = load_train_test_data(path_to_split_data, dataset_name)
    training_data = training_data[custom_params["pipeline"]["selected"]]
    if dataset_name == SEPHORA_DATASET:
        training_data = training_data[:50_000]
    test_data = test_data[custom_params["pipeline"]["selected"]]
    if dataset_name == SEPHORA_DATASET:
        test_data = test_data[:10_000]
    X_train = training_data.drop(columns=[custom_params["features"]["label"]])
    y_train = training_data[custom_params["features"]["label"]]
    X_test = test_data.drop(columns=[custom_params["features"]["label"]])
    y_test = test_data[custom_params["features"]["label"]]
    for clf_name, clf in classifiers.items():
        log_info(
            f"TESTING {best_features['best_features']} FOR {clf_name} CLASSIFIER ON {best_vec_method} VECTORIZATION"
        )
        wandb.init(
            project="pdiow-model-test",
            name=f"{clf_name}_{dataset_name}",
            reinit=True,
            config={
                "classifier": f"{clf_name}",
                "use_data": best_features["best_features"],
                "params": clf.get_params(),
            },
        )
        log_info("Building pipeline")
        pipeline = build_pipeline(
            custom_params,
            dataset_name,
            best_features["best_features"],
            clf,
            best_vec_method,
        )
        log_info("Training model")
        pipeline.fit(X_train, y_train)
        log_info("Saving model")
        joblib.dump(
            pipeline, f"data/models/{dataset_name}/{clf_name}_final_pipeline.pkl"
        )
        log_info("Predict test")
        y_pred = pipeline.predict(X_test)
        log_info("Saving data")

        plot_confusion_matrix(y_test, y_pred)

        test_acc = accuracy_score(y_test, y_pred)
        test_prec = precision_score(y_test, y_pred, average="weighted", zero_division=0)
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

        experiment_name = f"{clf_name}_{dataset_name}"
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
    dataset_name = sys.argv[3]
    os.environ["WANDB_SILENT"] = "true"

    common_params, custom_params = get_params(dataset_name)

    run_test_classifiers(path_to_split_data, metrics_file, custom_params, dataset_name)

import os
import sys
import json
from pathlib import Path

METRICS_FILE_SUFFIX = "_metrics.json"
PARAMS_FILE = "best_params.json"
METRICS_TEST_FILE = "test_metrics.json"
MODELS = ["Dummy", "SVM", "Random Forest"]


def extract_params(params_path, dataset_key):
    params = {"Dummy": "", "SVM": "", "Random Forest": ""}
    full_path = Path(params_path) / dataset_key / PARAMS_FILE

    if not full_path.exists():
        return params

    with open(full_path) as f:
        model_params = json.load(f)

    if "RandomForest" in model_params:
        rf = model_params["RandomForest"]
        params["Random Forest"] = f'n_estimators={rf.get("classifier__n_estimators", "")}'
    if "SVM" in model_params:
        svm = model_params["SVM"]
        params["SVM"] = f'C={svm.get("classifier__C", "")}, kernel={svm.get("classifier__kernel", "")}'

    return params


def extract_metrics(metrics_path, dataset_key):
    metrics_file = os.path.join(metrics_path, dataset_key, METRICS_TEST_FILE)

    with open(metrics_file) as f:
        data = json.load(f)

    result = []
    for model in MODELS:
        key = f"{model.lower().replace(' ', '_')}_{dataset_key}_metrics"
        if key in data:
            metrics = data[key]
            result.append({
                "model": model,
                "accuracy": metrics.get("test accuracy", ""),
                "precision": metrics.get("test precision", ""),
                "recall": metrics.get("test recall", ""),
                "f1": metrics.get("test f1_score", "")
            })
    return result


def generate_markdown(datasets, all_metrics, all_params, output_path):
    lines = [
        "| Dataset     | Model           | Params          | Accuracy | Precision | Recall | F1 Score |",
        "|-------------|-----------------|-----------------|----------|-----------|--------|----------|"
    ]

    for dataset in datasets:
        for row in all_metrics[dataset]:
            lines.append(
                f"| {dataset} | {row['model']} | {all_params[dataset].get(row['model'], '')} | "
                f"{row['accuracy']:.4f} | {row['precision']:.4f} | {row['recall']:.4f} | {row['f1']:.4f} |"
            )

    with open(output_path, "w") as f:
        f.write("\n".join(lines))


def main(metrics_dir, models_dir, output_file):
    datasets = [d.name for d in Path(metrics_dir).iterdir() if d.is_dir()]

    all_metrics = {}
    all_params = {}

    for dataset in datasets:
        all_metrics[dataset] = extract_metrics(metrics_dir, dataset)
        all_params[dataset] = extract_params(models_dir, dataset)

    generate_markdown(datasets, all_metrics, all_params, output_file)


if __name__ == "__main__":
    metrics_dir = sys.argv[1]
    models_dir = sys.argv[2]
    output_file = sys.argv[3]

    main(metrics_dir, models_dir, output_file)

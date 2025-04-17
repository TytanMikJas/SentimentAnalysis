import numpy as np
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold
from src.utils import plot_confusion_matrix, save_f1_score, log_info
import wandb


def evaluate_classifier(
    clf,
    X,
    y,
    use_data,
    clf_name,
    dataset_name,
    label,
    custom_params,
    build_pipeline_fn,
    wandb_project,
    wandb_run_name_prefix="",
    n_splits=5,
):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=1)
    f1_train_scores = []
    f1_test_scores = []
    y_train_true_all = []
    y_train_pred_all = []
    y_test_true_all = []
    y_test_pred_all = []

    for fold, (train_idx, test_idx) in enumerate(skf.split(X, y)):
        log_info(
            f"{use_data} for {clf_name} iteration {dataset_name} {fold + 1}/{n_splits}"
        )
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        pipeline = build_pipeline_fn(
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

    avg_f1_train = np.mean(f1_train_scores)
    avg_f1_test = np.mean(f1_test_scores)

    plot_confusion_matrix(
        y_train_true_all, y_train_pred_all, y_test_true_all, y_test_pred_all
    )

    wandb.init(
        project=wandb_project,
        name=f"{wandb_run_name_prefix}_{clf_name}_{dataset_name}",
        reinit=True,
    )
    wandb.log(
        {
            "avg F1 Score Train": avg_f1_train,
            "avg F1 Score Test": avg_f1_test,
        }
    )
    wandb.finish()

    return avg_f1_test

import sys
import yaml
from sklearn.dummy import DummyClassifier
from sklearn.metrics import f1_score
from src.utils import load_train_test_data, save_f1_score


def eval_model(path_to_split_data, strategy, metrics_file):
    training_data, test_data = load_train_test_data(path_to_split_data)

    X_train = training_data.drop(columns=["LABEL-rating"])
    y_train = training_data["LABEL-rating"]
    X_test = test_data.drop(columns=["LABEL-rating"])
    y_test = test_data["LABEL-rating"]

    dummy_clf = DummyClassifier(strategy=strategy)
    dummy_clf.fit(X_train, y_train)

    y_hat = dummy_clf.predict(X_test)

    f1_scores = f1_score(y_test, y_hat, average="weighted")

    save_f1_score(f1_scores, metrics_file)


if __name__ == "__main__":
    path_to_split_data = sys.argv[1]

    with open(sys.argv[2], "r") as file:
        params = yaml.safe_load(file)

    strategy = params["model"]["strategy"]
    metrics_file = sys.argv[3]
    eval_model(path_to_split_data, strategy, metrics_file)

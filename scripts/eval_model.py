import sys
import yaml
import wandb
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.dummy import DummyClassifier
from sklearn.metrics import f1_score, confusion_matrix
from src.utils import load_train_test_data, save_f1_score


def plot_confusion_matrix(y_train, y_hat_train, y_test, y_hat_test):
    cm_train = confusion_matrix(y_train, y_hat_train)
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm_train, annot=True, fmt="d", cmap="Blues")
    plt.title("Confusion Matrix - Train")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    wandb.log({"Confusion Matrix Train": wandb.Image(plt)})
    plt.close()

    cm_test = confusion_matrix(y_test, y_hat_test)
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm_test, annot=True, fmt="d", cmap="Reds")
    plt.title("Confusion Matrix - Test")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    wandb.log({"Confusion Matrix Test": wandb.Image(plt)})
    plt.close()


# Zaraportuj metryki klasyfikacji stosując walidację krzyżową (cross-validation). Nie korzystamy ze zbioru testowego.
# Przeprowadź następujące eksperymenty:
# - użyj tylko danych tekstowych
# - użyj pozostałych danych oprócz tekstowych
# - użyj wszystkich danych


# skategoryzować brand_name, product_name, primary_category, secondary_category, limited_edition, new, online_only, out_of_stock, sephora_exclusive, is_recommended
# highlights MultiLabelBinarizer - ale top 20 najczęściej występujących


# Wykonaj kilka iteracji i eksperymentów, sprawdź, jakie kombinacje metod/cech pozwalają poprawić rezultaty, a jakie nie.
# scale_numeric_variables
# loves_count, rating, reviews, price_usd, helpfulness, total_feedback_count, total_pos_feedback_count, total_neg_feedback_count, child_count, exclamation_count, unique_word_count, review_length
# bag of words
# wstrzykujemy klasyfikator
def eval_classifier(model): ...


def run_classifiers(path_to_split_data, strategy, metrics_file):
    wandb.init(project="pdiow-lab-4", name=f"dummy_clf_{strategy}")

    training_data, test_data = load_train_test_data(path_to_split_data)

    X_train = training_data.drop(columns=["LABEL-rating"])
    y_train = training_data["LABEL-rating"]
    X_test = test_data.drop(columns=["LABEL-rating"])
    y_test = test_data["LABEL-rating"]

    dummy_clf = DummyClassifier(strategy=strategy)
    dummy_clf.fit(X_train, y_train)

    y_hat_train = dummy_clf.predict(X_train)
    y_hat_test = dummy_clf.predict(X_test)

    f1_train = f1_score(y_train, y_hat_train, average="weighted")
    f1_test = f1_score(y_test, y_hat_test, average="weighted")

    save_f1_score(f1_test, metrics_file)

    wandb.log({"F1 Score Train": f1_train, "F1 Score Test": f1_test})

    plot_confusion_matrix(y_train, y_hat_train, y_test, y_hat_test)

    wandb.finish()


if __name__ == "__main__":
    path_to_split_data = sys.argv[1]

    with open("params.yaml", "r") as file:
        params = yaml.safe_load(file)

    strategy = params["model"]["strategy"]
    metrics_file = sys.argv[3]
    run_classifiers(path_to_split_data, strategy, metrics_file)

import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import KFold
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    roc_curve,
    auc,
    precision_recall_curve,
    average_precision_score,
    classification_report,
    precision_recall_fscore_support,
    accuracy_score,
    matthews_corrcoef,
    confusion_matrix
)
from numpy import interp


def load_features(file_path):
    """
    Load feature matrix from a CSV file.
    """
    try:
        return np.loadtxt(file_path, delimiter=",")
    except FileNotFoundError:
        raise FileNotFoundError(f"Feature file {file_path} not found.")
    except Exception as e:
        raise Exception(f"Error loading feature file: {str(e)}")


def train_and_evaluate_fold(
    clf,
    x_train,
    y_train,
    x_test,
    y_test,
    fold,
    output_file
):
    """
    Train the model and evaluate one fold.
    """
    clf.fit(x_train, y_train)

    y_pred_prob = clf.predict_proba(x_test)[:, 1]
    y_pred = (y_pred_prob > 0.5).astype(int)

    precision, recall, f1_score, _ = precision_recall_fscore_support(
        y_test,
        y_pred,
        average="binary"
    )

    accuracy = accuracy_score(y_test, y_pred)
    mcc = matthews_corrcoef(y_test, y_pred)

    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    specificity = tn / (tn + fp)

    fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
    roc_auc = auc(fpr, tpr)

    precision_pr, recall_pr, _ = precision_recall_curve(
        y_test,
        y_pred_prob
    )

    aupr = average_precision_score(y_test, y_pred_prob)

    np.save(f"Y_pred_fold{fold - 1}.npy", y_pred_prob)
    np.save(f"Y_test_fold{fold - 1}.npy", y_test)

    print(f"\nFold {fold}:")
    print(classification_report(y_test, y_pred))
    print(
        f"Accuracy: {accuracy:.4f}, "
        f"Precision: {precision:.4f}, "
        f"Recall: {recall:.4f}"
    )
    print(
        f"Specificity: {specificity:.4f}, "
        f"F1-score: {f1_score:.4f}, "
        f"MCC: {mcc:.4f}"
    )
    print(f"AUC: {roc_auc:.4f}, AUPR: {aupr:.4f}")

    with open(output_file, "a", encoding="utf-8") as f:
        f.write(
            f"{fold}\t{accuracy:.4f}\t{precision:.4f}\t"
            f"{recall:.4f}\t{specificity:.4f}\t"
            f"{f1_score:.4f}\t{mcc:.4f}\t"
            f"{roc_auc:.4f}\t{aupr:.4f}\n"
        )

    return precision, recall, f1_score, accuracy, mcc, specificity, roc_auc, aupr


def plot_roc_curves(n_folds=5):
    """
    Plot ROC curves across all folds.
    """
    tprs = []
    aucs = []
    mean_fpr = np.linspace(0, 1, 10000)

    plt.figure()

    for i in range(n_folds):
        predicted = np.load(f"Y_pred_fold{i}.npy")
        true_labels = np.load(f"Y_test_fold{i}.npy")

        fpr, tpr, _ = roc_curve(true_labels, predicted)
        roc_auc = auc(fpr, tpr)
        aucs.append(roc_auc)

        tprs.append(interp(mean_fpr, fpr, tpr))
        tprs[-1][0] = 0.0

        plt.plot(
            fpr,
            tpr,
            lw=1.5,
            label=f"ROC fold {i + 1} (AUC = {roc_auc:.4f})"
        )

    plt.plot(
        [0, 1],
        [0, 1],
        linestyle="--",
        lw=1,
        color="r",
        alpha=0.5
    )

    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0

    mean_auc = auc(mean_fpr, mean_tpr)
    std_auc = np.std(aucs, ddof=1)

    plt.plot(
        mean_fpr,
        mean_tpr,
        color="black",
        lw=1.5,
        label=f"Mean ROC (AUC = {mean_auc:.4f} ± {std_auc:.4f})"
    )

    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend(loc="lower right")
    plt.savefig("ROC_Curve.tif", dpi=300, bbox_inches="tight")
    plt.show()

    return mean_auc, std_auc


def plot_pr_curves(n_folds=5):
    """
    Plot Precision-Recall curves across all folds.
    """
    mean_recall = np.linspace(0, 1, 1000)
    precision_values = []
    aupr_values = []

    plt.figure()

    for i in range(n_folds):
        predicted = np.load(f"Y_pred_fold{i}.npy")
        true_labels = np.load(f"Y_test_fold{i}.npy")

        precision, recall, _ = precision_recall_curve(
            true_labels,
            predicted
        )

        aupr = average_precision_score(true_labels, predicted)
        aupr_values.append(aupr)

        precision_interp = interp(
            mean_recall,
            recall[::-1],
            precision[::-1]
        )
        precision_values.append(precision_interp)

        plt.plot(
            recall,
            precision,
            lw=1.5,
            label=f"PR fold {i + 1} (AUPR = {aupr:.4f})"
        )

    mean_precision = np.mean(precision_values, axis=0)
    mean_aupr = np.mean(aupr_values)
    std_aupr = np.std(aupr_values, ddof=1)

    plt.plot(
        mean_recall,
        mean_precision,
        color="black",
        lw=1.5,
        label=f"Mean PR (AUPR = {mean_aupr:.4f} ± {std_aupr:.4f})"
    )

    plt.xlabel("Recall", fontsize=14)
    plt.ylabel("Precision", fontsize=14)
    plt.legend(loc="lower left")
    plt.savefig("PR_Curve.tif", dpi=300, bbox_inches="tight")
    plt.show()

    return mean_aupr, std_aupr


def main():
    """
    Main function for 5-fold cross-validation and evaluation.
    """
    feature_file = "SampleFeatures.csv"

    X = load_features(feature_file)

    y = np.concatenate(
        (
            np.ones(len(X) // 2),
            np.zeros(len(X) // 2)
        )
    )

    clf = MLPClassifier(
        learning_rate_init=0.00012,
        random_state=80
    )

    skf = KFold(
        n_splits=5,
        shuffle=True,
        random_state=80
    )

    output_file = "5_fold_results.txt"

    metrics = {
        "precision": [],
        "recall": [],
        "f1_score": [],
        "accuracy": [],
        "mcc": [],
        "specificity": [],
        "auc": [],
        "aupr": []
    }

    with open(output_file, "w", encoding="utf-8") as f:
        f.write(
            "Fold\tAccuracy\tPrecision\tRecall\tSpecificity\t"
            "F1-score\tMCC\tAUC\tAUPR\n"
        )

    for fold, (train_idx, test_idx) in enumerate(skf.split(X, y), 1):
        x_train, y_train = X[train_idx], y[train_idx]
        x_test, y_test = X[test_idx], y[test_idx]

        fold_metrics = train_and_evaluate_fold(
            clf,
            x_train,
            y_train,
            x_test,
            y_test,
            fold,
            output_file
        )

        for key, value in zip(metrics.keys(), fold_metrics):
            metrics[key].append(value)

    print("\nAverage Performance Metrics:")

    for key in metrics:
        print(f"Average {key}: {np.mean(metrics[key]):.4f}")

    mean_auc, std_auc = plot_roc_curves(n_folds=5)
    mean_aupr, std_aupr = plot_pr_curves(n_folds=5)

    with open(output_file, "a", encoding="utf-8") as f:
        f.write(
            f"Average\t{np.mean(metrics['accuracy']):.4f}\t"
            f"{np.mean(metrics['precision']):.4f}\t"
            f"{np.mean(metrics['recall']):.4f}\t"
            f"{np.mean(metrics['specificity']):.4f}\t"
            f"{np.mean(metrics['f1_score']):.4f}\t"
            f"{np.mean(metrics['mcc']):.4f}\t"
            f"{mean_auc:.4f}\t{mean_aupr:.4f}\n"
        )

        f.write(
            f"Std\t{np.std(metrics['accuracy'], ddof=1):.4f}\t"
            f"{np.std(metrics['precision'], ddof=1):.4f}\t"
            f"{np.std(metrics['recall'], ddof=1):.4f}\t"
            f"{np.std(metrics['specificity'], ddof=1):.4f}\t"
            f"{np.std(metrics['f1_score'], ddof=1):.4f}\t"
            f"{np.std(metrics['mcc'], ddof=1):.4f}\t"
            f"{std_auc:.4f}\t{std_aupr:.4f}\n"
        )


if __name__ == "__main__":
    main()
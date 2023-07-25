import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (confusion_matrix, classification_report, roc_curve, roc_auc_score,
                             f1_score, jaccard_score, log_loss, accuracy_score)


def evaluate_model_performance(name_of_model, y_hat, y_probs, y_test, plot=True):
    """
    Evaluate the Predictions made by the Model.
    """
    # Evaluate Model Performance #
    print("\n---------------------------------------------------------")
    print(f"MODEL PERFORMANCE EVALUATION METRICS: {name_of_model}")
    # Accuracy Score of the Test Set #
    print("\nTest Model Accuracy/Hit Rate:  ", np.round(accuracy_score(y_test, y_hat) * 100.0, 3), "%")

    # Confusion Matrix #
    cm = confusion_matrix(y_test, y_hat)
    print("\nConfusion Matrix:")
    print("       Pred:-1|Pred:1")
    print("Actual:-1", cm[0, :])
    print("Actual:1 ", cm[1, :])

    # ROC AUC Score #
    roc_auc = roc_auc_score(y_test, y_probs)
    print("\nROC AUC:   ", np.round(roc_auc * 100.0, 3), "% (no-skill = 50%)")

    # Log-Loss #
    logloss = log_loss(y_test, y_probs)
    print("Log-loss:  ", np.round(logloss * 100.0, 3), "%")

    # Jacard #
    jacard = jaccard_score(y_test, y_hat)
    print("Jacard:    ", np.round(jacard * 100.0, 3), "%")

    # F1 Score #
    print("F1-Score:  ", np.round(f1_score(y_test, y_hat) * 100.0, 3), "%")

    # Classification Report #
    cr = classification_report(y_test, y_hat)
    print("\n                       Classification Report")
    print(cr)
    print("---------------------------------------------------------")

    # Visualize the ROC-AUC Score #
    fpr, tpr, _ = roc_curve(y_test, y_probs)

    if plot == True:
        fig = plt.figure(dpi=50)
        plt.plot(fpr, tpr, "black")
        plt.fill_between(fpr, tpr, facecolor='red', alpha=0.4)
        plt.text(0.95, 0.05, 'AUC = %0.4f' % roc_auc, ha='right', fontsize=12, weight='bold', color='black')
        plt.plot([0, 1], [0, 1], c="black", ls="--")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("ROC-AUC CURVE")
        plt.xlim([0, 1])
        plt.ylim([0, 1])
        plt.show()
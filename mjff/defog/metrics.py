import numpy as np
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score

def compute_metrics(y_true, y_pred, num_classes):
    cm = confusion_matrix(y_true, y_pred, labels=list(range(num_classes)))

    acc = np.mean(np.array(y_true) == np.array(y_pred))
    bal_acc = np.mean([
        cm[i, i] / cm[i].sum() if cm[i].sum() > 0 else 0.0
        for i in range(num_classes)
    ])

    precision = precision_score(y_true, y_pred, average="macro", zero_division=0)
    recall = recall_score(y_true, y_pred, average="macro", zero_division=0)
    f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)

    return {
        "accuracy": acc,
        "balanced_accuracy": bal_acc,
        "precision": precision,
        "recall_sensitivity": recall,
        "f1": f1,
        "confusion_matrix": cm.tolist()
    }

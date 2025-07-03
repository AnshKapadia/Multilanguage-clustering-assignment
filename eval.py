import numpy as np
from sklearn.metrics import accuracy_score
from scipy.optimize import linear_sum_assignment

def acc(filenames, predicted_labels):
    # Inputs
    true_labels = [int(f.split("_")[0]) for f in filenames]
    true_labels = np.array(true_labels)
    predicted_labels = np.array(predicted_labels)

    # Confusion matrix
    D = max(true_labels.max(), predicted_labels.max()) + 1
    confusion = np.zeros((D, D), dtype=np.int64)
    for t, p in zip(true_labels, predicted_labels):
        confusion[t, p] += 1

    # Hungarian alignment
    row_ind, col_ind = linear_sum_assignment(-confusion)
    label_map = {col: row for row, col in zip(row_ind, col_ind)}

    # Remap predicted labels
    mapped_preds = [label_map[p] for p in predicted_labels]

    # Accuracy
    accuracy = accuracy_score(true_labels, mapped_preds) * 100
    return accuracy
    

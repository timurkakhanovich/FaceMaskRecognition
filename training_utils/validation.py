from typing import Tuple
import numpy as np

from sklearn.metrics import auc
from sklearn.model_selection import KFold
from scipy import interpolate


def evaluate_lfw(
    distances: np.ndarray,
    labels: np.ndarray,
    num_folds: int = 10,
    far_target: float = 1e-1,
) -> Tuple[float]:
    # Calculate ROC metrics
    thresholds_roc = np.arange(0, 4, 0.01)
    TPR, FPR, precision, recall, accuracy, best_distances = \
        calculate_roc_values(
            thresholds=thresholds_roc, distances=distances, labels=labels, num_folds=num_folds
        )
    
    roc_auc = auc(FPR, TPR)

    # Calculate validation rate
    thresholds_val = np.arange(0, 4, 0.01)
    TAR, FAR = calculate_val(
        thresholds_val=thresholds_val, distances=distances, labels=labels, far_target=far_target, num_folds=num_folds
    )

    return TPR, FPR, precision, recall, accuracy, roc_auc, best_distances, TAR, FAR


def calculate_roc_values(
    thresholds: np.ndarray,
    distances: np.ndarray,
    labels: np.ndarray,
    num_folds: int,
) -> Tuple[float]:
    num_pairs = min(len(labels), len(distances))
    num_thresholds = len(thresholds)
    k_fold = KFold(n_splits=num_folds, shuffle=False)

    true_positive_rates = np.zeros((num_folds, num_thresholds))
    false_positive_rates = np.zeros((num_folds, num_thresholds))
    precision = np.zeros(num_folds)
    recall = np.zeros(num_folds)
    accuracy = np.zeros(num_folds)
    best_distances = np.zeros(num_folds)

    indices = np.arange(num_pairs)

    for fold_index, (train_set, test_set) in enumerate(k_fold.split(indices)):
        # Find the best distance threshold for the k-fold cross validation using the train set
        accuracies_trainset = np.zeros(num_thresholds)
        for threshold_index, threshold in enumerate(thresholds):
            _, _, _, _, accuracies_trainset[threshold_index] = calculate_metrics(
                threshold=threshold, dist=distances[train_set], actual_issame=labels[train_set]
            )
        best_threshold_index = np.argmax(accuracies_trainset)
    
        # Test on test set using the best distance threshold
        for threshold_index, threshold in enumerate(thresholds):
            true_positive_rates[fold_index, threshold_index], false_positive_rates[fold_index, threshold_index], _, _, _ = \
                calculate_metrics(
                    threshold=threshold, dist=distances[test_set], actual_issame=labels[test_set]
                )
        
        _, _, precision[fold_index], recall[fold_index], accuracy[fold_index] = calculate_metrics(
            threshold=thresholds[best_threshold_index], dist=distances[test_set], actual_issame=labels[test_set]
        )

        TPR = np.mean(true_positive_rates, 0)
        FPR = np.mean(false_positive_rates, 0)
        best_distances[fold_index] = thresholds[best_threshold_index]
    
    return TPR, FPR, precision, recall, accuracy, best_distances


def calculate_metrics(threshold: float, dist: float, actual_issame: bool) -> Tuple[float]:
    predict_issame = np.less(dist, threshold)

    TPs = np.sum(np.logical_and(predict_issame, actual_issame))
    FPs = np.sum(np.logical_and(predict_issame, np.logical_not(actual_issame)))
    TNs = np.sum(np.logical_and(np.logical_not(predict_issame), np.logical_not(actual_issame)))
    FNs = np.sum(np.logical_and(np.logical_not(predict_issame), actual_issame))

    # For dealing with Divide By Zero exception
    TPR = 0 if (TPs + FPs == 0) else \
        float(TPs) / float(TPs + FNs)

    FPR = 0 if (FPs + TNs == 0) else \
        float(FPs) / float(FPs + TNs)

    precision = 0 if (TPs + FPs == 0) else \
        float(TPs) / float(TPs + FPs)

    recall = 0 if (TPs + FNs == 0) else \
        float(TPs) / float(TPs + FNs)
    
    accuracy = float(TPs + TNs) / dist.size

    return TPR, FPR, precision, recall, accuracy


def calculate_val(
    thresholds_val: float,
    distances: np.ndarray,
    labels: np.ndarray,
    far_target: float = 1e-3,
    num_folds: int = 10,
) -> Tuple[float]:
    num_pairs = min(len(distances), len(labels))
    num_thresholds = len(thresholds_val)
    k_folds = KFold(n_splits=num_folds, shuffle=False)

    TAR = np.zeros(num_folds)
    FAR = np.zeros(num_folds)

    indices = np.arange(num_pairs)

    for fold_index, (train_set, test_set) in enumerate(k_folds.split(indices)):
        # Find the euclidean distance threshold that gives false acceptance rate (far) = far_target
        FAR_train = np.zeros(num_thresholds)
        for threshold_index, threshold in enumerate(thresholds_val):
            _, FAR_train[threshold_index] = calculate_val_far(
                threshold=threshold, dist=distances[train_set], actual_issame=labels[train_set]
            )
        if np.max(FAR_train) >= far_target:
            f = interpolate.interp1d(FAR_train, thresholds_val, kind='slinear')
            threshold = f(far_target)
        else:
            threshold = 0.0
        
        TAR[fold_index], FAR[fold_index] = calculate_val_far(
            threshold=threshold, dist=distances[test_set], actual_issame=labels[test_set]
        )
    
    return TAR, FAR


def calculate_val_far(threshold: float, dist: float, actual_issame: bool) -> Tuple[float]:
    # If distance is less than threshold, then prediction is set to True
    predict_issame = np.less(dist, threshold)

    TA = np.sum(np.logical_and(predict_issame, actual_issame))
    FA = np.sum(np.logical_and(predict_issame, np.logical_not(actual_issame)))

    num_same = np.sum(actual_issame)
    num_diff = np.sum(np.logical_not(actual_issame))

    if num_diff == 0:
        num_diff = 1
    if num_same == 0:
        return 0, 0
    
    TAR = float(TA) / float(num_same)
    FAR = float(FA) / float(num_diff)

    return TAR, FAR

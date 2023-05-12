from enum import Enum

ERROR_WEIGHTS = {
    "precision": {
        "true_positives": 1,
        "false_negatives": 1,
        "false_positives": 2,
        "ClassificationError": 10,
        "ClassificationAndLocalizationError": 5,
        "BackgroundError": 10,
    },
    "recall": {
        "true_positives": 1,
        "false_positives": 1,
        "false_negatives": 5,
        "MissedError": 10,
        "ClassificationError": 1.001,
        "ClassificationAndLocalizationError": 1.001,
        "LocalizationError": 1.001,
    },
    "f1": {
        "true_positives": 1,
        "false_positives": 2,
        "false_negatives": 5,
        "MissedError": 12,
        "ClassificationError": 10,
        "ClassificationAndLocalizationError": 5,
        "BackgroundError": 10,
    },
}

ERROR_ORDERS = {
    "precision": {
        "confusions": 10,
    },
    "recall": {
        "confusions": 1.001,
    },
    "f1": {
        "confusions": 10,
    },
}


class ErrorWeights(str, Enum):
    PRECISION = "precision"
    RECALL = "recall"
    F1 = "f1"

    @property
    def weights(self):
        return ERROR_WEIGHTS[self]

    @property
    def orders(self):
        return ERROR_ORDERS[self]

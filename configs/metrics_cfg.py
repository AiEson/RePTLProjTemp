import torch
from torchmetrics import (Accuracy, F1Score, FBetaScore, JaccardIndex,
                          MetricCollection, Precision, Recall)

_METRICS = MetricCollection(
    [
        Accuracy(num_classes=2, task="binary"),
        F1Score(num_classes=2, task="binary"),
        FBetaScore(num_classes=2, task="binary"),
        JaccardIndex(num_classes=2, task="binary"),
        Precision(num_classes=2, task="binary"),
        Recall(num_classes=2, task="binary"),
    ]
)


def get_metrics_collection():
    return _METRICS


if __name__ == "__main__":
    metrics = get_metrics_collection()
    mask = torch.randint(size=[2, 1, 128, 128], low=0, high=2)
    logestic = torch.randn(2, 1, 128, 128)

    __import__("pprint").pprint(metrics(mask.float(), mask))

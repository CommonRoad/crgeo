from typing import Tuple, TypedDict

import torch
from torch import IntTensor, FloatTensor

# https://en.m.wikipedia.org/wiki/Confusion_matrix#Table_of_confusion


class BaseConfusionMatrix(TypedDict):
    positive: IntTensor
    predicted_positive: IntTensor
    negative: IntTensor
    predicted_negative: IntTensor
    true_positive: IntTensor
    false_negative: IntTensor
    false_positive: IntTensor
    true_negative: IntTensor


class ExtendedConfusionMatrix(BaseConfusionMatrix):
    sensitivity: FloatTensor
    specificity: FloatTensor
    prevalence: FloatTensor
    precision: FloatTensor
    accuracy: FloatTensor
    f1_score: FloatTensor


def _to_float32(*ts: IntTensor) -> Tuple[FloatTensor, ...]:
    return tuple(t.type(torch.float32) for t in ts)


def complete_confusion_matrix(conf_matrix: BaseConfusionMatrix) -> ExtendedConfusionMatrix:
    P, N = _to_float32(conf_matrix["positive"], conf_matrix["negative"])
    # PP, PN = _to_float32(conf_matrix["predicted_positive"], conf_matrix["predicted_negative"])
    TP, FN, FP, TN = _to_float32(conf_matrix["true_positive"], conf_matrix["false_negative"],
                                 conf_matrix["false_positive"], conf_matrix["true_negative"])
    sensitivity = TP / P
    precision = TP / (TP + FP)
    return ExtendedConfusionMatrix(
        **conf_matrix,
        sensitivity=sensitivity,
        specificity=TN / N,
        prevalence=P / (P + N),
        precision=precision,
        accuracy=(TP + TN) / (P + N),
        f1_score=2 * (precision * sensitivity) / (precision + sensitivity),
    )

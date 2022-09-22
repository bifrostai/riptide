from typing import List, Union

import torch
from torchvision.ops.boxes import box_iou

"""Error types as defined in TIDE (https://arxiv.org/pdf/2008.08115.pdf).

IoUmax to denote a false positive's maximum IoU overlap with a ground truth of the
given category. The foreground IoU iou_threshold is denoted as tf and the background
iou_threshold is denoted as tb, which are set to 0.5 and 0.1 unless otherwise noted.

(0) Classification: IoUmax ≥ tf for GT of the incorrect class (i.e., localized
    correctly but classified incorrectly).
(1) Localization: tb ≤ IoUmax ≤ tf for GT of the correct class (i.e., classified
    correctly but localized incorrectly).
(2) Classification-Localization: tb ≤ IoUmax ≤ tf for GT of the incorrect class
    (i.e., classified incorrectly and localized incorrectly).
(3) Duplicate: IoUmax ≥ tf for GT of the correct class but another higher-scoring
    detection already matched that GT (i.e., would be correct if not for a higher
    scoring detection).
(4) Background: IoUmax ≤ tb for all GT (i.e., detected background as foreground).
(5) Missed: All undetected ground truth (false negatives) not already covered by
    classification or localization error.
"""


class WrongCategoryException(Exception):
    """Raised when there is an unexpected category for the error"""

    __module__ = Exception.__module__


class LowConfidenceException(Exception):
    """Raised when the prediction has low confidence and should have been discarded"""

    __module__ = Exception.__module__


class HighOverlapException(Exception):
    """Raised when the overlap exceeds the threshold to be considered an error"""

    __module__ = Exception.__module__


class NotDuplicateException(Exception):
    """Raised when the prediction does not seem to be a duplicate"""

    __module__ = Exception.__module__


def jaccard_overlap(pred_bbox: torch.Tensor, gt_bbox: torch.Tensor) -> float:
    if len(pred_bbox.size()) == 1:
        pred_bbox = pred_bbox.unsqueeze(0)
    if len(gt_bbox.size()) == 1:
        gt_bbox = gt_bbox.unsqueeze(0)
    return box_iou(pred_bbox, gt_bbox).item()


class Error:
    def __repr__(self) -> str:
        attrs = [f"{x}={getattr(self, x)}" for x in dir(self) if not x.startswith("_")]
        attrs = ", ".join(attrs)
        return f"{self.__class__.__name__}({attrs})"


class ClassificationError(Error):
    def __init__(
        self,
        pred_idx: int,
        gt_idx: int,
        pred_label: int,
        pred_bbox: torch.Tensor,
        confidence: float,
        gt_label: int,
        gt_bbox: torch.Tensor,
        conf_threshold: float = 0.5,
    ) -> None:
        if pred_label == 0:
            raise WrongCategoryException(
                "Classification errors must have prediction category > 0 "
                "(zero is background)"
            )
        if gt_label == 0:
            raise WrongCategoryException(
                "The ground truth category must have value > 0"
            )
        if confidence < conf_threshold:
            raise LowConfidenceException(
                f"The prediction must have conf >= {conf_threshold}, got {confidence}"
            )
        self.pred_idx = pred_idx
        self.gt_idx = gt_idx
        self.pred_label = pred_label
        self.pred_bbox = pred_bbox
        self.confidence = confidence
        self.gt_label = gt_label
        self.gt_bbox = gt_bbox
        self.conf_threshold = conf_threshold
        self.code = "CLS"


class LocalizationError(Error):
    def __init__(
        self,
        pred_idx: int,
        gt_idx: int,
        pred_label: int,
        pred_bbox: torch.Tensor,
        confidence: float,
        gt_label: int,
        gt_bbox: torch.Tensor,
        iou_threshold: float = 0.5,
        conf_threshold: float = 0.5,
    ) -> None:
        if pred_label == 0:
            raise WrongCategoryException(
                "Localization errors must have prediction category > 0 "
                "(zero is background)"
            )
        if pred_label != gt_label:
            raise WrongCategoryException(
                "For a localization error, the predicted category must be the same as "
                "the target category. Use ClassificationAndLocalizationError instead"
            )
        if gt_label == 0:
            raise WrongCategoryException(
                "The ground truth category must have value > 0"
            )
        if jaccard_overlap(pred_bbox, gt_bbox) > iou_threshold:
            raise HighOverlapException(
                f"The overlap between the two specified bboxes ({pred_bbox} and "
                f"{gt_bbox}) greater than the the specified iou_threshold "
                f"({iou_threshold})"
            )
        if confidence < conf_threshold:
            raise LowConfidenceException(
                f"The prediction must have conf >= {conf_threshold}, got {confidence}"
            )
        self.pred_idx = pred_idx
        self.gt_idx = gt_idx
        self.pred_label = pred_label
        self.pred_bbox = pred_bbox
        self.confidence = confidence
        self.gt_label = gt_label
        self.gt_bbox = gt_bbox
        self.iou_threshold = iou_threshold
        self.conf_threshold = conf_threshold
        self.code = "LOC"


class ClassificationAndLocalizationError(Error):
    def __init__(
        self,
        pred_idx: int,
        gt_idx: int,
        pred_label: int,
        pred_bbox: torch.Tensor,
        confidence: float,
        gt_label: int,
        gt_bbox: torch.Tensor,
        iou_threshold: float = 0.5,
        conf_threshold: float = 0.5,
    ) -> None:
        if pred_label == 0:
            raise WrongCategoryException(
                "Classification-localization errors must have prediction category > 0 "
                "(zero is background)"
            )
        if gt_label == 0:
            raise WrongCategoryException(
                "The ground truth category must have value > 0"
            )
        if pred_label == gt_label:
            raise WrongCategoryException(
                "For a classification-localization error, the predicted category must "
                "be different from the target category. Use LocalizationError instead"
            )
        if jaccard_overlap(pred_bbox, gt_bbox) > iou_threshold:
            raise HighOverlapException(
                f"The overlap between the two specified bboxes ({pred_bbox} and "
                f"{gt_bbox}) is greater than the specified iou_threshold "
                f"({iou_threshold})"
            )
        if confidence < conf_threshold:
            raise LowConfidenceException(
                f"The prediction must have conf >= {conf_threshold}, got {confidence}"
            )
        self.pred_idx = pred_idx
        self.gt_idx = gt_idx
        self.pred_label = pred_label
        self.pred_bbox = pred_bbox
        self.confidence = confidence
        self.gt_label = gt_label
        self.gt_bbox = gt_bbox
        self.iou_threshold = iou_threshold
        self.conf_threshold = conf_threshold
        self.code = "CLL"


class DuplicateError(Error):
    def __init__(
        self,
        pred_idx: int,
        best_pred_idx: int,
        gt_idx: int,
        pred_label: int,
        pred_bbox: torch.Tensor,
        confidence: float,
        best_pred_label: int,
        best_pred_bbox: torch.Tensor,
        best_confidence: float,
        gt_label: int,
        gt_bbox: torch.Tensor,
        iou_threshold: float = 0.5,
        conf_threshold: float = 0.5,
    ) -> None:
        if pred_label == 0:
            raise WrongCategoryException(
                "Duplicate errors must have prediction category > 0 "
                "(zero is background)"
            )
        if gt_label == 0:
            raise WrongCategoryException(
                "The ground truth category must have value > 0"
            )
        if pred_label != gt_label:
            raise WrongCategoryException(
                f"For a duplicate error, the predicted category ({pred_label}) must"
                " match the target category"
            )
        if best_pred_label != gt_label:
            raise WrongCategoryException(
                "For a duplicate error, the best predicted category "
                f"({best_pred_label}) must have been correct (must match the target"
                " category)"
            )
        if pred_label != best_pred_label:
            raise WrongCategoryException(
                f"For a duplicate error, the predicted category ({pred_label}) must"
                " match the highest scoring predicted category"
            )
        if jaccard_overlap(pred_bbox, gt_bbox) < iou_threshold:
            raise HighOverlapException(
                f"The overlap between the prediction ({pred_bbox} and the target "
                f"{gt_bbox}) is not >= iou_threshold ({iou_threshold})"
            )
        if jaccard_overlap(pred_bbox, gt_bbox) > jaccard_overlap(
            best_pred_bbox, gt_bbox
        ):
            raise NotDuplicateException(
                f"The overlap between the target bbox ({gt_bbox}) and the predicted "
                f"bbox ({pred_bbox}) must be smaller than the overlap between the "
                f"target bbox and the best bbox ({best_pred_bbox})"
            )
        if confidence < conf_threshold:
            raise LowConfidenceException(
                f"The prediction must have conf >= {conf_threshold}, got {confidence}"
            )
        self.pred_idx = pred_idx
        self.best_pred_idx = best_pred_idx
        self.gt_idx = gt_idx
        self.pred_label = pred_label
        self.pred_bbox = pred_bbox
        self.confidence = confidence
        self.best_pred_label = best_pred_label
        self.best_pred_bbox = best_pred_bbox
        self.best_confidence = best_confidence
        self.gt_label = gt_label
        self.gt_bbox = gt_bbox
        self.iou_threshold = iou_threshold
        self.conf_threshold = conf_threshold
        self.code = "DUP"


class BackgroundError(Error):
    def __init__(
        self,
        pred_idx: int,
        pred_label: int,
        pred_bbox: torch.Tensor,
        confidence: float,
        conf_threshold: float = 0.5,
    ) -> None:
        if pred_label == 0:
            raise WrongCategoryException(
                "Background errors must have prediction category > 0 (zero is "
                "background). Background errors refer to background areas detected as "
                "foreground objects."
            )
        if confidence < conf_threshold:
            raise LowConfidenceException(
                f"The prediction must have conf >= {conf_threshold}, got {confidence}"
            )
        self.pred_idx = pred_idx
        self.pred_label = pred_label
        self.confidence = confidence
        self.pred_bbox = pred_bbox
        self.conf_threshold = conf_threshold
        self.code = "BKG"


class MissedError(Error):
    def __init__(
        self,
        gt_idx: int,
        gt_label: int,
        gt_bbox: torch.Tensor,
    ) -> None:
        if gt_label == 0:
            raise WrongCategoryException(
                f"The ground truth category must have value > 0, got {gt_label}"
            )
        self.gt_idx = gt_idx
        self.gt_label = gt_label
        self.gt_bbox = gt_bbox
        self.code = "MIS"


class Errors:
    """Errors assigned to predictions or ground truths."""

    def __init__(self, num_preds: int, num_gt: int) -> None:
        self.num_preds = num_preds
        self.num_gt = num_gt
        self.pred_errors = [None] * num_preds
        self.gt_errors = [None] * num_gt

    def assign_prediction_error(self, error: Error) -> None:
        if isinstance(error, MissedError):
            raise Exception(
                "A prediction error must be a BackgroundError, ClassificationError, "
                "LocalizationError, ClassificationAndLocalizationError, or "
                "DuplicateError"
            )
        if self.pred_errors[error.pred_idx] is not None:
            raise Exception("Tried to assign an already assigned error")
        self.pred_errors[error.pred_idx] = error

    def assign_gt_error(self, error: MissedError) -> None:
        if not isinstance(error, MissedError):
            raise Exception("A ground truth error must be a MissedError")
        if self.gt_errors[error.gt_idx] is not None:
            raise Exception("Tried to assign an already assigned error")
        self.gt_errors[error.gt_idx] = error

    def remove_gt_error(self, gt_idx: int) -> None:
        if self.gt_errors[gt_idx] is None:
            raise Exception("Trying to remove the MissedError but it does not exist")
        self.gt_errors[gt_idx] = None

    def check_prediction_error(self, pred_idx: int) -> Union[Error, None]:
        return self.pred_errors[pred_idx]

    def check_gt_error(self, gt_idx: int) -> Union[Error, None]:
        return self.gt_errors[gt_idx]

    def get_prediction_errors(self) -> List[Error]:
        return [error for error in self.pred_errors if error]

    def get_gt_errors(self) -> List[Error]:
        return [error for error in self.gt_errors if error]

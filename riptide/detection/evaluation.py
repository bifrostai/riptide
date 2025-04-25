from __future__ import annotations

import os
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple, Type, Union

import pandas as pd
import torch
from termcolor import colored
from torchvision.ops.boxes import box_iou
from torchvision.transforms.functional import crop, to_pil_image
from torchvision.utils import draw_bounding_boxes

from riptide.detection.confusions import Confusion, Confusions
from riptide.detection.errors import (
    BackgroundError,
    ClassificationAndLocalizationError,
    ClassificationError,
    DuplicateError,
    Error,
    Errors,
    LocalizationError,
    MissedError,
    NonError,
)
from riptide.io.loaders import COCOLoader, DictLoader
from riptide.utils.colors import ErrorColor
from riptide.utils.image import read_image
from riptide.utils.logging import logger
from riptide.utils.models import GTData
from riptide.utils.obb import obb_iou

ERROR_TYPES: List[Error] = [
    BackgroundError,
    ClassificationError,
    LocalizationError,
    ClassificationAndLocalizationError,
    DuplicateError,
    MissedError,
    NonError,
]


class Status:
    """A status for a prediction or ground truth."""

    def __init__(
        self,
        state: Union[Error, Confusion],
        score: float,
        gt_label: Optional[int] = None,
        pred_label: Optional[int] = None,
        iou: Optional[float] = None,
    ) -> None:
        self.state = state
        self.gt_label = gt_label
        self.pred_label = pred_label
        self.score = score
        self.iou = iou or 0

    @property
    def code(self) -> str:
        return self.state.code

    def __str__(self) -> str:
        return str(self.state.code)

    def __repr__(self) -> str:
        return (
            f"Status(state={self.code}, gt_label={self.gt_label},"
            f" pred_label={self.pred_label}, score={self.score}, iou={self.iou})"
        )

    def __eq__(self, other: Status) -> bool:
        return (
            self.state == other.state
            and self.gt_label == other.gt_label
            and self.pred_label == other.pred_label
            and self.score == other.score
            and self.iou == other.iou
        )

    def __hash__(self) -> int:
        return hash((self.state, self.gt_label, self.pred_label, self.score, self.iou))

    def todict(self) -> Dict[str, Any]:
        return {
            "state": self.state.code,
            "gt_label": self.gt_label,
            "pred_label": self.pred_label,
            "score": self.score,
            "iou": self.iou,
        }

    def copy(self) -> Status:
        return Status(
            state=self.state,
            score=self.score,
            gt_label=self.gt_label,
            pred_label=self.pred_label,
            iou=self.iou,
        )


class Evaluation:

    pred_errors: List[Error] = []
    gt_errors: List[Error] = []

    def __init__(
        self,
        idx: Union[str, int],
        /,
        pred_scores: torch.Tensor,
        pred_labels: torch.Tensor,
        gt_labels: torch.Tensor,
        gt_offset: int = 0,
        bg_iou_threshold: float = 0.1,
        fg_iou_threshold: float = 0.5,
        conf_threshold: float = 0.5,
    ) -> None:
        self.idx = idx

        self.num_preds = len(pred_labels)
        self.num_gts = len(gt_labels)
        self.gt_offset = gt_offset

        self.pred_scores = pred_scores
        self.pred_labels = pred_labels
        self.gt_labels = gt_labels
        self.conf_threshold = conf_threshold
        self.bg_iou_threshold = bg_iou_threshold
        self.fg_iou_threshold = fg_iou_threshold

        used_mask = pred_scores >= conf_threshold
        unused_mask = ~used_mask
        self.used_mask = used_mask

        pred_idxs = torch.arange(self.num_preds)
        self.used_pred_idxs = pred_idxs[used_mask]
        self.unused_pred_idxs = pred_idxs[unused_mask]

        self.errors = Errors(
            num_preds=self.num_preds,
            num_gts=self.num_gts,
        )

        self.confusions = Confusions(
            evaluation=self,
            num_preds=self.num_preds,
            num_gts=self.num_gts,
        )

    @property
    def instances(self) -> List[Error]:
        return self.pred_errors + self.gt_errors

    def _get_gt_id(self, gt_idx: int) -> Any:
        """Returns the ID of the ground truth at the given index."""
        return self.gt_offset + gt_idx if gt_idx is not None else None

    def get_gt_status(self) -> Dict[int, Status]:
        """Returns the statuses for the ground truths.

        Returns
        -------
        Dict[Any, Union[Confusion, Error]]
            A dictionary with the ground truth IDs as keys and the status as values.
        """
        statuses: Dict[Any, Status] = dict()
        for idx, confusion in enumerate(self.confusions.get_gt_confusions()):
            gt_id = self._get_gt_id(idx)
            gt_label = self.gt_labels[idx].item()
            pred_label = None
            error = self.errors.check_gt_error(idx)
            if error is None:
                error = confusion

            statuses[gt_id] = Status(
                state=error, gt_label=gt_label, pred_label=pred_label, score=0
            )

        return statuses

    def get_pred_status(self) -> Dict[Any, List[Status]]:
        """Returns the statuses for the predictions."""
        statuses: Dict[Any, List[Status]] = {None: []}
        for idx, confusion in enumerate(self.confusions.get_prediction_confusions()):
            gt_id = None
            pred_label = self.pred_labels[idx].item()
            gt_label = None
            error = self.errors.check_prediction_error(idx)
            if error is not None:
                gt_id = self._get_gt_id(error.gt_idx)
                gt_label = error.gt_label
                iou = (
                    self.ious[idx, error.gt_idx].item()
                    if isinstance(self, ObjectDetectionEvaluation)
                    and error.gt_idx is not None
                    else None
                )
            else:
                error = confusion
                iou = None

            value = Status(
                state=error,
                gt_label=gt_label,
                pred_label=pred_label,
                score=self.pred_scores[idx].item(),
                iou=iou,
            )
            if gt_id not in statuses:
                statuses[gt_id] = [value]
            else:
                statuses[gt_id].append(value)

        for status_list in statuses.values():
            status_list.sort(key=lambda x: (x.iou or 0, x.score), reverse=True)

        return statuses

    def get_status(self) -> Dict[Any, List[Status]]:
        """Returns the statuses for the predictions and ground truths.

        Returns
        -------
        Dict[Any, Union[Confusion, Error]]
            A dictionary with the ground truth IDs as keys and the status as values.
        """
        statuses = self.get_pred_status()

        for gt_idx, status in self.get_gt_status().items():
            if status.state is Confusion.TRUE_POSITIVE:
                continue

            if gt_idx not in statuses:
                statuses[gt_idx] = [status]
            else:
                statuses[gt_idx].append(status)

        return statuses

    def get_errors_by_gt(self) -> Dict[Any, List[Error]]:
        """Returns the detections for the ground truths.

        Returns
        -------
        Dict[Any, List[Error]]
            A dictionary with the ground truth IDs as keys and the errors as values.
        """
        errors: Dict[Any, List[Error]] = dict()
        for error in self.errors.gt_errors + self.errors.pred_errors:
            if error is None or error.gt_idx is None:
                continue
            gt_id = error.idx
            if gt_id not in errors:
                errors[gt_id] = [error]
            else:
                errors[gt_id].append(error)

        return errors


class ObjectDetectionEvaluation(Evaluation):
    """An object that creates and stores Errors and Confusions for a particular image."""

    def __init__(
        self,
        image_path: str,
        pred_bboxes: torch.Tensor,
        pred_scores: torch.Tensor,
        pred_labels: torch.Tensor,
        gt_bboxes: torch.Tensor,
        gt_labels: torch.Tensor,
        gt_offset: int = 0,
        bg_iou_threshold: float = 0.1,
        fg_iou_threshold: float = 0.5,
        conf_threshold: float = 0.5,
        obb: bool = False,
    ) -> None:
        super().__init__(
            image_path,
            pred_scores=pred_scores,
            pred_labels=pred_labels,
            gt_labels=gt_labels,
            gt_offset=gt_offset,
            bg_iou_threshold=bg_iou_threshold,
            fg_iou_threshold=fg_iou_threshold,
            conf_threshold=conf_threshold,
        )
        self.pred_bboxes = pred_bboxes
        self.gt_bboxes = gt_bboxes

        if not obb:
            self.ious = box_iou(pred_bboxes, gt_bboxes)
        else:
            self.ious = obb_iou(pred_bboxes, gt_bboxes)

        self._pred_errors = None
        self._gt_errors = None

        # There are no ground truths in the image
        if len(gt_bboxes) == 0:
            # All preds are BackgroundErrors and false positives
            for pred_idx, (pred_bbox, pred_conf, pred_label) in enumerate(
                zip(self.pred_bboxes, self.pred_scores, self.pred_labels)
            ):
                if pred_idx in self.unused_pred_idxs:
                    self.confusions.assign_prediction_confusion(
                        pred_idx, Confusion.UNUSED
                    )
                    continue

                background_error = BackgroundError(
                    evaluation=self,
                    pred_idx=pred_idx,
                    pred_label=pred_label,
                    pred_bbox=pred_bbox,
                    confidence=pred_conf,
                    conf_threshold=self.conf_threshold,
                )
                self.errors.assign_prediction_error(background_error)
                self.confusions.assign_prediction_confusion(
                    pred_idx, Confusion.FALSE_POSITIVE
                )
            return

        # No predictions were made
        if len(self.used_pred_idxs) == 0:
            for pred_idx in range(len(pred_bboxes)):
                self.confusions.assign_prediction_confusion(pred_idx, Confusion.UNUSED)

            # All gts are MissedErrors and false negatives
            for gt_idx, (gt_label, gt_bbox) in enumerate(zip(gt_labels, gt_bboxes)):
                missed_error = MissedError(
                    evaluation=self,
                    gt_idx=gt_idx,
                    gt_label=gt_label,
                    gt_bbox=gt_bbox,
                )
                self.errors.assign_gt_error(missed_error)
                self.confusions.assign_gt_confusion(gt_idx, Confusion.FALSE_NEGATIVE)
            return

        # First register all definite true positives
        for pred_idx, (pred_iou, pred_bbox, pred_conf, pred_label) in enumerate(
            zip(self.ious, self.pred_bboxes, self.pred_scores, self.pred_labels)
        ):
            if pred_idx in self.used_pred_idxs:
                idx_of_best_gt_match = pred_iou.argmax()
                iou_of_best_gt_match = pred_iou.max()
                label_of_best_gt_match = gt_labels[idx_of_best_gt_match]

                if self.is_true_positive(
                    pred_label,
                    iou_of_best_gt_match,
                    idx_of_best_gt_match,
                    label_of_best_gt_match,
                ):
                    true_positive = NonError(
                        evaluation=self,
                        pred_idx=pred_idx,
                        gt_idx=idx_of_best_gt_match,
                        pred_label=pred_label,
                        pred_bbox=pred_bbox,
                        confidence=pred_conf,
                        gt_label=label_of_best_gt_match,
                        gt_bbox=gt_bboxes[idx_of_best_gt_match],
                        iou_threshold=self.fg_iou_threshold,
                        conf_threshold=self.conf_threshold,
                    )
                    self.errors.assign_prediction_error(true_positive)
                    self.confusions.assign_prediction_confusion(
                        pred_idx, Confusion.TRUE_POSITIVE
                    )
                    self.confusions.assign_gt_confusion(
                        idx_of_best_gt_match, Confusion.TRUE_POSITIVE
                    )

        # Predictions were made with ground truths
        for pred_idx, (pred_iou, pred_bbox, pred_conf, pred_label) in enumerate(
            zip(self.ious, self.pred_bboxes, self.pred_scores, self.pred_labels)
        ):
            if pred_idx in self.unused_pred_idxs:
                self.confusions.assign_prediction_confusion(pred_idx, Confusion.UNUSED)
                continue

            if (
                self.confusions.check_prediction_confusion(pred_idx)
                is Confusion.TRUE_POSITIVE
            ):
                continue

            if self.is_background_error(pred_iou):
                background_error = BackgroundError(
                    evaluation=self,
                    pred_idx=pred_idx,
                    pred_label=pred_label,
                    pred_bbox=pred_bbox,
                    confidence=pred_conf,
                    conf_threshold=self.conf_threshold,
                )
                self.errors.assign_prediction_error(background_error)
                self.confusions.assign_prediction_confusion(
                    pred_idx, Confusion.FALSE_POSITIVE
                )
                continue

            idx_of_best_gt_match = pred_iou.argmax()
            iou_of_best_gt_match = pred_iou.max()
            box_of_best_gt_match = gt_bboxes[idx_of_best_gt_match]
            label_of_best_gt_match = gt_labels[idx_of_best_gt_match]

            # Test for LocalizationError: correct class but poor localization
            # This detection would have been positive if it had higher IoU with this GT
            if self.is_localization_error(
                pred_label,
                iou_of_best_gt_match,
                label_of_best_gt_match,
            ):
                localization_error = LocalizationError(
                    evaluation=self,
                    pred_idx=pred_idx,
                    gt_idx=idx_of_best_gt_match,
                    pred_label=pred_label,
                    pred_bbox=pred_bbox,
                    confidence=pred_conf,
                    gt_label=label_of_best_gt_match,
                    gt_bbox=box_of_best_gt_match,
                    iou_threshold=self.fg_iou_threshold,
                    conf_threshold=self.conf_threshold,
                )
                self.errors.assign_prediction_error(localization_error)

                # Track the false negative for this GT if not already tracked
                if self.confusions.check_gt_confusion(idx_of_best_gt_match) is None:
                    self.confusions.assign_gt_confusion(
                        idx_of_best_gt_match, Confusion.FALSE_NEGATIVE
                    )

                # Assign the false positive to the prediction
                self.confusions.assign_prediction_confusion(
                    pred_idx, Confusion.FALSE_POSITIVE
                )
                continue

            # Test for ClassificationError: accurate localization but wrong class
            # This detection would have been a positive if it was the correct class
            if self.is_classification_error(
                pred_label,
                iou_of_best_gt_match,
                label_of_best_gt_match,
            ):
                classification_error = ClassificationError(
                    evaluation=self,
                    pred_idx=pred_idx,
                    gt_idx=idx_of_best_gt_match,
                    pred_label=pred_label,
                    pred_bbox=pred_bbox,
                    confidence=pred_conf,
                    gt_label=label_of_best_gt_match,
                    gt_bbox=box_of_best_gt_match,
                    conf_threshold=self.conf_threshold,
                )
                self.errors.assign_prediction_error(classification_error)

                # Track the false negative for this GT if not already tracked
                if self.confusions.check_gt_confusion(idx_of_best_gt_match) is None:
                    self.confusions.assign_gt_confusion(
                        idx_of_best_gt_match, Confusion.FALSE_NEGATIVE
                    )

                # Assign the false positive to the prediction
                self.confusions.assign_prediction_confusion(
                    pred_idx, Confusion.FALSE_POSITIVE
                )
                continue

            # Test for ClassificationAndLocalizationError: Wrong class and poor localization
            if self.is_classification_and_localization_error(
                pred_label,
                iou_of_best_gt_match,
                label_of_best_gt_match,
            ):
                classification_localization_error = ClassificationAndLocalizationError(
                    evaluation=self,
                    pred_idx=pred_idx,
                    gt_idx=idx_of_best_gt_match,
                    pred_label=pred_label,
                    pred_bbox=pred_bbox,
                    confidence=pred_conf,
                    gt_label=label_of_best_gt_match,
                    gt_bbox=box_of_best_gt_match,
                    iou_threshold=self.fg_iou_threshold,
                    conf_threshold=self.conf_threshold,
                )
                self.errors.assign_prediction_error(classification_localization_error)

                # Track the false negative for this GT if not already tracked
                if self.confusions.check_gt_confusion(idx_of_best_gt_match) is None:
                    self.confusions.assign_gt_confusion(
                        idx_of_best_gt_match, Confusion.FALSE_NEGATIVE
                    )

                # Assign the false positive to the prediction
                self.confusions.assign_prediction_confusion(
                    pred_idx, Confusion.FALSE_POSITIVE
                )
                continue

            # Test for DuplicateError: predictions with a lower IoU than an existing true positive
            # This detection would have been a true positive if it had the highest IoU with this GT
            if self.is_duplicate_error(
                pred_label,
                iou_of_best_gt_match,
                idx_of_best_gt_match,
                label_of_best_gt_match,
            ):
                col_including_unused = self.ious[:, idx_of_best_gt_match].clone()
                col_including_unused[self.unused_pred_idxs] = float("-Inf")
                col_including_unused[
                    self.pred_labels != label_of_best_gt_match
                ] = float("-Inf")
                idx_of_best_pred_match = col_including_unused.argmax()
                best_pred_label = self.pred_labels[idx_of_best_pred_match]
                best_pred_bbox = self.pred_bboxes[idx_of_best_pred_match]
                best_confidence = self.pred_scores[idx_of_best_pred_match]
                duplicate_error = DuplicateError(
                    evaluation=self,
                    pred_idx=pred_idx,
                    best_pred_idx=idx_of_best_pred_match,
                    gt_idx=idx_of_best_gt_match,
                    pred_label=pred_label,
                    pred_bbox=pred_bbox,
                    confidence=pred_conf,
                    best_pred_label=best_pred_label,
                    best_pred_bbox=best_pred_bbox,
                    best_confidence=best_confidence,
                    gt_label=label_of_best_gt_match,
                    gt_bbox=box_of_best_gt_match,
                    iou_threshold=self.fg_iou_threshold,
                    conf_threshold=self.conf_threshold,
                )
                self.errors.assign_prediction_error(duplicate_error)
                self.confusions.assign_prediction_confusion(
                    pred_idx, Confusion.FALSE_POSITIVE
                )
                continue

            # Error canditates that are not actually errors are true positives
            true_positive = NonError(
                evaluation=self,
                pred_idx=pred_idx,
                gt_idx=idx_of_best_gt_match,
                pred_label=pred_label,
                pred_bbox=pred_bbox,
                confidence=pred_conf,
                gt_label=label_of_best_gt_match,
                gt_bbox=gt_bboxes[idx_of_best_gt_match],
                iou_threshold=self.fg_iou_threshold,
                conf_threshold=self.conf_threshold,
            )
            self.errors.assign_prediction_error(true_positive)
            self.confusions.assign_prediction_confusion(
                pred_idx, Confusion.TRUE_POSITIVE
            )
            self.confusions.assign_gt_confusion(
                idx_of_best_gt_match, Confusion.TRUE_POSITIVE
            )

        # Consolidate remaining GTs as MissedErrors and false negatives
        for gt_idx in range(len(gt_bboxes)):
            # If the gt has been matched to an error, don't count it
            # since it's already been accounted for above.
            gt_confusion = self.confusions.check_gt_confusion(gt_idx)
            if gt_confusion is None:
                self.confusions.assign_gt_confusion(gt_idx, Confusion.FALSE_NEGATIVE)

            if gt_confusion is Confusion.TRUE_POSITIVE:
                continue

            # This list tells us if this gt has prediction error matches, and hence
            # won't be a MissedError
            pred_idx_matches = [None] * len(pred_bboxes)

            for pred_idx in range(len(pred_bboxes)):
                error = self.errors.check_prediction_error(pred_idx)
                if error is None:
                    continue
                if isinstance(error, ClassificationError):
                    if error.gt_idx == gt_idx:
                        pred_idx_matches[pred_idx] = (pred_idx, error)
                elif isinstance(error, LocalizationError):
                    if error.gt_idx == gt_idx:
                        pred_idx_matches[pred_idx] = (pred_idx, error)
                elif isinstance(error, ClassificationAndLocalizationError):
                    if error.gt_idx == gt_idx:
                        pred_idx_matches[pred_idx] = (pred_idx, error)

            if all(i is None for i in pred_idx_matches):
                missed_error = MissedError(
                    evaluation=self,
                    gt_idx=gt_idx,
                    gt_label=gt_labels[gt_idx],
                    gt_bbox=gt_bboxes[gt_idx],
                )
                self.errors.assign_gt_error(missed_error)

        # Final checks
        self.confusions.assert_valid_confusions()

    @property
    def image_path(self) -> str:
        return self.idx

    def is_true_positive(
        self,
        pred_label: torch.Tensor,
        iou_of_best_gt_match: torch.Tensor,
        idx_of_best_gt_match: torch.Tensor,
        label_of_best_gt_match: torch.Tensor,
    ) -> bool:
        if not iou_of_best_gt_match >= self.fg_iou_threshold:
            return False

        if not pred_label == label_of_best_gt_match:
            return False

        if self.confusions.check_gt_confusion(idx_of_best_gt_match):
            return False

        # We want to claim the gt `idx_of_best_gt_match`
        # Filter out those with higher IoUs but wrong label
        competitor_ious = self.ious[self.used_pred_idxs, idx_of_best_gt_match]

        # If the competitor has a higher IoU but it's the wrong label, remove it
        higher_iou_competitors_mask = competitor_ious > iou_of_best_gt_match
        higher_iou_competitors_labels = self.pred_labels[self.used_pred_idxs][
            higher_iou_competitors_mask
        ]
        if any(higher_iou_competitors_labels == label_of_best_gt_match):
            return False
        return True

    def is_background_error(self, iou: torch.Tensor) -> bool:
        return sum(iou > self.bg_iou_threshold) == 0

    def is_localization_error(
        self,
        pred_label: torch.Tensor,
        iou_of_best_gt_match: torch.Tensor,
        label_of_best_gt_match: torch.Tensor,
    ) -> bool:
        return (
            self.bg_iou_threshold <= iou_of_best_gt_match < self.fg_iou_threshold
            and pred_label == label_of_best_gt_match
        )

    def is_classification_error(
        self,
        pred_label: torch.Tensor,
        iou_of_best_gt_match: torch.Tensor,
        label_of_best_gt_match: torch.Tensor,
    ) -> bool:
        return (
            iou_of_best_gt_match >= self.fg_iou_threshold
            and pred_label != label_of_best_gt_match
        )

    def is_classification_and_localization_error(
        self,
        pred_label: torch.Tensor,
        iou_of_best_gt_match: torch.Tensor,
        label_of_best_gt_match: torch.Tensor,
    ) -> bool:
        return (
            self.bg_iou_threshold <= iou_of_best_gt_match < self.fg_iou_threshold
            and pred_label != label_of_best_gt_match
        )

    def is_duplicate_error(
        self,
        pred_label: torch.Tensor,
        iou_of_best_gt_match: torch.Tensor,
        idx_of_best_gt_match: torch.Tensor,
        label_of_best_gt_match: torch.Tensor,
    ):

        iou_mask = torch.logical_and(
            self.used_mask, self.errors.get_tp_mask(idx_of_best_gt_match)
        )
        assigned_ious = self.ious[:, idx_of_best_gt_match][iou_mask]
        is_highest_iou = (
            assigned_ious.numel() == 0
            or iou_of_best_gt_match > self.ious[:, idx_of_best_gt_match][iou_mask].max()
        )
        return (
            iou_of_best_gt_match >= self.fg_iou_threshold
            and pred_label == label_of_best_gt_match
            and not is_highest_iou
        )

    @property
    def pred_errors(self) -> List[Error]:
        if self._pred_errors is None:
            self._pred_errors = self.errors.get_prediction_errors()
        return self._pred_errors

    @property
    def gt_errors(self) -> List[Error]:
        if self._gt_errors is None:
            self._gt_errors = self.errors.get_gt_errors()
        return self._gt_errors

    def _get_iou_color(self, value: float) -> str:
        if value < self.bg_iou_threshold:
            return "red"
        elif self.bg_iou_threshold <= value < self.fg_iou_threshold:
            return "yellow"
        else:
            return "green"

    def _get_conf_color(self, score: float) -> str:
        if score < self.conf_threshold:
            return "red"
        else:
            return "green"

    def __repr__(self):
        n_rows, n_cols = self.ious.size()
        repr = f"ObjectDetectionEvaluation(pred={n_rows}, gt={n_cols})\n"
        repr += f"{os.path.basename(self.image_path)}\n"
        repr += "".join(
            ["\t\t"] + [f"gt{str(c)}" + "\t" for c in range(n_cols)] + ["\n"]
        )
        repr += "=" * 8 * (n_cols + 5) + "\n"
        repr += "".join(
            ["\tcls\t"]
            + [str(int(self.gt_labels[c].item())) + "\t" for c in range(n_cols)]
            + ["cfu\terr\tscore\t\n"]
        )
        repr += "=" * 8 * (n_cols + 5) + "\n"
        for i in range(n_rows):
            row = f"pred{i}\t"
            row += f"{int(self.pred_labels[i].item())}\t"
            for j in range(n_cols):
                value = round(self.ious[i, j].item(), 2)
                if i in self.unused_pred_idxs:
                    color = "white"
                else:
                    color = self._get_iou_color(value)
                value = colored(str(value), color)
                row += value + "\t"
            if i in self.unused_pred_idxs:
                row += f"[{colored('UN', 'white')}]\t"
            elif (
                self.confusions.check_prediction_confusion(i) is Confusion.TRUE_POSITIVE
            ):
                row += f"[{colored('TP', 'green')}]\t"
            else:
                row += f"[{colored('FP', 'red')}]\t"

            error = self.errors.check_prediction_error(i)
            if error:
                row += f"[{error.code}]\t"
            else:
                row += " --- \t"

            score = round(self.pred_scores[i].item(), 2)
            score_color = self._get_conf_color(score)
            score = colored(str(score), score_color)
            row += score + "\t"
            repr += row + "\n"

        repr += "=" * 8 * (n_cols + 5) + "\n"
        repr += "\tcfu\t"
        for j in range(n_cols):
            if self.confusions.check_gt_confusion(j) is Confusion.FALSE_NEGATIVE:
                repr += f"[{colored('FN', 'red')}]\t"
            elif self.confusions.check_gt_confusion(j) is Confusion.TRUE_POSITIVE:
                repr += f"[{colored('TP', 'green')}]\t"
            else:
                repr += "\t"

        repr += "\n\terr\t"
        for j in range(n_cols):
            error = self.errors.check_gt_error(j)
            if isinstance(error, MissedError):
                repr += f"[{error.code}]\t"
            else:
                repr += " --- \t"

        repr += "\n"
        return repr

    def draw_image_errors(self, image: Optional[torch.Tensor] = None) -> torch.Tensor:
        if image is None:
            image = read_image(self.image_path)

        tp_idxs = [
            c
            for c in range(len(self.confusions.pred_confusions))
            if self.confusions.check_prediction_confusion(c) is Confusion.TRUE_POSITIVE
        ]

        image = draw_bounding_boxes(
            image,
            boxes=self.pred_bboxes[tp_idxs],
            labels=[
                f"p{int(i.item())} TruePositive" for i in self.pred_labels[tp_idxs]
            ],
            colors=ErrorColor.TP.hex,
            width=2,
        )

        if len(self.instances) > 0:
            for err in self.instances:
                if isinstance(err, BackgroundError):
                    boxes = err.pred_bbox.unsqueeze(0)
                    labels = [f"p{err.pred_idx} BackgroundError"]
                    colors = [ErrorColor.BKG.hex]
                elif isinstance(err, ClassificationError):
                    boxes = torch.stack([err.gt_bbox, err.pred_bbox])
                    labels = [
                        f"gt ({err.gt_label})",
                        f"p{err.pred_idx} ClassificationError"
                        f" ({err.gt_label}->{err.pred_label})",
                    ]
                    colors = [ErrorColor.WHITE.hex, ErrorColor.CLS.hex]
                elif isinstance(err, LocalizationError):
                    boxes = torch.stack([err.gt_bbox, err.pred_bbox])
                    labels = ["gt", f"p{err.pred_idx} LocalizationError"]
                    colors = [ErrorColor.WHITE.hex, ErrorColor.LOC.hex]
                elif isinstance(err, ClassificationAndLocalizationError):
                    boxes = torch.stack([err.gt_bbox, err.pred_bbox])
                    labels = [
                        f"gt ({err.gt_label})",
                        f"p{err.pred_idx} ClsLocError"
                        f" ({err.gt_label}->{err.pred_label})",
                    ]
                    colors = [ErrorColor.WHITE.hex, ErrorColor.CLL.hex]
                elif isinstance(err, DuplicateError):
                    boxes = torch.stack(
                        [err.gt_bbox, err.best_pred_bbox, err.pred_bbox]
                    )
                    labels = ["gt", "best", f"p{err.pred_idx} DuplicateError"]
                    colors = [
                        ErrorColor.WHITE.hex,
                        ErrorColor.BST.hex,
                        ErrorColor.DUP.hex,
                    ]
                elif isinstance(err, MissedError):
                    boxes = err.gt_bbox.unsqueeze(0)
                    labels = [f"g{err.gt_idx} MissedError"]
                    colors = [ErrorColor.MIS.hex]
                image = draw_bounding_boxes(
                    image,
                    boxes=boxes,
                    labels=labels,
                    colors=colors,
                )
        return image

    def save_image_errors(self) -> None:
        os.makedirs("errors", exist_ok=True)
        image = self.draw_image_errors()
        to_pil_image(image).save(
            os.path.join("errors", os.path.basename(self.image_path))
        )


class Evaluator:
    def __init__(
        self,
        evaluations: list[Evaluation],
        name: str = "Model",
        gt_ids_map: torch.Tensor = None,
        categories: dict[int, str] = None,
    ) -> None:
        self.created_datetime = datetime.now()
        self.updated_datetime = self.created_datetime
        self.evaluations = evaluations
        self.name = name
        self.categories = categories or {}
        self._gt_ids_map = gt_ids_map
        self._collate_evaluations()

    def _collate_evaluations(self):
        errorlist_dict = self._init_errorlist_dict()
        confusion_matrices = {error_type.code: {} for error_type in ERROR_TYPES}
        confusion_matrices[None] = {}

        for evaluation in self.evaluations:
            errors = evaluation.errors.pred_errors + evaluation.errors.gt_errors
            confusions = (
                evaluation.confusions.pred_confusions
                + evaluation.confusions.gt_confusions
            )
            for error, confusion in zip(errors, confusions):
                if error is not None:
                    error_code = error.code
                    self._add_error_to_errorlist_dict(error, errorlist_dict, evaluation)
                    confusion_pair = (error.gt_label or 0, error.pred_label or 0)
                else:
                    error_code = None
                    confusion_pair = (0, 0)

                confusion_matrix = confusion_matrices[error_code]
                if confusion_pair not in confusion_matrix:
                    confusion_matrix[confusion_pair] = [
                        0 for _ in range(len(Confusion))
                    ]
                confusion_matrix[confusion_pair][confusion.value] += 1

        confusions_list = []
        for error_type, v in confusion_matrices.items():
            for k, confusions in v.items():
                confusions_list.append([error_type or "NON", *k, *confusions])

        self._errorlist_dict = errorlist_dict
        self._confusion_matrices = confusion_matrices
        self._confusion_df = pd.DataFrame(
            confusions_list,
            columns=["error", "gt", "pred", *[c.code for c in Confusion]],
        )

    def _init_errorlist_dict(self) -> Dict[str, List[Error]]:
        return {error_type.__name__: [] for error_type in ERROR_TYPES}

    def _init_confusion_matrices(self) -> Dict[str, List[Error]]:
        mat = {error_type.__name__: {} for error_type in ERROR_TYPES}
        mat[None] = {}
        return mat

    def _add_error_to_errorlist_dict(
        self,
        error: Error,
        errorlist_dict: Dict[str, List[Error]],
        evaluation: Evaluation,
    ):
        errorlist_dict[error.__class__.__name__].append(error)

    @classmethod
    def from_coco(
        cls,
        predictions_file: str,
        targets_file: str,
        image_dir: str,
        conf_threshold: float = 0.5,
        name: str = "Model",
        obb: bool = False,
    ) -> Evaluator:
        if cls == ObjectDetectionEvaluator:
            evaluation_cls = ObjectDetectionEvaluation
            evaluator_cls = ObjectDetectionEvaluator
        else:
            raise NotImplementedError("Evaluator class not implemented")
        loader = COCOLoader(
            annotations_file=targets_file,
            predictions_file=predictions_file,
            image_dir=image_dir,
        )
        return loader.load(evaluation_cls, evaluator_cls, conf_threshold, name=name, obb=obb)

    @classmethod
    def from_dict(
        cls,
        dict_file: str,
        image_dir: str,
        conf_threshold: float = 0.5,
        name: str = "Model",
    ) -> Evaluator:
        if cls == ObjectDetectionEvaluator:
            evaluation_cls = ObjectDetectionEvaluation
            evaluator_cls = ObjectDetectionEvaluator
        else:
            raise NotImplementedError("Evaluator class not implemented")
        loader = DictLoader(dict_file, image_dir)
        return loader.load(evaluation_cls, evaluator_cls, conf_threshold, name=name)

    @classmethod
    def from_dicts(
        cls,
        targets_dict_file: str,
        predictions_dict_file: str,
        image_dir: str,
        conf_threshold: float,
        name: str = "Model",
    ) -> Evaluator:
        if cls == ObjectDetectionEvaluator:
            evaluation_cls = ObjectDetectionEvaluation
            evaluator_cls = ObjectDetectionEvaluator
        else:
            raise NotImplementedError("Evaluator class not implemented")
        loader = DictLoader.from_dict_files(
            targets_dict_file, predictions_dict_file, image_dir
        )
        return loader.load(evaluation_cls, evaluator_cls, conf_threshold, name=name)

    @property
    def conf_threshold(self) -> float:
        return self.evaluations[0].conf_threshold

    def summarize(self) -> Dict[str, any]:
        raise NotImplementedError()

    def classwise_summarize(self) -> Dict[str, any]:
        raise NotImplementedError()

    def get_errorlist_dict(self) -> Dict[str, List[Error]]:
        """Get a mapping of error type to list of errors.

        Returns
        -------
        Dict[str, List[Error]]
            Dictionary of lists of errors.
        """
        if self._errorlist_dict is None:
            self._collate_evaluations()
        return self._errorlist_dict

    def get_confusions(
        self, by: str = None, key: str = None
    ) -> Union[Dict[Any, Dict[tuple, list]], pd.DataFrame, pd.Series]:
        if self._confusion_matrices is None:
            self._collate_evaluations()
        if by is None:
            return self._confusion_matrices
        elif by == "all":
            return self._confusion_df[["FN", "FP", "TP", "UN"]].sum(axis=0)

        assert by in [
            "error",
            "gt",
            "pred",
        ], "by must be one of 'error', 'gt', or 'pred'"
        assert (
            key in self._confusion_df[by].unique()
        ), f"For by={by}, key must be one of {self._confusion_df[by].unique()}"

        return self._confusion_df[self._confusion_df[by] == key][
            ["FN", "FP", "TP", "UN"]
        ].sum(axis=0)


class ObjectDetectionEvaluator(Evaluator):
    """Object Detection Evaluator class.

    Creates an Evaluator from:
        (1) Regular instantiation: a list of ObjectDetectionEvaluations,
        (2) ``from_coco()``: from COCO predictions and targets JSON files, or
        (3) ``from_dict()`` and ``from_dicts()``: using from a single or separate
            dictionary of predictions and targets.
    """

    evaluations: List[ObjectDetectionEvaluation]

    def __init__(
        self,
        evaluations: List[ObjectDetectionEvaluation],
        image_dir: str = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(evaluations, **kwargs)
        self.image_dir = image_dir
        self.num_images = len(evaluations)
        self.num_errors = sum([len(e.instances) for e in evaluations])

        self._crops: List[torch.Tensor] = None
        self._gt_errors: Dict[int, List[Error]] = None

        self.gt_data: GTData = None

    @property
    def iou_thresholds(self) -> Tuple[float, float]:
        return (
            self.evaluations[0].bg_iou_threshold,
            self.evaluations[0].fg_iou_threshold,
        )

    def _initialize_error_dict(self) -> Dict[str, int]:
        return {
            "ClassificationError": 0,
            "LocalizationError": 0,
            "ClassificationAndLocalizationError": 0,
            "DuplicateError": 0,
            "MissedError": 0,
            "BackgroundError": 0,
        }

    def summarize(self) -> Dict[str, Union[int, float]]:
        summary = self._initialize_error_dict()
        tp, fp, fn, un = 0, 0, 0, 0

        for evaluation in self.evaluations:
            for error in evaluation.instances:
                error_name = error.__class__.__name__
                summary[error_name] += 1
            tp += evaluation.confusions.get_true_positives()
            fp += evaluation.confusions.get_false_positives()
            fn += evaluation.confusions.get_false_negatives()
            un += evaluation.confusions.get_unused()
        precision = self.get_precision(tp, fp)
        recall = self.get_recall(tp, fn)
        f1 = self.get_f1(precision, recall)

        summary.update(
            {
                "true_positives": tp,
                "false_positives": fp,
                "false_negatives": fn,
                "unused": un,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "total_count": tp + fn,
            }
        )
        return summary

    def classwise_summarize(self) -> Dict[int, Dict[str, Union[int, float]]]:
        classwise_summary: Dict[int, dict] = dict()
        for evaluation in self.evaluations:
            for error in evaluation.instances:
                error_name = error.__class__.__name__
                if error.gt_label is not None:
                    category = error.gt_label
                elif error.pred_label is not None:
                    category = error.pred_label
                if category not in classwise_summary:
                    classwise_summary[category] = self._initialize_error_dict()
                classwise_summary[category][error_name] += 1

            # Get FN from gt confusions
            for idx, category in enumerate(evaluation.gt_labels.tolist()):
                if category not in classwise_summary:
                    classwise_summary[category] = self._initialize_error_dict()
                assert (
                    evaluation.confusions.is_complete
                ), evaluation.confusions.gt_confusions
                if evaluation.confusions.gt_confusions[idx] is Confusion.FALSE_NEGATIVE:
                    if "false_negatives" not in classwise_summary[category]:
                        classwise_summary[category]["false_negatives"] = 1
                    else:
                        classwise_summary[category]["false_negatives"] += 1

            # Get TP/FP from prediction confusions
            for idx, category in enumerate(evaluation.pred_labels.tolist()):
                if category not in classwise_summary:
                    classwise_summary[category] = self._initialize_error_dict()
                if (
                    evaluation.confusions.pred_confusions[idx]
                    is Confusion.FALSE_POSITIVE
                ):
                    if "false_positives" not in classwise_summary[category]:
                        classwise_summary[category]["false_positives"] = 1
                    else:
                        classwise_summary[category]["false_positives"] += 1
                elif (
                    evaluation.confusions.pred_confusions[idx]
                    is Confusion.TRUE_POSITIVE
                ):
                    if "true_positives" not in classwise_summary[category]:
                        classwise_summary[category]["true_positives"] = 1
                    else:
                        classwise_summary[category]["true_positives"] += 1

        # Calculate P, R, F1
        for category, summary in classwise_summary.items():
            tp = summary.get("true_positives", 0)
            fn = summary.get("false_negatives", 0)
            fp = summary.get("false_positives", 0)

            precision = self.get_precision(tp, fp)
            recall = self.get_recall(tp, fn)
            f1 = self.get_f1(precision, recall)
            summary["precision"] = precision
            summary["recall"] = recall
            summary["f1"] = f1
            summary["total_count"] = tp + fn

        return classwise_summary

    def _init_errorlist_dict(self) -> Dict[str, Dict[str, List[Error]]]:
        return {error_type.__name__: dict() for error_type in ERROR_TYPES}

    def _add_error_to_errorlist_dict(
        self,
        error: Error,
        errorlist_dict: Dict[str, Dict[str, List[Error]]],
        evaluation: ObjectDetectionEvaluation,
    ):
        error_name = error.__class__.__name__
        if evaluation.image_path not in errorlist_dict[error_name]:
            errorlist_dict[error_name][evaluation.image_path] = []
        errorlist_dict[error_name][evaluation.image_path].append(error)

    @logger()
    def get_errorlist_dict(self) -> Dict[str, Dict[str, List[Error]]]:
        """Get a nested mapping from error type to lists of errors.
        The first key is the error type, the second key is the image path, and the value is a list of errors.
        Returns
        -------
        Dict[str, Dict[str, List[Error]]]
            Dictionary of dictionaries of lists of errors.
        """
        return super().get_errorlist_dict()

    def get_true_positives(self) -> int:
        return self.get_confusions(by="error", key="NON")["TP"]

    def get_false_positives(self) -> int:
        return self.get_confusions(by="all")["FP"]

    def get_false_negatives(self) -> int:
        return self.get_confusions(by="all")["FN"]

    def get_unused(self) -> int:
        return self.get_confusions(by="all")["UN"]

    def get_precision(self, tp: int, fp: int) -> float:
        return tp / (tp + fp + 1e-7)

    def get_recall(self, tp: int, fn: int) -> float:
        return tp / (tp + fn + 1e-7)

    def get_f1(self, precision: float, recall: float):
        return (2 * precision * recall) / (precision + recall + 1e-7)

    def crop_objects(
        self,
        pad: int = 0,
        axis: int = 1,
        by_type: Type[Error] | List[Type[Error]] = None,
    ) -> Tuple[List[torch.Tensor], List[Error]]:
        images: List[torch.Tensor] = [None] * len(self.evaluations)
        num_bboxes: torch.Tensor = torch.zeros(len(self.evaluations)).long()
        bboxes: List[torch.Tensor] = []
        errors: List[List[Error]] = [None] * len(self.evaluations)

        attr_prefix = "gt" if axis == 0 else "pred"
        errors_attr = f"{attr_prefix}_errors"
        bboxes_attr = f"{attr_prefix}_bboxes"

        if by_type is None:

            def is_type(error: Error):
                return True

        elif isinstance(by_type, Type):

            def is_type(error: Error):
                return isinstance(error, by_type)

        else:
            assert isinstance(by_type, (list, tuple)), "by_type must be a list or tuple"
            assert all(
                isinstance(t, Type) for t in by_type
            ), "by_type must be a list of types"

            def is_type(error: Error):
                return error.__class__ in by_type

        for i, e in enumerate(self.evaluations):
            errors[i] = getattr(e.errors, errors_attr)
            img_bboxes: torch.Tensor = getattr(e, bboxes_attr)

            num_bboxes[i] = img_bboxes.shape[0]
            bboxes.append(img_bboxes)

            images[i] = read_image(e.image_path)

        combined_errors = [
            error for error_list in errors for error in error_list if is_type(error)
        ]

        combined_bboxes = torch.concat(bboxes, dim=0).long()
        combined_bboxes[:, 2:] = (
            combined_bboxes[:, 2:] - combined_bboxes[:, :2] + 2 * pad
        )
        combined_bboxes[:, :2] = combined_bboxes[:, :2] - pad

        crops: List[torch.Tensor] = [None] * combined_bboxes.shape[0]
        idx = 0
        for i, image in enumerate(images):
            for _ in range(num_bboxes[i]):
                crops[idx] = crop(
                    image,
                    combined_bboxes[idx, 1],
                    combined_bboxes[idx, 0],
                    combined_bboxes[idx, 3],
                    combined_bboxes[idx, 2],
                )
                idx += 1

        if len(crops) != len(combined_errors):
            i = 0
            masked_crops = []
            for error_list in errors:
                for error in error_list:
                    if is_type(error):
                        masked_crops.append(crops[i])
                    i += 1
            crops = masked_crops

        return crops, combined_errors

    def get_gt_data(self, pad: int = 0) -> GTData:
        if self.gt_data is not None:
            return self.gt_data
        gt_errors: Dict[int, List[Error]] = {}
        gt_labels = []
        crops: List[torch.Tensor] = []
        images: list = []
        for evaluation in self.evaluations:
            image = read_image(evaluation.image_path)
            bboxes = evaluation.gt_bboxes.long().clone()
            gt_labels.append(evaluation.gt_labels)
            bboxes[:, 2:] = (
                evaluation.gt_bboxes[:, 2:] - evaluation.gt_bboxes[:, :2] + 2 * pad
            )
            bboxes[:, :2] = evaluation.gt_bboxes[:, :2] - pad
            for bbox in bboxes:
                crops.append(crop(image, bbox[1], bbox[0], bbox[3], bbox[2]))
                images.append(evaluation.image_path)

            gt_errors.update(evaluation.get_errors_by_gt())

        gt_labels = torch.cat(gt_labels, dim=0)

        self.gt_data = GTData(
            crops=crops,
            gt_labels=gt_labels,
            gt_errors=gt_errors,
            images=images,
        )
        return self.gt_data

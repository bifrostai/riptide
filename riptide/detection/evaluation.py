from __future__ import annotations

import os
from datetime import datetime
from typing import Dict, List, Optional, Union
from warnings import warn

import torch
from termcolor import colored
from torchvision.io import read_image
from torchvision.ops.boxes import box_iou
from torchvision.transforms.functional import to_pil_image
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
)
from riptide.io.loaders import COCOLoader, DictLoader
import ujson as json


class ObjectDetectionEvaluation:
    """An object that creates and stores Errors and Confusions for a particular image.
    """

    def __init__(
        self,
        image_path: str,
        pred_bboxes: torch.Tensor,
        pred_scores: torch.Tensor,
        pred_labels: torch.Tensor,
        gt_bboxes: torch.Tensor,
        gt_labels: torch.Tensor,
        bg_iou_threshold: float = 0.1,
        fg_iou_threshold: float = 0.5,
        conf_threshold: float = 0.5,
    ) -> None:
        self.image_path = image_path

        self.bg_iou_threshold = bg_iou_threshold
        self.fg_iou_threshold = fg_iou_threshold
        self.conf_threshold = conf_threshold

        used_mask = pred_scores >= conf_threshold
        unused_mask = ~used_mask

        pred_idxs = torch.arange(len(pred_bboxes))
        self.used_pred_idxs = pred_idxs[used_mask]
        self.unused_pred_idxs = pred_idxs[unused_mask]

        self.pred_bboxes = pred_bboxes
        self.pred_scores = pred_scores
        self.pred_labels = pred_labels
        self.gt_bboxes = gt_bboxes
        self.gt_labels = gt_labels

        self.ious = box_iou(pred_bboxes, gt_bboxes)

        self.errors = Errors(
            num_preds=len(pred_bboxes),
            num_gt=len(gt_bboxes),
        )

        self.confusions = Confusions(
            evaluation=self,
            num_preds=len(pred_bboxes),
            num_gt=len(gt_bboxes),
        )

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
                    gt_idx=gt_idx,
                    gt_label=gt_label,
                    gt_bbox=gt_bbox,
                )
                self.errors.assign_gt_error(missed_error)
                self.confusions.assign_gt_confusion(gt_idx, Confusion.FALSE_NEGATIVE)
            return

        # First register all true positives
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
                    if error.gt_idx == gt_idx and error.gt_label != gt_labels[gt_idx]:
                        pred_idx_matches[pred_idx] = (pred_idx, error)

            if all(i is None for i in pred_idx_matches):
                missed_error = MissedError(
                    gt_idx=gt_idx,
                    gt_label=gt_labels[gt_idx],
                    gt_bbox=gt_bboxes[gt_idx],
                )
                self.errors.assign_gt_error(missed_error)

        # Final checks
        self.confusions.assert_valid_confusions()

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
        return (
            iou_of_best_gt_match >= self.fg_iou_threshold
            and pred_label == label_of_best_gt_match
            and iou_of_best_gt_match
            <= self.ious[:, idx_of_best_gt_match][self.used_pred_idxs].max()
        )

    @property
    def instances(self) -> List[Error]:
        pred_errors = [error for error in self.errors.get_prediction_errors() if error]
        gt_errors = [error for error in self.errors.get_gt_errors() if error]
        return pred_errors + gt_errors

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
                f"p({int(i.item())}) TruePositive" for i in self.pred_labels[tp_idxs]
            ],
            colors="lime",
            width=2,
        )

        if len(self.instances) > 0:
            for err in self.instances:
                if isinstance(err, BackgroundError):
                    boxes = err.pred_bbox.unsqueeze(0)
                    labels = [f"p{err.pred_idx} BackgroundError"]
                    colors = "magenta"
                elif isinstance(err, ClassificationError):
                    boxes = torch.stack([err.gt_bbox, err.pred_bbox])
                    labels = [
                        f"gt ({err.gt_label})",
                        (
                            f"p{err.pred_idx} ClassificationError"
                            f" ({err.gt_label}->{err.pred_label})"
                        ),
                    ]
                    colors = ["white", "crimson"]
                elif isinstance(err, LocalizationError):
                    boxes = torch.stack([err.gt_bbox, err.pred_bbox])
                    labels = ["gt", f"p{err.pred_idx} LocalizationError"]
                    colors = ["white", "gold"]
                elif isinstance(err, ClassificationAndLocalizationError):
                    boxes = torch.stack([err.gt_bbox, err.pred_bbox])
                    labels = [
                        f"gt ({err.gt_label})",
                        (
                            f"p{err.pred_idx} ClsLocError"
                            f" ({err.gt_label}->{err.pred_label})"
                        ),
                    ]
                    colors = ["white", "darkorange"]
                elif isinstance(err, DuplicateError):
                    boxes = torch.stack(
                        [err.gt_bbox, err.best_pred_bbox, err.pred_bbox]
                    )
                    labels = ["gt", "best", f"p{err.pred_idx} DuplicateError"]
                    colors = ["white", "blue", "cyan"]
                elif isinstance(err, MissedError):
                    boxes = err.gt_bbox.unsqueeze(0)
                    labels = [f"g{err.gt_idx} MissedError"]
                    colors = ["yellowgreen"]
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
    @classmethod
    def from_coco(
        cls, predictions_file: str, targets_file: str, conf_threshold: float = 0.5
    ) -> Evaluator:
        if cls == ObjectDetectionEvaluator:
            evaluation_cls = ObjectDetectionEvaluation
            evaluator_cls = ObjectDetectionEvaluator
        else:
            raise NotImplementedError("Evaluator class not implemented")
        loader = COCOLoader(predictions_file, targets_file)
        return loader.load(evaluation_cls, evaluator_cls, conf_threshold)

    @classmethod
    def from_dict(
        cls, dict_file: str, image_dir: str, conf_threshold: float = 0.5
    ) -> Evaluator:
        if cls == ObjectDetectionEvaluator:
            evaluation_cls = ObjectDetectionEvaluation
            evaluator_cls = ObjectDetectionEvaluator
        else:
            raise NotImplementedError("Evaluator class not implemented")
        loader = DictLoader(dict_file, image_dir)
        return loader.load(evaluation_cls, evaluator_cls, conf_threshold)

    @classmethod
    def from_dicts(
        cls,
        targets_dict_file: str,
        predictions_dict_file: str,
        image_dir: str,
        conf_threshold: float,
    ) -> Evaluator:
        if cls == ObjectDetectionEvaluator:
            evaluation_cls = ObjectDetectionEvaluation
            evaluator_cls = ObjectDetectionEvaluator
        else:
            raise NotImplementedError("Evaluator class not implemented")
        loader = DictLoader.from_dict_files(
            targets_dict_file, predictions_dict_file, image_dir
        )
        return loader.load(evaluation_cls, evaluator_cls, conf_threshold)


class ObjectDetectionEvaluator(Evaluator):
    """Object Detection Evaluator class.

    Creates an Evaluator from:
        (1) Regular instantiation: a list of ObjectDetectionEvaluations,
        (2) ``from_coco()``: from COCO predictions and targets JSON files, or
        (3) ``from_dict()`` and ``from_dicts()``: using from a single or separate
            dictionary of predictions and targets.
    """

    def __init__(
        self,
        evaluations: List[ObjectDetectionEvaluation],
    ) -> None:
        self.created_datetime = datetime.now()
        self.updated_datetime = self.created_datetime
        self.evaluations = evaluations
        self.num_images = len(evaluations)
        self.num_errors = sum([len(e.instances) for e in evaluations])

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
        tp, fp, fn = 0, 0, 0

        for evaluation in self.evaluations:
            for error in evaluation.instances:
                error_name = error.__class__.__name__
                summary[error_name] += 1
            tp += evaluation.confusions.get_true_positives()
            fp += evaluation.confusions.get_false_positives()
            fn += evaluation.confusions.get_false_negatives()
        precision = self.get_precision(tp, fp)
        recall = self.get_recall(tp, fn)
        f1 = self.get_f1(precision, recall)

        summary["true_positives"] = tp
        summary["false_positives"] = fp
        summary["false_negatives"] = fn
        summary["precision"] = precision
        summary["recall"] = recall
        summary["f1"] = f1
        summary["total_count"] = tp + fn
        return summary

    def classwise_summarize(self) -> Dict[int, Dict[str, Union[int, float]]]:
        classwise_summary = dict()
        for evaluation in self.evaluations:
            for error in evaluation.instances:
                error_name = error.__class__.__name__
                if hasattr(error, "gt_label"):
                    category = getattr(error, "gt_label").item()
                elif hasattr(error, "pred_label"):
                    category = getattr(error, "pred_label").item()
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
            tp, fn, fp = 0, 0, 0
            if "true_positives" in summary:
                tp = summary["true_positives"]
            if "false_negatives" in summary:
                fn = summary["false_negatives"]
            if "false_positives" in summary:
                fp = summary["false_positives"]
            precision = self.get_precision(tp, fp)
            recall = self.get_recall(tp, fn)
            f1 = self.get_f1(precision, recall)
            summary["precision"] = precision
            summary["recall"] = recall
            summary["f1"] = f1
            summary["total_count"] = tp + fn

        return classwise_summary

    def get_true_positives(self) -> int:
        tp = 0
        for evaluation in self.evaluations:
            tp += evaluation.confusions.get_true_positives()
        return tp

    def get_false_positives(self) -> int:
        fp = 0
        for evaluation in self.evaluations:
            fp += evaluation.confusions.get_false_positives()
        return fp

    def get_false_negatives(self) -> int:
        fn = 0
        for evaluation in self.evaluations:
            fn += evaluation.confusions.get_false_negatives()
        return fn

    def get_precision(self, tp: int, fp: int) -> float:
        return tp / (tp + fp + 1e-7)

    def get_recall(self, tp: int, fn: int) -> float:
        return tp / (tp + fn + 1e-7)

    def get_f1(self, precision: float, recall: float):
        return (2 * precision * recall) / (precision + recall + 1e-7)

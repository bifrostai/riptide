import numpy as np
import torch

from riptide.detection.evaluation import ObjectDetectionEvaluator


def get_centroids(boxes: torch.Tensor) -> torch.Tensor:
    cx = boxes[:, 0] + boxes[:, 2] / 2
    cy = boxes[:, 1] + boxes[:, 3] / 2
    return torch.stack([cx, cy], dim=1)


def get_areas(boxes: torch.Tensor) -> torch.Tensor:
    return (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])


def distance_size_score(
    gt_boxes: torch.Tensor, all_boxes: torch.Tensor
) -> torch.Tensor:
    gt_centroids = get_centroids(gt_boxes)
    all_centroids = get_centroids(all_boxes)
    distances = torch.cdist(gt_centroids, all_centroids)
    areas = get_areas(all_boxes)
    score = torch.sqrt(areas) * (8 / distances)
    score[score == float("Inf")] = 0
    score = torch.sum(torch.triu(score))
    return score


def sort_by_crowding(evaluator: ObjectDetectionEvaluator) -> dict:
    scores = []
    for evaluation in evaluator.evaluations:
        missed_gt_idxs = [
            err.gt_idx for err in evaluation.errors.get_gt_errors() if err.code == "MIS"
        ]
        if len(missed_gt_idxs) == 0:
            continue
        scores.append(
            distance_size_score(
                evaluation.gt_bboxes[missed_gt_idxs],
                evaluation.gt_bboxes,
            )
        )

    sorted_scores_indices = np.argsort(scores)
    sorted_scores_indices_descending = sorted_scores_indices[::-1]
    evals = np.array(evaluator.evaluations)
    return evals[sorted_scores_indices_descending], np.sort(scores)


def compute_aspect_variance(evaluator: ObjectDetectionEvaluator) -> dict:
    """Compute object aspect variance for MissedErrors."""
    aspect_ratios = {}
    for evaluation in evaluator.evaluations:
        for error in evaluation.instances:
            error_name = error.__class__.__name__
            if error_name == "MissedError":
                w = error.gt_bbox[2] - error.gt_bbox[0]
                h = error.gt_bbox[3] - error.gt_bbox[1]
                class_idx = int(error.gt_label.item())
                if class_idx not in aspect_ratios:
                    aspect_ratios[class_idx] = []
                aspect_ratios[class_idx].append(w / h)
    for class_idx, aspect_ratio_list in aspect_ratios.items():
        aspect_ratio_list = [_.item() for _ in aspect_ratio_list]
        aspect_ratios[class_idx] = round(np.var(aspect_ratio_list), 2)
    return aspect_ratios


def compute_size_variance(evaluator: ObjectDetectionEvaluator) -> dict:
    """Compute object size variance for MissedErrors."""
    areas = {}
    for evaluation in evaluator.evaluations:
        for error in evaluation.instances:
            error_name = error.__class__.__name__
            if error_name == "MissedError":
                area = (error.gt_bbox[2] - error.gt_bbox[0]) * (
                    error.gt_bbox[3] - error.gt_bbox[1]
                )
                class_idx = int(error.gt_label.item())
                if class_idx not in areas:
                    areas[class_idx] = []
                areas[class_idx].append(area)
    for class_idx, area_list in areas.items():
        area_list = [_.item() for _ in area_list]
        areas[class_idx] = f"{np.var(area_list):.2E}"
    return areas

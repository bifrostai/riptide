# import os

# import tidecv
import torch
from torchvision.ops import box_iou

from riptide.detection.errors import (
    BackgroundError,
    ClassificationAndLocalizationError,
    ClassificationError,
    DuplicateError,
    LocalizationError,
    MissedError,
)
from riptide.detection.evaluation import (
    ObjectDetectionEvaluation,
    ObjectDetectionEvaluator,
)


def test_background_error():
    """A background error is defined as having the following conditions:

    1. valid prediction (score > score threshold)
    2. iou <= background threshold
    """
    pred_bboxes = torch.Tensor(
        [
            [9, 9, 11, 11],
        ]
    )
    gt_bboxes = torch.Tensor([[0, 0, 10, 10]])
    pred_scores = torch.Tensor([0.5])
    pred_labels = torch.Tensor([1.0])
    gt_labels = torch.Tensor([1.0])

    assert pred_scores >= 0.5
    assert box_iou(pred_bboxes, gt_bboxes) <= 0.1
    evaluation = ObjectDetectionEvaluation(
        image_path="",
        pred_bboxes=pred_bboxes,
        pred_scores=pred_scores,
        pred_labels=pred_labels,
        gt_bboxes=gt_bboxes,
        gt_labels=gt_labels,
    )
    assert sum(isinstance(e, BackgroundError) for e in evaluation.instances) == 1


def test_classification_error():
    """A classification error is defined as having the following conditions:

    1. valid prediction (score > score threshold)
    2. iou >= foreground threshold
    3. pred_label != gt_label
    """
    pred_bboxes = torch.Tensor(
        [
            [1, 1, 11, 11],
        ]
    )
    gt_bboxes = torch.Tensor([[0, 0, 10, 10]])
    pred_scores = torch.Tensor([0.5])
    pred_labels = torch.Tensor([2.0])
    gt_labels = torch.Tensor([1.0])

    assert pred_scores >= 0.5
    assert box_iou(pred_bboxes, gt_bboxes) >= 0.5
    assert pred_labels != gt_labels
    evaluation = ObjectDetectionEvaluation(
        image_path="",
        pred_bboxes=pred_bboxes,
        pred_scores=pred_scores,
        pred_labels=pred_labels,
        gt_bboxes=gt_bboxes,
        gt_labels=gt_labels,
    )
    assert sum(isinstance(e, ClassificationError) for e in evaluation.instances) == 1


def test_localization_error():
    """A localization error is defined as having the following conditions:

    1. valid prediction (score > score threshold)
    2. background threshold <= iou <= foreground threshold
    3. pred_label == gt_label
    """
    pred_bboxes = torch.Tensor(
        [
            [5, 5, 11, 11],
        ]
    )
    gt_bboxes = torch.Tensor([[0, 0, 10, 10]])
    pred_scores = torch.Tensor([0.5])
    pred_labels = torch.Tensor([1.0])
    gt_labels = torch.Tensor([1.0])

    assert pred_scores >= 0.5
    assert box_iou(pred_bboxes, gt_bboxes) >= 0.1
    assert box_iou(pred_bboxes, gt_bboxes) <= 0.5
    assert pred_labels == gt_labels
    evaluation = ObjectDetectionEvaluation(
        image_path="",
        pred_bboxes=pred_bboxes,
        pred_scores=pred_scores,
        pred_labels=pred_labels,
        gt_bboxes=gt_bboxes,
        gt_labels=gt_labels,
    )
    assert sum(isinstance(e, LocalizationError) for e in evaluation.instances) == 1


def test_classification_localization_error():
    """A classification + localization error is defined as having the following
    conditions:

    1. valid prediction (score > score threshold)
    2. background threshold <= iou <= foreground threshold
    3. pred_label != gt_label
    """
    pred_bboxes = torch.Tensor(
        [
            [5, 5, 11, 11],
        ]
    )
    gt_bboxes = torch.Tensor([[0, 0, 10, 10]])
    pred_scores = torch.Tensor([0.5])
    pred_labels = torch.Tensor([2.0])
    gt_labels = torch.Tensor([1.0])

    assert pred_scores >= 0.5
    assert box_iou(pred_bboxes, gt_bboxes) >= 0.1
    assert box_iou(pred_bboxes, gt_bboxes) <= 0.5
    assert pred_labels != gt_labels
    evaluation = ObjectDetectionEvaluation(
        image_path="",
        pred_bboxes=pred_bboxes,
        pred_scores=pred_scores,
        pred_labels=pred_labels,
        gt_bboxes=gt_bboxes,
        gt_labels=gt_labels,
    )
    assert (
        sum(
            isinstance(e, ClassificationAndLocalizationError)
            for e in evaluation.instances
        )
        == 1
    )


def test_duplicate_error():
    """A duplicate error is defined as having the following conditions:

    1. valid prediction (score > score threshold)
    2. iou >= foreground threshold
    3. pred_label == gt_label
    4. there is an existing prediction that has already matched gt
    """
    pred_bboxes = torch.Tensor(
        [
            [1, 1, 10, 10],
            [2, 2, 10, 10],
        ]
    )
    gt_bboxes = torch.Tensor([[0, 0, 10, 10]])
    pred_scores = torch.Tensor([0.5, 0.5])
    pred_labels = torch.Tensor([1.0, 1.0])
    gt_labels = torch.Tensor([1.0])

    assert all(pred_scores >= 0.5)
    assert all(box_iou(pred_bboxes, gt_bboxes) >= 0.5)
    assert all(pred_labels == gt_labels)
    evaluation = ObjectDetectionEvaluation(
        image_path="",
        pred_bboxes=pred_bboxes,
        pred_scores=pred_scores,
        pred_labels=pred_labels,
        gt_bboxes=gt_bboxes,
        gt_labels=gt_labels,
    )
    assert sum(isinstance(e, DuplicateError) for e in evaluation.instances) == 1


def test_missed_error():
    """A missed error is defined as having the following conditions:

    1. valid prediction (score > score threshold)
    2. iou < background threshold
    """
    pred_bboxes = torch.Tensor([[100, 100, 150, 150]])
    gt_bboxes = torch.Tensor([[0, 0, 10, 10]])
    pred_scores = torch.Tensor([0.5])
    pred_labels = torch.Tensor([1.0])
    gt_labels = torch.Tensor([1.0])

    assert all(pred_scores >= 0.5)
    assert all(box_iou(pred_bboxes, gt_bboxes) < 0.1)
    evaluation = ObjectDetectionEvaluation(
        image_path="",
        pred_bboxes=pred_bboxes,
        pred_scores=pred_scores,
        pred_labels=pred_labels,
        gt_bboxes=gt_bboxes,
        gt_labels=gt_labels,
    )
    assert sum(isinstance(e, MissedError) for e in evaluation.instances) == 1


def test_empty_predictions():
    """Test for empty predictions input to ObjectDetectionEvaluation."""
    pred_bboxes = torch.empty((0, 4))
    gt_bboxes = torch.Tensor([[0, 0, 10, 10]])
    pred_scores = torch.Tensor([])
    pred_labels = torch.Tensor([])
    gt_labels = torch.Tensor([1.0])

    assert len(pred_bboxes) == 0
    evaluation = ObjectDetectionEvaluation(
        image_path="",
        pred_bboxes=pred_bboxes,
        pred_scores=pred_scores,
        pred_labels=pred_labels,
        gt_bboxes=gt_bboxes,
        gt_labels=gt_labels,
    )
    assert sum(isinstance(e, MissedError) for e in evaluation.instances) == 1


def test_empty_predictions_targets():
    """Test for empty predictions and targets input to ObjectDetectionEvaluation."""
    pred_bboxes = torch.empty((0, 4))
    gt_bboxes = torch.empty((0, 4))
    pred_scores = torch.Tensor([])
    pred_labels = torch.Tensor([])
    gt_labels = torch.Tensor([])

    assert len(pred_bboxes) == 0
    evaluation = ObjectDetectionEvaluation(
        image_path="",
        pred_bboxes=pred_bboxes,
        pred_scores=pred_scores,
        pred_labels=pred_labels,
        gt_bboxes=gt_bboxes,
        gt_labels=gt_labels,
    )
    assert len(evaluation.instances) == 0


def test_confusion():
    pred_bboxes = torch.Tensor(
        [
            [0, 0, 10, 10],  # true positive
            [7, 7, 13, 13],  # true positive
            [1, 1, 11, 11],  # ClassificationError
            [1, 1, 11, 11],  # ClassificationError
            [5, 5, 11, 11],  # LocalizationError
            [5, 5, 11, 11],  # ClassificationAndLocalizationError
            [1, 1, 10, 10],  # DuplicateError
            [9, 9, 10, 10],  # BackgroundError
        ]
    )
    gt_bboxes = torch.Tensor(
        [
            [0, 0, 10, 10],
            [9, 9, 20, 20],  # MissedError
            [7, 7, 13, 13],
        ]
    )
    pred_scores = torch.Tensor([0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5])
    pred_labels = torch.Tensor([1.0, 1.0, 2.0, 2.0, 1.0, 2.0, 1.0, 1.0])
    gt_labels = torch.Tensor([1.0, 1.0, 1.0])

    evaluation = ObjectDetectionEvaluation(
        image_path="",
        pred_bboxes=pred_bboxes,
        pred_scores=pred_scores,
        pred_labels=pred_labels,
        gt_bboxes=gt_bboxes,
        gt_labels=gt_labels,
    )
    issue = ObjectDetectionEvaluator([evaluation])
    tp = issue.get_true_positives()
    fn = issue.get_false_negatives()
    fp = issue.get_false_positives()
    un = issue.get_unused()

    print(evaluation)
    assert tp == 2
    assert fp == 6
    assert fn == 1
    assert un == 0
    assert tp + fp == len(pred_labels)
    assert tp + fn == len(gt_labels)


def test_confusion_with_unused_predictions():
    pred_bboxes = torch.Tensor(
        [
            [0, 0, 10, 10],  # true positive
            [7, 7, 13, 13],  # true positive [LOW CONFIDENCE]
            [1, 1, 11, 11],  # ClassificationError
            [1, 1, 11, 11],  # ClassificationError
            [5, 5, 11, 11],  # LocalizationError
            [5, 5, 11, 11],  # ClassificationAndLocalizationError
            [1, 1, 10, 10],  # DuplicateError
            [9, 9, 10, 10],  # BackgroundError
        ]
    )
    gt_bboxes = torch.Tensor(
        [
            [0, 0, 10, 10],
            [9, 9, 20, 20],  # MissedError
            [7, 7, 13, 13],
        ]
    )
    pred_scores = torch.Tensor([0.5, 0.1, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5])
    pred_labels = torch.Tensor([1.0, 1.0, 2.0, 2.0, 1.0, 2.0, 1.0, 1.0])
    gt_labels = torch.Tensor([1.0, 1.0, 1.0])

    evaluation = ObjectDetectionEvaluation(
        image_path="",
        pred_bboxes=pred_bboxes,
        pred_scores=pred_scores,
        pred_labels=pred_labels,
        gt_bboxes=gt_bboxes,
        gt_labels=gt_labels,
    )
    issue = ObjectDetectionEvaluator([evaluation])
    tp = issue.get_true_positives()
    fn = issue.get_false_negatives()
    fp = issue.get_false_positives()
    un = issue.get_unused()

    print(evaluation)
    assert tp == 1
    assert fp == 6
    assert fn == 2
    assert un == 1
    assert tp + fp + un == len(pred_labels)
    assert tp + fn == len(gt_labels)


# def test_output_against_tidecv():
#     # TODO: Replace this with proper test cases. Right now we can use any arbitrary
#     # predictions and targets that have been saved after model inference.
#     preds_dict = torch.load(
#         "/home/ubuntu/benchmark-one/epoch=195-step=24499/predictions.pt"
#     )
#     targets_dict = torch.load(
#         "/home/ubuntu/benchmark-one/epoch=195-step=24499/targets.pt"
#     )

#     for idx, image_path in enumerate(preds_dict.keys()):
#         # Setup for issues module
#         all_evaluation = []

#         # Setup for tidecv
#         predictions = tidecv.Data("predictions")
#         targets = tidecv.Data("targets")

#         preds = preds_dict[image_path]
#         target = targets_dict[image_path]
#         preds_boxes = preds["boxes"].cpu()
#         preds_scores = preds["scores"].cpu()
#         preds_labels = preds["labels"].cpu()

#         target_boxes = target["boxes"].cpu()
#         target_labels = target["labels"].cpu()

#         evaluation = ObjectDetectionEvaluation(
#             image_path=os.path.join("test/PS-RGB_tiled", image_path),
#             pred_bboxes=preds_boxes,
#             pred_scores=preds_scores,
#             pred_labels=preds_labels,
#             gt_bboxes=target_boxes,
#             gt_labels=target_labels,
#         )
#         all_evaluation.append(evaluation)

#         # Convert boxes to numpy
#         preds_boxes_np = preds_boxes.cpu().numpy()
#         preds_scores_np = preds_scores.cpu().numpy()
#         preds_labels_np = preds_labels.cpu().numpy()
#         target_boxes_np = target_boxes.cpu().numpy()
#         target_labels_np = target_labels.cpu().numpy()

#         # Convert boxes to xywh for pycocotools
#         preds_boxes_np[:, 2] = preds_boxes_np[:, 2] - preds_boxes_np[:, 0]
#         preds_boxes_np[:, 3] = preds_boxes_np[:, 3] - preds_boxes_np[:, 1]
#         target_boxes_np[:, 2] = target_boxes_np[:, 2] - target_boxes_np[:, 0]
#         target_boxes_np[:, 3] = target_boxes_np[:, 3] - target_boxes_np[:, 1]

#         for box, score, label in zip(preds_boxes_np, preds_scores_np, preds_labels_np):
#             if score >= 0.5:
#                 predictions.add_detection(idx, label, score, box, mask=None)

#         for box, label in zip(target_boxes_np, target_labels_np):
#             targets.add_ground_truth(idx, label, box, mask=None)

#         tide = tidecv.TIDE()
#         tide_run = tide.evaluate(
#             targets,
#             predictions,
#             pos_threshold=0.5,
#             background_threshold=0.1,
#             mode="bbox",
#         )
#         tide_result = tide_run.count_errors()
#         issue = ObjectDetectionEvaluator(all_evaluation)
#         issues_result = issue.summarize()

#         if (
#             image_path
#             == "25_104001003ACA9500_tile_417.png"  # TIDE fails to assign CLL error when the CLL is also a LOC for other nearby boxes
#             or image_path
#             == "84_1040010006ABC200_tile_114.png"  # TIDE wrongly missed CLS error
#             or image_path
#             == "25_104001003ACA9500_tile_418.png"  # TIDE wrongly missed CLS error
#             or image_path
#             == "55_1040010049CD5600_tile_272.png"  # TIDE wrongly missed CLS error
#             or image_path
#             == "115_104001004B8E2D00_tile_183.png"  # TIDE wrongly missed CLS error
#             or image_path
#             == "115_104001004B8E2D00_tile_182.png"  # TIDE wrongly missed CLS error
#         ):
#             continue

#         print(evaluation)
#         if "BackgroundError" in issues_result:
#             assert (
#                 tide_result[tidecv.BackgroundError] == issues_result["BackgroundError"]
#             )

#         if "ClassificationError" in issues_result:
#             assert (
#                 tide_result[tidecv.ClassError] == issues_result["ClassificationError"]
#             )

#         if "LocalizationError" in issues_result:
#             assert tide_result[tidecv.BoxError] == issues_result["LocalizationError"]

#         if "ClassificationAndLocalizationError" in issues_result:
#             assert (
#                 tide_result[tidecv.OtherError]
#                 == issues_result["ClassificationAndLocalizationError"]
#             )

#         if "MissedError" in issues_result:
#             assert tide_result[tidecv.MissedError] == issues_result["MissedError"]

#         if "DuplicateError" in issues_result:
#             assert tide_result[tidecv.DuplicateError] == issues_result["DuplicateError"]

import torch
from hypothesis import given, settings
from hypothesis import strategies as st

from riptide.detection.errors import MissedError
from riptide.detection.evaluation import (
    ObjectDetectionEvaluation,
    ObjectDetectionEvaluator,
)
from riptide.detection.visualization import missed_groups as inspect_missed_groups
from tests.strategies.image import st_bbox, st_image_and_bboxes


@settings(deadline=None)
@given(st_bbox(max_value=1920), st.integers(min_value=0, max_value=16))
def test_missed_truncated(gt_bboxes: torch.Tensor, min_size: int):
    """Missed errors close to the image border should be classified as truncated"""
    torch.random.manual_seed(1234)
    h, w = 1920, 1920
    n = gt_bboxes.shape[0]

    left = gt_bboxes.clone()
    left[:, 0] = torch.min(torch.tensor([min_size]), gt_bboxes[:, 2])
    right = gt_bboxes.clone()
    right[:, 2] = torch.max(gt_bboxes[:, 0], torch.tensor([w - min_size]))
    top = gt_bboxes.clone()
    top[:, 1] = torch.min(torch.tensor([min_size]), gt_bboxes[:, 3])
    bottom = gt_bboxes.clone()
    bottom[:, 3] = torch.max(gt_bboxes[:, 1], torch.tensor([h - min_size]))

    gt_bboxes = torch.cat([left, right, top, bottom])
    gt_labels = torch.ones((gt_bboxes.shape[0],))

    evaluation = ObjectDetectionEvaluation(
        image_path="",
        pred_bboxes=torch.empty((0, 4)),
        pred_scores=torch.Tensor([]),
        pred_labels=torch.Tensor([]),
        gt_bboxes=gt_bboxes,
        gt_labels=gt_labels,
    )
    assert (
        sum(isinstance(e, MissedError) for e in evaluation.instances) == 4 * n
    ), "All errors should be missed errors"

    evaluator = ObjectDetectionEvaluator([evaluation])
    errors = evaluator.get_errorlist_dict()["MissedError"]

    missed_groups = inspect_missed_groups([errors])
    assert len(missed_groups[0]["truncated"]) == 4 * n, "All errors should be truncated"
    assert (
        len(missed_groups[0]["others"]) == 0
    ), "No errors should be classified as others"

from typing import Tuple

import torch
from hypothesis import given

from riptide.utils.image import convex_hull, crop_preview
from tests.strategies.image import st_bbox, st_image_and_bboxes


@given(st_bbox(dtype=int, xyxy=True))
def test_convex_hull_xyxy(bboxes: torch.Tensor):
    original = bboxes.clone()

    min_sides = bboxes[:, [0, 1]].min(dim=0)
    max_sides = bboxes[:, [2, 3]].max(dim=0)

    values, indices = convex_hull(bboxes)

    assert (values == torch.cat([min_sides.values, max_sides.values])).all()
    assert (indices == torch.cat([min_sides.indices, max_sides.indices])).all()
    assert (bboxes == original).all()


@given(st_bbox(dtype=int))
def test_convex_hull_xywh(bboxes: torch.Tensor):
    original = bboxes.clone()

    min_sides = bboxes[:, [0, 1]].min(dim=0)
    max_sides = (bboxes[:, [0, 1]] + bboxes[:, [2, 3]]).max(dim=0)
    wh_sides = max_sides.values - min_sides.values

    values, indices = convex_hull(bboxes, format="xywh")

    assert (values == torch.cat([min_sides.values, wh_sides])).all()
    assert (indices == torch.cat([min_sides.indices, max_sides.indices])).all()
    assert (bboxes == original).all()


@given(st_image_and_bboxes())
def test_padded_crop(input: Tuple[torch.Tensor, torch.Tensor]):
    PREVIEW_PADDING = 48
    image_tensor, bbox = input
    hull, _ = convex_hull(bbox)
    sides = hull[2:] - hull[:2]

    cropped = crop_preview(image_tensor, bbox, colors=None)

    assert (
        -2 < cropped.shape[1] - cropped.shape[2] < 2
    ), "Cropped image should be square"
    assert (
        -2 < cropped.shape[1] - (sides.max() + PREVIEW_PADDING * 2) < 2
    ), "Cropped image should be padded"
    # TODO: Check that the bounding boxes are in the right place

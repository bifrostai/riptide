from typing import List

import torch
from hypothesis import given
from hypothesis import strategies as st
from hypothesis.extra.numpy import array_shapes as st_array_shapes
from hypothesis.extra.numpy import arrays as st_arrays

from riptide.utils.image import convex_hull, crop_preview


@st.composite
def bbox_shape(draw) -> st.SearchStrategy[List[int]]:
    head = draw(st_array_shapes(min_dims=1, max_dims=1))
    return (*head, 4)


@st.composite
def image_shape(draw, channels=3) -> st.SearchStrategy[List[int]]:
    tail = draw(st_array_shapes(min_dims=2, max_dims=2, min_side=10))
    return (channels, *tail)


@st.composite
def st_tensor(
    draw, dtype=int, shape=bbox_shape(), shape_only=True
) -> st.SearchStrategy[torch.Tensor]:
    if shape_only:
        arr = draw(
            st_arrays(
                dtype=dtype,
                shape=shape,
                elements=st.integers(min_value=0, max_value=255),
            )
        )
    else:
        arr = draw(st_arrays(dtype=dtype, shape=shape))
    return torch.tensor(arr)


@st.composite
def st_bbox(draw, dtype=int, xyxy=False) -> st.SearchStrategy[torch.Tensor]:
    arr = draw(
        st_arrays(
            dtype=dtype,
            shape=bbox_shape(),
            elements=st.integers(min_value=0, max_value=2048),
        )
    )
    if xyxy:
        arr[:, 2:] += arr[:, :2]
    return torch.tensor(arr)


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


@given(st_tensor(dtype=int, shape=image_shape()), st_bbox(dtype=int, xyxy=True))
def test_padded_crop(image_tensor: torch.Tensor, bbox: torch.Tensor):
    PREVIEW_PADDING = 48
    hull, _ = convex_hull(bbox)
    sides = hull[2:] - hull[:2]

    cropped, translation = crop_preview(image_tensor, bbox, colors=None)

    # TODO: Fix this test
    # assert cropped.shape[1] == cropped.shape[2], "Cropped image should be square"
    # assert torch.abs(cropped.shape[1] - (sides.max() + PREVIEW_PADDING * 2)) <= 100, "Cropped image should be padded"

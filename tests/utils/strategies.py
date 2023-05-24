from typing import List, Tuple

import torch
from hypothesis import strategies as st
from hypothesis.extra.numpy import array_shapes as st_array_shapes
from hypothesis.extra.numpy import arrays as st_arrays


@st.composite
def bbox_shape(draw) -> st.SearchStrategy[List[int]]:
    head = draw(st_array_shapes(min_dims=1, max_dims=1))
    return (*head, 4)


@st.composite
def image_shape(
    draw, channels=3, min_side=10, max_side=4096
) -> st.SearchStrategy[List[int]]:
    tail = draw(
        st_array_shapes(min_dims=2, max_dims=2, min_side=min_side, max_side=max_side)
    )
    return (channels, *tail)


@st.composite
def st_bbox(
    draw, dtype=int, xyxy=False, min_side=0, max_value=2048
) -> st.SearchStrategy[torch.Tensor]:
    arr = draw(
        st_arrays(
            dtype=dtype,
            shape=bbox_shape(),
            elements=st.integers(min_value=0, max_value=max_value),
        )
    )
    arr[:, 2:] += min_side

    if xyxy:
        arr[:, 2:] += arr[:, :2]
    return torch.tensor(arr)


@st.composite
def st_image(
    draw, dtype=int, channels=3, shape=None
) -> st.SearchStrategy[torch.Tensor]:
    arr = draw(
        st_arrays(
            dtype=dtype,
            shape=shape or image_shape(channels=channels),
            elements=st.integers(min_value=0, max_value=255),
        )
    )
    return torch.tensor(arr)


@st.composite
def st_images(
    draw, num=10, dtype=int, channels=3
) -> st.SearchStrategy[List[torch.Tensor]]:
    images = [draw(st_image(dtype=dtype, channels=channels)) for _ in range(num)]
    return images


@st.composite
def st_image_and_bboxes(
    draw, dtype=int, channels=3
) -> st.SearchStrategy[List[torch.Tensor]]:
    img_shape = draw(image_shape(channels=channels))
    image = draw(st_image(dtype=dtype, shape=img_shape))
    bboxes = draw(st_bbox(dtype=dtype, xyxy=True, max_value=max(img_shape)))
    bboxes[:, [0, 2]] = torch.clamp(bboxes[:, [0, 2]], min=0, max=img_shape[-1])
    bboxes[:, [1, 3]] = torch.clamp(bboxes[:, [1, 3]], min=0, max=img_shape[-2])
    return image, bboxes

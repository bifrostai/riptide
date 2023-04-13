import base64
import io
from typing import Any, List, Tuple, Union

import matplotlib.pyplot as plt
import plotly.graph_objects as go
import torch
from matplotlib.figure import Figure
from PIL import Image
from torchvision.transforms.functional import crop
from torchvision.utils import draw_bounding_boxes

PREVIEW_PADDING = 48


def crop_preview(
    image_tensor: torch.Tensor,
    bbox: torch.Tensor,
    colors: Union[List[Union[str, Tuple[int, int, int]]], str, Tuple[int, int, int]],
) -> torch.Tensor:
    """Crop a preview of the bounding box

    Parameters
    ----------
    image_tensor : torch.Tensor
        Image tensor
    bbox : torch.Tensor
        Bounding box(es) in xyxy format
    color : str
        Color of bounding box

    Returns
    -------
    torch.Tensor
        Cropped image tensor
    """
    bbox = bbox.long()
    if colors is not None:
        image_tensor = draw_bounding_boxes(image_tensor, bbox, colors=colors, width=2)
    image_tensor, translation = get_padded_bbox_crop(image_tensor, bbox)

    return image_tensor


def convex_hull(
    bboxes: torch.Tensor, format: str = "xyxy"
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Get the convex hull of a set of bounding boxes

    Parameters
    ----------
    bboxes : torch.Tensor
        Bounding boxes
    format : str, optional. One of "xyxy" or "xywh"
        Format of bounding boxes, by default "xyxy"

    Returns
    -------
    Tuple[torch.Tensor, torch.Tensor]
        Convex hull and indices of bounding boxes that make up the convex hull
    """
    bboxes = bboxes.long().clone()
    if format == "xywh":
        bboxes[:, 2:] = bboxes[:, :2] + bboxes[:, 2:]
    t_max: torch.Tensor = torch.max(bboxes, dim=0)
    t_min: torch.Tensor = torch.min(bboxes, dim=0)
    values = torch.concat([t_min.values[:2], t_max.values[2:]])
    indices = torch.concat([t_min.indices[:2], t_max.indices[2:]])
    if format == "xywh":
        values[2:] = values[2:] - values[:2]
    return values, indices


def get_padded_bbox_crop(
    image_tensor: torch.Tensor,
    bbox: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Get a padded crop of the bounding box(es)

    Parameters
    ----------
    image_tensor : torch.Tensor
        Image tensor
    bbox : torch.Tensor
        Bounding box(es) in xyxy format
    pad : int, optional
        Padding around bounding box, by default 48
    size : int, optional
        Size of output image, by default 224

    Returns
    -------
    Tuple[torch.Tensor, torch.Tensor]
        Cropped image and translation tensor to apply to bounding boxes
    """
    hull, _ = convex_hull(bbox)
    x1, y1, x2, y2 = hull
    long_edge = torch.argmax(hull[2:] - hull[:2])  # 0 is w, 1 is h
    if long_edge == 0:
        x1 -= PREVIEW_PADDING
        x2 += PREVIEW_PADDING
        short_edge_padding = torch.div((x2 - x1) - (y2 - y1), 2, rounding_mode="floor")
        y1 = max(0, y1 - short_edge_padding)
        y2 = min(image_tensor.size(1), y2 + short_edge_padding)
    else:
        y1 -= PREVIEW_PADDING
        y2 += PREVIEW_PADDING
        short_edge_padding = torch.div(
            ((y2 - y1) - (x2 - x1)), 2, rounding_mode="floor"
        )
        x1 = max(0, x1 - short_edge_padding)
        x2 = min(image_tensor.size(2), x2 + short_edge_padding)

    return (
        crop(image_tensor, y1, x1, y2 - y1, x2 - x1),
        torch.tensor([x1, y1, x2, y2]) - hull,
    )


def get_bbox_stats(bbox: torch.Tensor) -> Tuple[int, int, int]:
    """Get the width, height, and area of a bounding box

    Parameters
    ----------
    bbox : torch.Tensor
        Bounding box in xyxy format

    Returns
    -------
    Tuple[int, int, int]
        Width, height, and area of bounding box
    """
    area = round(((bbox[2] - bbox[0]) * (bbox[3] - bbox[1])).item(), 2)
    width = int((bbox[2] - bbox[0]).item())
    height = int((bbox[3] - bbox[1]).item())
    return width, height, area


def encode_base64(input: Any) -> bytes:
    bytesio = io.BytesIO()
    if isinstance(input, Image.Image):
        input.save(bytesio, format="jpeg", quality=100)
    elif isinstance(input, Figure):
        input.savefig(bytesio, format="png")
        plt.close(input)
    elif isinstance(input, go.Figure):
        input.write_image(bytesio, format="png")
    else:
        raise Exception(f"Input type {input.__class__.__name__} not supported.")
    return base64.b64encode(bytesio.getvalue()).decode("utf-8")

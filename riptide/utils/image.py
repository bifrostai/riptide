import base64
import io
from typing import Any, List, Tuple, Union

import matplotlib.pyplot as plt
import plotly.graph_objects as go
import torch
from matplotlib.figure import Figure
from PIL import Image
from torchvision.io import read_image as read_image_torch
from torchvision.transforms.functional import crop, resize
from torchvision.utils import draw_bounding_boxes

PREVIEW_PADDING = 48
PREVIEW_SIZE = 192


def crop_preview(
    image_tensor: torch.Tensor,
    bboxes: torch.Tensor,
    *,
    colors: Union[
        List[Union[str, Tuple[int, int, int]]], str, Tuple[int, int, int]
    ] = "white",
    preview_size: int = PREVIEW_SIZE,
    preview_padding: int = PREVIEW_PADDING,
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
    bboxes = bboxes.long()

    crop_box = get_padded_bbox_crop(bboxes, padding=preview_padding)
    x1, y1, x2, y2 = crop_box.tolist()
    cropped = crop(image_tensor, y1, x1, y2 - y1, x2 - x1)

    if colors is not None:
        bboxes = bboxes - crop_box[:2].repeat(2)
        bboxes = torch.div(bboxes * preview_size, x2 - x1, rounding_mode="floor")
        cropped = resize(cropped, (preview_size, preview_size))
        cropped = draw_bounding_boxes(cropped, bboxes, colors=colors, width=2)

    return cropped


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
    bbox: torch.Tensor,
    padding: int = PREVIEW_PADDING,
) -> torch.Tensor:
    """Get a padded crop of the bounding box(es)

    Parameters
    ----------
    bbox : torch.Tensor
        Bounding box(es) in xyxy format
    padding : int, optional
        Padding around bounding box, by default 48

    Returns
    -------
    torch.Tensor
        Bounding box of padded crop
    """
    hull, _ = convex_hull(bbox)
    x1, y1, x2, y2 = hull.tolist()
    long_edge = torch.argmax(hull[2:] - hull[:2])  # 0 is w, 1 is h
    if long_edge == 0:
        x1 -= padding
        x2 += padding
        short_edge_padding = torch.div(
            (x2 - x1) - (y2 - y1), 2, rounding_mode="floor"
        ).item()
        y1 -= short_edge_padding
        y2 += short_edge_padding
    else:
        y1 -= padding
        y2 += padding
        short_edge_padding = torch.div(
            ((y2 - y1) - (x2 - x1)), 2, rounding_mode="floor"
        ).item()
        x1 -= short_edge_padding
        x2 += short_edge_padding

    return torch.tensor([x1, y1, x2, y2])


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


def blank_image(width: int, height: int, *, fill_value: int = 255) -> torch.Tensor:
    """Create a blank image

    Parameters
    ----------
    width : int
        Width of image
    height : int
        Height of image

    Returns
    -------
    Image.Image
        Blank image
    """
    t = torch.full((3, height, width), fill_value=fill_value, dtype=torch.uint8)
    t[0, :, :] = torch.linspace(fill_value // 2, fill_value, width).repeat(height, 1)
    t[1, :, :] = torch.linspace(fill_value, fill_value // 2, height).repeat(width, 1).T
    return t


def read_image(
    image_path: str, *, width: int = 1920, height: int = 1920
) -> torch.Tensor:
    """Read an image from a path

    Parameters
    ----------
    image_path : str
        Path to image

    Returns
    -------
    Image.Image
        Image
    """
    try:
        image = read_image_torch(image_path)
    except:
        image = blank_image(width, height)
    return image

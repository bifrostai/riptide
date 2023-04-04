import base64
import io
from typing import Any, Dict, List, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib import colors
from PIL import Image
from torchvision.io import read_image
from torchvision.transforms.functional import crop, to_pil_image
from torchvision.utils import draw_bounding_boxes

from riptide.detection.confusions import Confusion, Confusions
from riptide.detection.errors import (
    BackgroundError,
    ClassificationAndLocalizationError,
    ClassificationError,
    DuplicateError,
    Error,
    LocalizationError,
    MissedError,
)
from riptide.detection.evaluation import ObjectDetectionEvaluator

PALETTE_DARKER = "#222222"
PALETTE_LIGHT = "#FFEECC"
PALETTE_DARK = "#2C3333"
PALETTE_GREEN = "#00FFD9"
PALETTE_BLUE = "#00B3FF"
TRANSPARENT = colors.to_hex((0, 0, 0, 0), keep_alpha=True)
PREVIEW_PADDING = 48
PREVIEW_SIZE = 128


def setup_mpl_params():
    plt.rcParams["text.color"] = PALETTE_LIGHT
    plt.rcParams["xtick.color"] = PALETTE_LIGHT
    plt.rcParams["ytick.color"] = PALETTE_LIGHT
    plt.rcParams["axes.facecolor"] = PALETTE_DARK
    plt.rcParams["axes.labelcolor"] = PALETTE_LIGHT
    plt.rcParams["axes.edgecolor"] = PALETTE_DARKER
    plt.rcParams["figure.facecolor"] = TRANSPARENT
    plt.rcParams["figure.edgecolor"] = PALETTE_LIGHT
    plt.rcParams["savefig.facecolor"] = TRANSPARENT
    plt.rcParams["savefig.edgecolor"] = PALETTE_LIGHT


def add_alpha(color: str, alpha: float) -> str:
    """Adds an alpha value to a color.

    Args:
        color (str): The color to add alpha to.
        alpha (float): The alpha value to add.

    Returns:
        str: The color with alpha.
    """
    rgb = colors.to_rgb(color)
    rgba = (*rgb, alpha)
    return colors.to_hex(rgba)


def gradient(c1: str, c2: str, step: float, output_rgb=False):
    # Linear interpolation from color c1 (at step=0) to c2 (step=1)
    c1 = np.array(colors.to_rgb(c1))
    c2 = np.array(colors.to_rgb(c2))
    rgb = np.clip((1 - step) * c1 + step * c2, 0, 1)
    if output_rgb:
        return rgb
    return colors.to_hex(rgb)


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
    t_max = torch.max(bboxes, dim=0)
    t_min = torch.min(bboxes, dim=0)
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


def get_bbox_stats(bbox: torch.Tensor) -> Tuple[int]:
    area = round(((bbox[2] - bbox[0]) * (bbox[3] - bbox[1])).item(), 2)
    width = int((bbox[2] - bbox[0]).item())
    height = int((bbox[3] - bbox[1]).item())
    return width, height, area


def encode_base64(input: Any) -> bytes:
    bytesio = io.BytesIO()
    if isinstance(input, Image.Image):
        input.save(bytesio, format="jpeg", quality=100)
    elif isinstance(input, plt.Figure):
        input.savefig(bytesio, format="png")
    else:
        raise Exception("Input type not supported.")
    return base64.b64encode(bytesio.getvalue()).decode("utf-8")


def inspect_error_confidence(
    evaluator: ObjectDetectionEvaluator,
    error_type: Error,
) -> Union[bytes, None]:
    """Plots the confidence of the particular error type.

    Args:
        evaluator (ObjectDetectionEvaluator): The evaluator to inspect.
    """
    if error_type == MissedError:
        return None
    confidence_list = []
    for evaluation in evaluator.evaluations:
        for error in evaluation.instances:
            if not type(error) == error_type:
                continue
            confidence_list.append(error.confidence)

    if len(confidence_list) == 0:
        return None

    setup_mpl_params()
    fig, ax = plt.subplots(figsize=(6, 3), dpi=150, constrained_layout=True)
    _, _, patches = ax.hist(confidence_list, bins=41)
    for patch in patches:
        patch.set_facecolor(
            gradient(
                PALETTE_GREEN,
                PALETTE_BLUE,
                1 - (patch.get_x() / max(confidence_list)),
            )
        )
    ax.set_title(f"{error_type.__name__} Confidence")
    ax.set_xlabel("Confidence score")
    ax.set_xlim(0.45, 1.05)
    ax.set_ylabel("Number of Occurences")
    fig = encode_base64(fig)
    return fig


def inspect_background_error(
    evaluator: ObjectDetectionEvaluator,
) -> Dict[int, Dict]:
    """Saves the BackgroundErrors (false positives) of the evaluator to the given
    output directory.

    Args:
        evaluator (ObjectDetectionEvaluator): The evaluator to inspect.

    Returns:
        Dict[int, list]: A dictionary mapping the class id to a list of dictionaries
        containing the images and metadata of the false positives.
    """

    classwise_dict = {}
    for evaluation in evaluator.evaluations:
        for error in evaluation.instances:
            if not isinstance(error, BackgroundError):
                continue
            width, height, area = get_bbox_stats(error.pred_bbox)
            image_tensor = read_image(evaluation.image_path)
            image_tensor = crop_preview(
                image_tensor, error.pred_bbox.unsqueeze(0), "magenta"
            )
            image = to_pil_image(image_tensor)
            image = image.resize((PREVIEW_SIZE, PREVIEW_SIZE))
            image_name = evaluation.image_path.split("/")[-1]
            pred_class_int = int(error.pred_label)
            if pred_class_int not in classwise_dict:
                classwise_dict[pred_class_int] = []
            classwise_dict[pred_class_int].append(
                {
                    "image_name": image_name,
                    "image_base64": encode_base64(image),
                    "class": pred_class_int,
                    "confidence": round(error.confidence, 2),
                    "bbox_width": width,
                    "bbox_height": height,
                    "bbox_area": area,
                }
            )
    return classwise_dict


def inspect_classification_error(
    evaluator: ObjectDetectionEvaluator,
) -> Tuple[Dict[int, Dict], bytes]:
    """Saves the ClassificationErrors of the evaluator to the given output directory.

    Args:
        evaluator (ObjectDetectionEvaluator): The evaluator to inspect.
    """

    classwise_dict = {}
    gt_list, pred_list = [], []
    for evaluation in evaluator.evaluations:
        for error in evaluation.instances:
            if not isinstance(error, ClassificationError):
                continue
            image_tensor = read_image(evaluation.image_path)
            image_tensor = crop_preview(
                image_tensor, error.pred_bbox.unsqueeze(0), "crimson"
            )
            image = to_pil_image(image_tensor)
            image = image.resize((PREVIEW_SIZE, PREVIEW_SIZE))
            image_name = evaluation.image_path.split("/")[-1]
            pred_class_int = error.pred_label
            gt_class_int = error.gt_label
            gt_list.append(gt_class_int)
            pred_list.append(pred_class_int)
            if gt_class_int not in classwise_dict:
                classwise_dict[gt_class_int] = []
            classwise_dict[gt_class_int].append(
                {
                    "image_name": image_name,
                    "image_base64": encode_base64(image),
                    "pred_class": pred_class_int,
                    "confidence": round(error.confidence, 2),
                }
            )
    confusion_dict = {}

    if len(gt_list) == 0:
        return classwise_dict, None
    for gt_class_int, pred_class_int in zip(gt_list, pred_list):
        confusion = (gt_class_int, pred_class_int)
        if confusion not in confusion_dict:
            confusion_dict[confusion] = 0
        confusion_dict[confusion] += 1

    ranked_idxs = sorted(confusion_dict, key=confusion_dict.get, reverse=True)
    confusion_dict = {k: confusion_dict[k] for k in ranked_idxs}

    setup_mpl_params()
    fig, ax = plt.subplots(
        figsize=(4, np.ceil(len(confusion_dict) * 0.4)),
        dpi=150,
        constrained_layout=True,
    )
    ax.barh(
        range(len(confusion_dict)),
        width=confusion_dict.values(),
        color=[
            gradient(
                PALETTE_GREEN, PALETTE_BLUE, 1 - (x / max(confusion_dict.values()))
            )
            for x in confusion_dict.values()
        ],
    )
    ax.set_yticks(range(len(confusion_dict)))
    ax.set_yticklabels(
        [f"gt={k[0]} pred={k[1]}" for k in confusion_dict.keys()], minor=False
    )
    ax.set_title("Classification Error Ranking")
    ax.set_xlabel("Number of Occurences")
    fig = encode_base64(fig)
    return classwise_dict, fig


def inspect_localization_error(
    evaluator: ObjectDetectionEvaluator,
    PREVIEW_SIZE: int = 192,
) -> None:
    """Saves the LocalizationErrors of the evaluator to the given output directory.

    Args:
        evaluator (ObjectDetectionEvaluator): The evaluator to inspect.
        PREVIEW_SIZE (int, optional): The size of the cropped images. Defaults to 192.
    """

    classwise_dict: Dict[int, List[Dict]] = {}
    gt_list, pred_list = [], []
    for evaluation in evaluator.evaluations:
        for error in evaluation.instances:
            if not isinstance(error, LocalizationError):
                continue
            width, height, area = get_bbox_stats(error.gt_bbox)
            image_tensor = read_image(evaluation.image_path)
            bboxes = torch.stack([error.gt_bbox, error.pred_bbox])
            image_tensor = crop_preview(image_tensor, bboxes, ["white", "gold"])
            image = to_pil_image(image_tensor)
            image = image.resize((PREVIEW_SIZE, PREVIEW_SIZE))
            image_name = evaluation.image_path.split("/")[-1]
            pred_class_int = int(error.pred_label)
            gt_class_int = int(error.gt_label)
            gt_list.append(gt_class_int)
            pred_list.append(pred_class_int)
            if pred_class_int not in classwise_dict:
                classwise_dict[gt_class_int] = []
            # breakpoint()
            classwise_dict[gt_class_int].append(
                {
                    "image_name": image_name,
                    "image_base64": encode_base64(image),
                    "class": gt_class_int,
                    "bbox_width": width,
                    "bbox_height": height,
                    "bbox_area": area,
                    "iou": round(
                        evaluation.ious[error.pred_idx][error.gt_idx].item(), 3
                    ),
                }
            )
    confusion_dict: Dict[Tuple[int, int], int] = {}

    if len(gt_list) == 0:
        return classwise_dict, None
    for gt_class_int, pred_class_int in zip(gt_list, pred_list):
        confusion = (gt_class_int, pred_class_int)
        if confusion not in confusion_dict:
            confusion_dict[confusion] = 0
        confusion_dict[confusion] += 1

    ranked_idxs = sorted(confusion_dict, key=confusion_dict.get, reverse=True)
    confusion_dict = {k: confusion_dict[k] for k in ranked_idxs}

    setup_mpl_params()
    fig, ax = plt.subplots(
        figsize=(6, np.ceil(len(confusion_dict) * 0.6)),
        dpi=200,
        constrained_layout=True,
    )
    ax.barh(range(len(confusion_dict)), width=confusion_dict.values())
    ax.set_yticks(range(len(confusion_dict)))
    ax.set_yticklabels(
        [f"gt={k[0]} pred={k[1]}" for k in confusion_dict.keys()], minor=False
    )
    ax.set_title("Classification Error Ranking")
    ax.set_xlabel("Number of Occurences")
    fig = encode_base64(fig)
    return classwise_dict, fig


def inspect_classification_and_localization_error(
    evaluator: ObjectDetectionEvaluator,
    PREVIEW_SIZE: int = 192,
) -> None:
    """Saves the ClassificationAndLocalizationError of the evaluator to the given output directory.

    Args:
        evaluator (ObjectDetectionEvaluator): The evaluator to inspect.
        PREVIEW_SIZE (int, optional): The size of the cropped images. Defaults to 192.
    """

    classwise_dict = {}
    gt_list, pred_list = [], []
    for evaluation in evaluator.evaluations:
        for error in evaluation.instances:
            if not isinstance(error, ClassificationAndLocalizationError):
                continue
            width, height, area = get_bbox_stats(error.gt_bbox)
            image_tensor = read_image(evaluation.image_path)
            bboxes = torch.stack([error.gt_bbox, error.pred_bbox])
            image_tensor = crop_preview(image_tensor, bboxes, ["white", "darkorange"])
            image = to_pil_image(image_tensor)
            image = image.resize((PREVIEW_SIZE, PREVIEW_SIZE))
            image_name = evaluation.image_path.split("/")[-1]
            pred_class_int = int(error.pred_label)
            gt_class_int = int(error.gt_label)
            gt_list.append(gt_class_int)
            pred_list.append(pred_class_int)
            if gt_class_int not in classwise_dict:
                classwise_dict[gt_class_int] = []
            classwise_dict[gt_class_int].append(
                {
                    "image_name": image_name,
                    "image_base64": encode_base64(image),
                    "class": pred_class_int,
                    "bbox_width": width,
                    "bbox_height": height,
                    "bbox_area": area,
                    "iou": round(
                        evaluation.ious[error.pred_idx][error.gt_idx].item(), 3
                    ),
                }
            )
    confusion_dict = {}

    if len(gt_list) == 0:
        return classwise_dict, None
    for gt_class_int, pred_class_int in zip(gt_list, pred_list):
        confusion = (gt_class_int, pred_class_int)
        if confusion not in confusion_dict:
            confusion_dict[confusion] = 0
        confusion_dict[confusion] += 1

    ranked_idxs = sorted(confusion_dict, key=confusion_dict.get, reverse=True)
    confusion_dict = {k: confusion_dict[k] for k in ranked_idxs}

    setup_mpl_params()
    fig, ax = plt.subplots(
        figsize=(6, np.ceil(len(confusion_dict) * 0.6)),
        dpi=200,
        constrained_layout=True,
    )
    ax.barh(
        range(len(confusion_dict)),
        width=confusion_dict.values(),
        color=[
            gradient(
                PALETTE_GREEN, PALETTE_BLUE, 1 - (x / max(confusion_dict.values()))
            )
            for x in confusion_dict.values()
        ],
    )
    ax.set_yticks(range(len(confusion_dict)))
    ax.set_yticklabels(
        [f"gt={k[0]} pred={k[1]}" for k in confusion_dict.keys()], minor=False
    )
    ax.set_title("Classification and Localization Error Ranking")
    ax.set_xlabel("Number of Occurences")
    fig = encode_base64(fig)
    return classwise_dict, fig


def inspect_duplicate_error(
    evaluator: ObjectDetectionEvaluator,
    PREVIEW_SIZE: int = 192,
) -> None:
    """Saves the DuplicateError of the evaluator to the given output directory.

    Args:
        evaluator (ObjectDetectionEvaluator): The evaluator to inspect.
        PREVIEW_SIZE (int, optional): The size of the cropped images. Defaults to 192.
    """

    classwise_dict = {}
    gt_list, pred_list = [], []
    for evaluation in evaluator.evaluations:
        for error in evaluation.instances:
            if not isinstance(error, DuplicateError):
                continue
            width, height, area = get_bbox_stats(error.gt_bbox)
            image_tensor = read_image(evaluation.image_path)
            bboxes = torch.stack([error.gt_bbox, error.best_pred_bbox, error.pred_bbox])
            image_tensor = crop_preview(image_tensor, bboxes, ["white", "lime", "red"])
            image = to_pil_image(image_tensor)
            image = image.resize((PREVIEW_SIZE, PREVIEW_SIZE))
            image_name = evaluation.image_path.split("/")[-1]
            pred_class_int = int(error.pred_label)
            gt_class_int = int(error.gt_label)
            gt_list.append(gt_class_int)
            pred_list.append(pred_class_int)
            if pred_class_int not in classwise_dict:
                classwise_dict[gt_class_int] = []
            # breakpoint()
            classwise_dict[gt_class_int].append(
                {
                    "image_name": image_name,
                    "image_base64": encode_base64(image),
                    "class": gt_class_int,
                    "bbox_width": width,
                    "bbox_height": height,
                    "bbox_area": area,
                    "iou": round(
                        evaluation.ious[error.pred_idx][error.gt_idx].item(), 3
                    ),
                    "best_iou": round(
                        evaluation.ious[error.best_pred_idx][error.gt_idx].item(), 3
                    ),
                    "conf": round(error.confidence, 2),
                    "best_conf": round(error.best_confidence, 2),
                }
            )
    confusion_dict = {}

    if len(gt_list) == 0:
        return classwise_dict, None
    for gt_class_int, pred_class_int in zip(gt_list, pred_list):
        confusion = (gt_class_int, pred_class_int)
        if confusion not in confusion_dict:
            confusion_dict[confusion] = 0
        confusion_dict[confusion] += 1

    ranked_idxs = sorted(confusion_dict, key=confusion_dict.get, reverse=True)
    confusion_dict = {k: confusion_dict[k] for k in ranked_idxs}

    setup_mpl_params()
    fig, ax = plt.subplots(
        figsize=(6, np.ceil(len(confusion_dict) * 0.6)),
        dpi=200,
        constrained_layout=True,
    )
    ax.barh(range(len(confusion_dict)), width=confusion_dict.values())
    ax.set_yticks(range(len(confusion_dict)))
    ax.set_yticklabels(
        [f"gt={k[0]} pred={k[1]}" for k in confusion_dict.keys()], minor=False
    )
    ax.set_title("Duplicate Error Ranking")
    ax.set_xlabel("Number of Occurences")
    fig = encode_base64(fig)
    return classwise_dict, fig


def inspect_missed_error(
    evaluator: ObjectDetectionEvaluator,
) -> Tuple[Dict[int, dict], bytes]:
    """Saves the MissedErrors (false negatives) of the evaluator to the given
    output directory.

    Args:
        evaluator (ObjectDetectionEvaluator): The evaluator to inspect.

    Returns:
        Dict[int, list]: A dictionary mapping the class id to a list of dictionaries
        containing the images and metadata of the false negatives.
    """

    classwise_dict = {}
    for evaluation in evaluator.evaluations:
        for error in evaluation.instances:
            if not isinstance(error, MissedError):
                continue
            width, height, area = get_bbox_stats(error.gt_bbox)
            image_tensor = read_image(evaluation.image_path)
            image_tensor = crop_preview(
                image_tensor, error.gt_bbox.unsqueeze(0), "yellowgreen"
            )
            image = to_pil_image(image_tensor)
            image = image.resize((PREVIEW_SIZE, PREVIEW_SIZE))
            image_name = evaluation.image_path.split("/")[-1]
            gt_class_int = int(error.gt_label)
            if gt_class_int not in classwise_dict:
                classwise_dict[gt_class_int] = []
            classwise_dict[gt_class_int].append(
                {
                    "image_name": image_name,
                    "image_base64": encode_base64(image),
                    "class": gt_class_int,
                    "bbox_width": width,
                    "bbox_height": height,
                    "bbox_area": area,
                }
            )

    # Plot the barplots of the classwise area
    setup_mpl_params()
    fig, ax = plt.subplots(figsize=(6, 6), dpi=150, constrained_layout=True)
    missed_areas = {}
    for class_idx, missed in classwise_dict.items():
        areas = [m["bbox_area"] for m in missed]
        missed_areas[class_idx] = areas

    for class_idx, area in missed_areas.items():
        ax.scatter(
            np.full_like(area, class_idx),
            area,
            color=[gradient(PALETTE_BLUE, PALETTE_GREEN, a / max(area)) for a in area],
            edgecolors="none",
        )
    boxplot = ax.boxplot(
        missed_areas.values(),
        positions=list(missed_areas.keys()),
        patch_artist=True,
    )

    for cap in boxplot["caps"]:
        cap.set_color(PALETTE_LIGHT)
    for whisker in boxplot["whiskers"]:
        whisker.set_color(PALETTE_LIGHT)
    for median in boxplot["medians"]:
        median.set_color(PALETTE_GREEN)
        median.set_linewidth(2)
    for box in boxplot["boxes"]:
        box.set(color=PALETTE_LIGHT, facecolor=TRANSPARENT)

    fig = encode_base64(fig)
    return classwise_dict, fig


def inspect_true_positives(
    evaluator: ObjectDetectionEvaluator,
) -> Tuple[Dict[int, Dict], bytes]:
    """Saves the TruePositives of the evaluator to the given output directory.
    Probably should create some Factory method for this shagload of inspecting functions lol
    another day i guess TODO

    Args:
        evaluator (ObjectDetectionEvaluator): The evaluator to inspect.
        PREVIEW_SIZE (int, optional): The size of the cropped images. Defaults to 192.
    """
    classwise_dict = {}
    for evaluation in evaluator.evaluations:
        ## get the array of predictions
        ## get the array of indexes in the predictions array
        ## get all other information using the index
        pred_conf_array = evaluation.confusions.pred_confusions
        pred_tp_idxs = [
            i
            for i in range(len(pred_conf_array))
            if pred_conf_array[i] is Confusion.TRUE_POSITIVE
        ]
        selected_pred_ious = [
            evaluation.ious[pred_idx]
            for pred_idx in range(len(pred_conf_array))
            if pred_idx in pred_tp_idxs
        ]
        gt_idxs = []

        for pred_iou in selected_pred_ious:
            ## this is the corresponding gt_idx given the pred_tp_idx
            idx_of_best_gt_match = pred_iou.argmax()
            gt_idxs.append(idx_of_best_gt_match)

        for tp_idx, gt_idx in zip(pred_tp_idxs, gt_idxs):
            width, height, area = get_bbox_stats(evaluation.pred_bboxes[tp_idx])
            image_tensor = read_image(evaluation.image_path)
            bboxes = torch.stack(
                [evaluation.pred_bboxes[tp_idx], evaluation.gt_bboxes[gt_idx]]
            )
            image_tensor = crop_preview(image_tensor, bboxes, ["white", "lime"])
            image = to_pil_image(image_tensor)
            image = image.resize((PREVIEW_SIZE, PREVIEW_SIZE))
            image_name = evaluation.image_path.split("/")[-1]
            pred_class_int = int(evaluation.pred_labels[tp_idx].item())
            if pred_class_int not in classwise_dict:
                classwise_dict[pred_class_int] = []
            classwise_dict[pred_class_int].append(
                {
                    "image_name": image_name,
                    "image_base64": encode_base64(image),
                    "class": pred_class_int,
                    "bbox_width": width,
                    "bbox_height": height,
                    "bbox_area": area,
                    "conf": round(evaluation.pred_scores[tp_idx].item(), 2),
                    "iou": round(evaluation.ious[tp_idx][gt_idx].item(), 2),
                }
            )

    # Plot the barplots of the classwise area
    setup_mpl_params()
    fig, ax = plt.subplots(figsize=(6, 6), dpi=150, constrained_layout=True)
    missed_areas = {}
    for class_idx, missed in classwise_dict.items():
        areas = [m["bbox_area"] for m in missed]
        missed_areas[class_idx] = areas

    for class_idx, area in missed_areas.items():
        ax.scatter(
            np.full_like(area, class_idx),
            area,
            color=[gradient(PALETTE_BLUE, PALETTE_GREEN, a / max(area)) for a in area],
            edgecolors="none",
        )
    boxplot = ax.boxplot(
        missed_areas.values(),
        positions=list(missed_areas.keys()),
        patch_artist=True,
    )

    for cap in boxplot["caps"]:
        cap.set_color(PALETTE_LIGHT)
    for whisker in boxplot["whiskers"]:
        whisker.set_color(PALETTE_LIGHT)
    for median in boxplot["medians"]:
        median.set_color(PALETTE_GREEN)
        median.set_linewidth(2)
    for box in boxplot["boxes"]:
        box.set(color=PALETTE_LIGHT, facecolor=TRANSPARENT)

    fig = encode_base64(fig)
    return classwise_dict, fig

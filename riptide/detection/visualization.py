import base64
import io
from typing import Any, Dict, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colors
from PIL import Image
from torchvision.io import read_image
from torchvision.transforms.functional import to_pil_image

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

plt.rcParams["text.color"] = "#FFEECC"
plt.rcParams["xtick.color"] = "#FFEECC"
plt.rcParams["ytick.color"] = "#FFEECC"
plt.rcParams["axes.facecolor"] = "#2C3333"
plt.rcParams["axes.labelcolor"] = "#FFEECC"
plt.rcParams["axes.edgecolor"] = "#222222"
plt.rcParams["figure.facecolor"] = "#00000000"
plt.rcParams["figure.edgecolor"] = "#FFEECC"
plt.rcParams["savefig.facecolor"] = "#00000000"
plt.rcParams["savefig.edgecolor"] = "#FFEECC"


GRADIENT_GREEN = "#00ffd9"
GRADIENT_BLUE = "#00b3ff"


def gradient(c1: str, c2: str, step: float, output_rgb=False):
    # Linear interpolation from color c1 (at step=0) to c2 (step=1)
    c1 = np.array(colors.to_rgb(c1))
    c2 = np.array(colors.to_rgb(c2))
    rgb = np.clip((1 - step) * c1 + step * c2, 0, 1)
    if output_rgb:
        return rgb
    return colors.to_hex(rgb)


def encode_base64(input: Any) -> bytes:
    bytesio = io.BytesIO()
    if isinstance(input, Image.Image):
        input.save(bytesio, format="jpeg")
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
            confidence_list.append(error.confidence.item())

    if len(confidence_list) == 0:
        return None

    fig, ax = plt.subplots(figsize=(6, 3), dpi=150, constrained_layout=True)
    _, _, patches = ax.hist(confidence_list, bins=41)
    for patch in patches:
        patch.set_facecolor(
            gradient(
                GRADIENT_GREEN,
                GRADIENT_BLUE,
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
    crop_size: int = 192,
) -> Dict[int, dict]:
    """Saves the BackgroundErrors (false positives) of the evaluator to the given
    output directory.

    Args:
        evaluator (ObjectDetectionEvaluator): The evaluator to inspect.
        crop_size (int, optional): The size of the cropped images. Defaults to 192.

    Returns:
        Dict[int, list]: A dictionary mapping the class id to a list of dictionaries
        containing the images and metadata of the false positives.
    """

    classwise_dict = {}
    for evaluation in evaluator.evaluations:
        for error in evaluation.instances:
            if not isinstance(error, BackgroundError):
                continue
            x1, y1, x2, y2 = [int(x.item()) for x in error.pred_bbox]
            bbox_area = (x2 - x1) * (y2 - y1)
            image_tensor = read_image(evaluation.image_path)
            image_tensor = image_tensor[:, y1:y2, x1:x2]
            image = to_pil_image(image_tensor)
            image = image.resize((crop_size, crop_size))
            image_name = evaluation.image_path.split("/")[-1]
            pred_class_int = int(error.pred_label.item())
            if pred_class_int not in classwise_dict:
                classwise_dict[pred_class_int] = []
            classwise_dict[pred_class_int].append(
                {
                    "image_name": image_name,
                    "image_base64": encode_base64(image),
                    "class": pred_class_int,
                    "confidence": round(error.confidence.item(), 2),
                    "bbox_area": bbox_area,
                }
            )
    return classwise_dict


def inspect_classification_error(
    evaluator: ObjectDetectionEvaluator,
    crop_size: int = 192,
) -> Tuple[Dict[int, dict], bytes]:
    """Saves the ClassificationErrors of the evaluator to the given output directory.

    Args:
        evaluator (ObjectDetectionEvaluator): The evaluator to inspect.
        crop_size (int, optional): The size of the cropped images. Defaults to 192.
    """

    classwise_dict = {}
    gt_list, pred_list = [], []
    for evaluation in evaluator.evaluations:
        for error in evaluation.instances:
            if not isinstance(error, ClassificationError):
                continue
            x1, y1, x2, y2 = [int(x.item()) for x in error.pred_bbox]
            image_tensor = read_image(evaluation.image_path)
            image_tensor = image_tensor[:, y1:y2, x1:x2]
            image = to_pil_image(image_tensor)
            image = image.resize((crop_size, crop_size))
            image_name = evaluation.image_path.split("/")[-1]
            pred_class_int = int(error.pred_label.item())
            gt_class_int = int(error.gt_label.item())
            gt_list.append(gt_class_int)
            pred_list.append(pred_class_int)
            if pred_class_int not in classwise_dict:
                classwise_dict[gt_class_int] = []
            classwise_dict[gt_class_int].append(
                {
                    "image_name": image_name,
                    "image_base64": encode_base64(image),
                    "pred_class": pred_class_int,
                    "confidence": round(error.confidence.item(), 2),
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
                GRADIENT_GREEN, GRADIENT_BLUE, 1 - (x / max(confusion_dict.values()))
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
    crop_size: int = 192,
) -> None:
    """Saves the LocalizationErrors of the evaluator to the given output directory.

    Args:
        evaluator (ObjectDetectionEvaluator): The evaluator to inspect.
        crop_size (int, optional): The size of the cropped images. Defaults to 192.
    """

    classwise_dict = {}
    gt_list, pred_list = [], []
    for evaluation in evaluator.evaluations:
        for error in evaluation.instances:
            if not isinstance(error, ClassificationError):
                continue
            x1, y1, x2, y2 = [int(x.item()) for x in error.pred_bbox]
            image_tensor = read_image(evaluation.image_path)
            image_tensor = image_tensor[:, y1:y2, x1:x2]
            image = to_pil_image(image_tensor)
            image = image.resize((crop_size, crop_size))
            image_name = evaluation.image_path.split("/")[-1]
            pred_class_int = int(error.pred_label.item())
            gt_class_int = int(error.gt_label.item())
            gt_list.append(gt_class_int)
            pred_list.append(pred_class_int)
            if pred_class_int not in classwise_dict:
                classwise_dict[gt_class_int] = []
            classwise_dict[gt_class_int].append(
                {
                    "image_name": image_name,
                    "image_base64": encode_base64(image),
                    "pred_class": pred_class_int,
                    "confidence": round(error.confidence.item(), 2),
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


def inspect_missed_error(
    evaluator: ObjectDetectionEvaluator,
    crop_size: int = 192,
) -> Dict[int, dict]:
    """Saves the MissedErrors (false negatives) of the evaluator to the given
    output directory.

    Args:
        evaluator (ObjectDetectionEvaluator): The evaluator to inspect.
        crop_size (int, optional): The size of the cropped images. Defaults to 192.

    Returns:
        Dict[int, list]: A dictionary mapping the class id to a list of dictionaries
        containing the images and metadata of the false negatives.
    """

    classwise_dict = {}
    for evaluation in evaluator.evaluations:
        for error in evaluation.instances:
            if not isinstance(error, MissedError):
                continue
            x1, y1, x2, y2 = [int(x.item()) for x in error.gt_bbox]
            bbox_area = (x2 - x1) * (y2 - y1)
            image_tensor = read_image(evaluation.image_path)
            image_tensor = image_tensor[:, y1:y2, x1:x2]
            image = to_pil_image(image_tensor)
            image = image.resize((crop_size, crop_size))
            image_name = evaluation.image_path.split("/")[-1]
            gt_class_int = int(error.gt_label.item())
            if gt_class_int not in classwise_dict:
                classwise_dict[gt_class_int] = []
            classwise_dict[gt_class_int].append(
                {
                    "image_name": image_name,
                    "image_base64": encode_base64(image),
                    "class": gt_class_int,
                    "bbox_area": bbox_area,
                }
            )
    return classwise_dict


def inspect_missed_error_size_distribution(
    evaluator: ObjectDetectionEvaluator,
) -> bytes:
    pass

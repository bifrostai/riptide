import base64
import io
import os
from typing import Any, Dict, Tuple, Type, Union

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colors
from PIL import Image
import torch
from torchvision.io import read_image
from torchvision.transforms.functional import crop, to_pil_image
from torchvision.transforms import Compose, ToTensor
from torchvision.transforms.functional import crop
from tqdm import tqdm

from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA

from mise_en_place.encoders import VariableLayerEncoder
from mise_en_place.transforms import normalize, inverse_normalize

from riptide.detection.evaluation import ObjectDetectionEvaluator

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
            if gt_class_int not in classwise_dict:
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


def inspect_missed_error(
    evaluator: ObjectDetectionEvaluator,
    crop_size: int = 192,
) -> Tuple[Dict[int, dict], bytes]:
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

    # Plot the barplots of the classwise area
    setup_mpl_params()
    fig, ax = plt.subplots(figsize=(6, 6), dpi=150, constrained_layout=True)
    missed_areas = []
    for idx, (class_int, missed) in enumerate(classwise_dict.items()):
        areas = [m["bbox_area"] for m in missed]
        mean, std = np.mean(areas), np.std(areas)
        scatter_colors = [
            gradient(
                PALETTE_BLUE,
                PALETTE_GREEN,
                np.abs(mean - a) / (2 * std),
            )
            for a in areas
        ]
        ax.scatter(np.full_like(areas, idx + 1), areas, color=scatter_colors)
        missed_areas.append(areas)

    boxplot = ax.boxplot(missed_areas, patch_artist=True)
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


def group_errors(
    evaluator: ObjectDetectionEvaluator,
    error_type: Type[Error],
    device: torch.device = torch.device("cuda"),
    use_cached_embeddings: bool = True,
) -> Dict[str, int]:
    """Group error instances across error type using perceptual similarity. Embeddings
    are first created using early layers of a pre-trained neural network. They are then
    normalized, dimension-reduced using PCA, and clustered using DBSCAN.

    Args:
        evaluator (ObjectDetectionEvaluator): Object detection evaluator.
        error_type (Type[Error]): Error type to group.
        device (torch.device, optional): Device to use. Defaults to torch.device("cuda").
        use_cached_embeddings (bool, optional): Use cached embeddings if available.
            Defaults to True.

    Returns:
        Dict[str, int]: Dictionary mapping evaluation image_path to cluster index.
    """

    if use_cached_embeddings:
        if not os.path.exists("embeddings_cache"):
            os.mkdir("embeddings_cache")

        embeddings_path = os.path.join("embeddings_cache", "embeddings.pt")

        if not os.path.exists(embeddings_path):
            encoder = VariableLayerEncoder("preconv")
            transform = Compose([ToTensor(), normalize()])

            filtered_errors = []
            embeddings = []

            for evaluation in tqdm(evaluator.evaluations):
                image = Image.open(evaluation.image_path)
                image = transform(image)

                specific_errors = evaluation.errors.get_errors(error_type)

                for error in specific_errors:
                    bbox = (
                        error.gt_bbox
                        if isinstance(error, MissedError)
                        else error.pred_bbox
                    )
                    x, y, h, w = [int(c) for c in bbox]

                    padding = 10
                    x = x - padding
                    y = y - padding
                    h = h + (2 * padding)
                    w = w + (2 * padding)

                    instance = crop(image, y, x, w, h).unsqueeze(0)

                    filtered_errors.append((evaluation.image_path, error))

                    with torch.no_grad():
                        embeddings.append(encoder(instance.to(device)).squeeze(0))

            embeddings = torch.stack(embeddings)
            embeddings = torch.nan_to_num(
                (embeddings - embeddings.min(dim=0)[0])
                / (embeddings.max(dim=0)[0] - embeddings.min(dim=0)[0])
            )
            torch.save(embeddings, embeddings_path)

        else:
            embeddings = torch.load(embeddings_path)

    # Dimension reduction using PCA
    pca = PCA(n_components=10)
    embeddings = pca.fit_transform(embeddings)

    # Clustering reduced embeddings using DBSCAN
    dbscan = DBSCAN()
    labels = dbscan.fit_predict(embeddings)
    filtered_errors = np.array(filtered_errors)
    cluster_dict = {}
    for label, errors in zip(labels, filtered_errors):
        if label not in cluster_dict:
            cluster_dict[label] = []
        cluster_dict[label].append(errors)

    return cluster_dict

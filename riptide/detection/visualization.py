import base64
import io
from typing import Any, Callable, Dict, Iterable, List, Tuple, Type, Union

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib import colors
from PIL import Image
from torchvision.io import read_image
from torchvision.ops.boxes import box_iou
from torchvision.transforms.functional import crop, to_pil_image
from torchvision.utils import draw_bounding_boxes

from riptide.detection.embeddings.projector import CropProjector
from riptide.detection.errors import (
    BackgroundError,
    ClassificationAndLocalizationError,
    ClassificationError,
    DuplicateError,
    Error,
    LocalizationError,
    MissedError,
    NonError,
)
from riptide.detection.evaluation import ObjectDetectionEvaluator
from riptide.utils.colors import ErrorColor
from riptide.utils.logging import logger

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
    elif isinstance(input, plt.Figure):
        input.savefig(bytesio, format="png")
        plt.close(input)
    else:
        raise Exception(f"Input type {input.__class__.__name__} not supported.")
    return base64.b64encode(bytesio.getvalue()).decode("utf-8")


def get_both_bboxes(error: Error, bbox_attr: str):
    return torch.stack([error.gt_bbox, error.pred_bbox])


class Inspector:
    def __init__(self, evaluator: ObjectDetectionEvaluator):
        self.evaluator = evaluator
        self.errorlist_dict = self.evaluator.get_errorlist_dict()

        crops, errors = evaluator.crop_objects()
        self.pred_projector = CropProjector(
            name=evaluator.name,
            images=crops,
            encoder_mode="preconv",
            normalize_embeddings=True,
            labels=[error.code if error is not None else "NON" for error in errors],
            device=torch.device("cpu"),
        )

        crops, errors = evaluator.crop_objects(axis=0)
        self.gt_projector = CropProjector(
            name=evaluator.name,
            images=crops,
            encoder_mode="preconv",
            normalize_embeddings=True,
            labels=[error.code if error is not None else "NON" for error in errors],
            device=torch.device("cpu"),
        )

    def _confidence_hist(
        self, confidence_list: List[float], error_name: str = "Error"
    ) -> bytes:
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
        ax.set_title(f"{error_name} Confidence")
        ax.set_xlabel("Confidence score")
        ax.set_xlim(0.45, 1.05)
        ax.set_ylabel("Number of Occurences")
        return encode_base64(fig)

    @logger()
    def error_confidence(
        self, error_types: Union[Type[Error], Iterable[Type[Error]]]
    ) -> Dict[str, bytes]:
        """Computes a dictionary of plots for the confidence of given error types.

        Arguments
        ---------
        error_types : Union[Error, Iterable[Error]]
            Error types to plot confidence for

        Returns
        -------
        Dict[str, bytes]
            Dictionary of plots for each error type
        """
        error_types = (
            tuple(error_types) if isinstance(error_types, Iterable) else (error_types,)
        )

        confidence_lists = {
            error_type.__name__: [
                error.confidence
                for errors in self.errorlist_dict.get(
                    error_type.__name__, dict()
                ).values()
                for error in errors
                if error.confidence is not None
            ]
            for error_type in error_types
        }

        confidence_hists = {
            error_name: self._confidence_hist(confidence_list, error_name)
            if confidence_list
            else None
            for error_name, confidence_list in confidence_lists.items()
        }

        return confidence_hists

    @logger()
    def boxplot(self, classwise_dict: Dict[int, List[Dict]]) -> bytes:
        """Computes a dictionary of plots for the confidence of given error types.

        Arguments
        ---------
        error_type : Error
            Error type to plot boxplot for

        Returns
        -------
        Dict[str, bytes]
            Dictionary of plots for each error type
        """
        # Plot the barplots of the classwise area
        setup_mpl_params()
        fig, ax = plt.subplots(figsize=(6, 6), dpi=150, constrained_layout=True)
        area_info = {}
        for class_idx, info_dict in classwise_dict.items():
            areas = [m["bbox_area"] for m in info_dict]
            area_info[class_idx] = areas

        for class_idx, area in area_info.items():
            ax.scatter(
                np.full_like(area, class_idx),
                area,
                color=[
                    gradient(PALETTE_BLUE, PALETTE_GREEN, a / max(area)) for a in area
                ],
                edgecolors="none",
            )
        boxplot = ax.boxplot(
            area_info.values(),
            positions=list(area_info.keys()),
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

        return encode_base64(fig)

    def error_classwise_dict(
        self,
        error_type: Type[Error],
        color: Union[str, ErrorColor, List[Union[str, ErrorColor]]],
        axis: int = 0,
        *,
        preview_size=128,
        get_bbox_func: Callable[[Tuple[Error, str]], torch.Tensor] = None,
        get_additional_metadata_func: Callable[[Error], Dict] = None,
        bbox_attr: str = None,
        label_attr: str = None,
        projector_attr: str = None,
    ) -> Dict[int, List[Dict]]:
        """Computes a dictionary of plots for the confidence of given error types, classwise.

        Arguments
        ---------
        error_types : Union[Error, Iterable[Error]]
            Error types to plot confidence for

        color : Union[str, ErrorColor, List[Union[str, ErrorColor]]]
            Color(s) for the bounding box(es). Can be a single color or a list of colors

        axis : int, optional
            Axis to crop image on. 0 for ground truth, 1 for predictions, by default 0

        Returns
        -------
        Dict[str, Dict[str, bytes]]
            Dictionary of plots for each error type, classwise
        """

        errors = self.errorlist_dict.get(error_type.__name__, dict())
        if len(errors) == 0:
            return dict()

        if not isinstance(color, list):
            color = [color]

        color = [c.colorstr if isinstance(c, ErrorColor) else c for c in color]

        if axis == 0:
            bbox_attr = bbox_attr or "gt_bbox"
            label_attr = label_attr or "gt_label"
            projector_attr = projector_attr or "gt_projector"
        else:
            bbox_attr = bbox_attr or "pred_bbox"
            label_attr = label_attr or "pred_label"
            projector_attr = projector_attr or "pred_projector"

        if get_bbox_func is None:

            def get_bbox_func(error: Error, bbox_attr: str) -> torch.Tensor:
                t = getattr(error, bbox_attr)
                assert isinstance(t, torch.Tensor), f"{bbox_attr} is not a tensor"
                return t.unsqueeze(0)

        if get_additional_metadata_func is None:

            def get_additional_metadata_func(error: Error) -> Dict:
                return dict()

        projector: CropProjector = getattr(self, projector_attr)

        code = "TP" if error_type is NonError else error_type.code
        clusters = projector.cluster(by_labels=code)

        classwise_dict: Dict[int, List[Dict]] = {}
        label_set = set()
        idx = 0
        for image_path, image_errors in errors.items():
            image_tensor = read_image(image_path)
            image_name = image_path.split("/")[-1]
            for error in image_errors:
                bbox: torch.Tensor = getattr(error, bbox_attr)
                label: int = getattr(error, label_attr)
                label_set.add(label)

                if bbox is not None:
                    width, height, area = get_bbox_stats(bbox)
                    bboxes = get_bbox_func(error, bbox_attr)

                    crop_tensor = (
                        crop_preview(image_tensor, bboxes, color)
                        if isinstance(bboxes, torch.Tensor)
                        else image_tensor
                    )
                    crop: Image.Image = to_pil_image(crop_tensor)
                    encoded_crop = encode_base64(
                        crop.resize((preview_size, preview_size))
                    )
                else:
                    width, height, area = (None, None, None)
                    encoded_crop = None

                if label not in classwise_dict:
                    classwise_dict[label] = []

                confidence = (
                    round(error.confidence, 2) if error.confidence is not None else None
                )
                iou = (
                    round(
                        box_iou(
                            error.pred_bbox.unsqueeze(0), error.gt_bbox.unsqueeze(0)
                        ).item(),
                        3,
                    )
                    if error.pred_bbox is not None and error.gt_bbox is not None
                    else None
                )

                classwise_dict[label].append(
                    {
                        "image_name": image_name,
                        "image_base64": encoded_crop,
                        "pred_class": error.pred_label,
                        "gt_class": error.gt_label,
                        "confidence": confidence,
                        "bbox_width": width,
                        "bbox_height": height,
                        "bbox_area": area,
                        "iou": iou,
                        "cluster": clusters[idx],
                        **get_additional_metadata_func(error),
                    }
                )
                idx += 1

        for label in label_set:
            classwise_dict[label] = sorted(
                classwise_dict[label], key=lambda x: x["cluster"], reverse=True
            )

        return classwise_dict

    def error_classwise_ranking(self, error_type: Type[Error]) -> bytes:

        labels = [
            (error.gt_label, error.pred_label)
            for errors in self.errorlist_dict.get(error_type.__name__, dict()).values()
            for error in errors
        ]

        if len(labels) == 0:
            return None

        confusion_dict = {confusion: 0 for confusion in set(labels)}

        for confusion in labels:
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
        ax.set_title("Classwise Error Ranking")
        ax.set_xlabel("Number of Occurences")
        return encode_base64(fig)

    @logger()
    def background_error(self) -> Dict[int, List[Dict]]:
        """Saves the BackgroundErrors (false positives) of the evaluator to the given
        output directory.

        Returns
        -------
        Dict[int, List]
            A dictionary mapping the class id to a list of dictionaries
            containing the images and metadata of the false positives.
        """
        figs = self.error_classwise_dict(BackgroundError, color=ErrorColor.BKG, axis=1)
        return dict(
            sorted(
                figs.items(),
                key=lambda x: len(x[1]),
                reverse=True,
            )
        )

    @logger()
    def classification_error(self) -> Tuple[Dict[int, List[Dict]], bytes]:
        """Saves the ClassificationErrors of the evaluator to the given output directory.

        Returns
        -------
        Dict[int, List]
            A dictionary mapping the class id to a list of dictionaries
            containing the images and metadata of the false positives.
        """

        classwise_dict = self.error_classwise_dict(
            ClassificationError, color=ErrorColor.CLS, axis=1, label_attr="gt_label"
        )
        fig = self.error_classwise_ranking(ClassificationError)

        return classwise_dict, fig

    def localization_error(self) -> Tuple[Dict[int, Dict], bytes]:
        """Saves the LocalizationErrors of the evaluator to the given output directory.

        Arguments
        ---------
        PREVIEW_SIZE : int, optional
            Size of the cropped images, by default 192
        """

        classwise_dict = self.error_classwise_dict(
            LocalizationError,
            color=[ErrorColor.WHITE, ErrorColor.LOC],
            axis=1,
            get_bbox_func=get_both_bboxes,
            preview_size=192,
        )
        fig = self.error_classwise_ranking(LocalizationError)

        return classwise_dict, fig

    def classification_and_localization_error(self) -> Tuple[Dict[int, Dict], bytes]:
        """Saves the ClassificationAndLocalizationErrors of the evaluator to the given output directory.

        Returns
        -------
        Dict[int, List]
            A dictionary mapping the class id to a list of dictionaries
            containing the images and metadata of the false positives.
        """
        classwise_dict = self.error_classwise_dict(
            ClassificationAndLocalizationError,
            color=[ErrorColor.WHITE, ErrorColor.CLL],
            axis=1,
            get_bbox_func=get_both_bboxes,
            preview_size=192,
            label_attr="gt_label",
        )
        fig = self.error_classwise_ranking(ClassificationAndLocalizationError)

        return classwise_dict, fig

    def duplicate_error(self) -> Tuple[Dict[int, Dict], bytes]:
        """Saves the DuplicateErrors of the evaluator to the given output directory.

        Returns
        -------
        Dict[int, List]
            A dictionary mapping the class id to a list of dictionaries
            containing the images and metadata of the false positives.
        """

        def get_bbox_func(error: DuplicateError, attr: str):
            return torch.stack([error.gt_bbox, error.best_pred_bbox, error.pred_bbox])

        def get_additional_metadata_func(error: DuplicateError):
            iou = round(
                box_iou(
                    error.best_pred_bbox.unsqueeze(0), error.gt_bbox.unsqueeze(0)
                ).item(),
                3,
            )
            return {
                "best_iou": iou,
                "best_conf": round(error.best_confidence, 2),
            }

        classwise_dict = self.error_classwise_dict(
            DuplicateError,
            color=[ErrorColor.WHITE, ErrorColor.TP, ErrorColor.DUP],
            axis=1,
            get_bbox_func=get_bbox_func,
            preview_size=192,
            get_additional_metadata_func=get_additional_metadata_func,
        )
        classwise_dict = dict(
            sorted(
                classwise_dict.items(),
                key=lambda x: len(x[1]),
                reverse=True,
            )
        )

        fig = self.boxplot(classwise_dict)

        return classwise_dict, fig

    def missed_error(self) -> Tuple[Dict[int, Dict], bytes]:
        """Saves the MissedErrors of the evaluator to the given output directory.

        Returns
        -------
        Dict[int, List]
            A dictionary mapping the class id to a list of dictionaries
            containing the images and metadata of the false positives.
        """
        classwise_dict = self.error_classwise_dict(
            MissedError, color=ErrorColor.MIS, axis=0
        )
        classwise_dict = dict(
            sorted(
                classwise_dict.items(),
                key=lambda x: len(x[1]),
                reverse=True,
            )
        )

        fig = self.boxplot(classwise_dict)

        return classwise_dict, fig

    def true_positives(self) -> Tuple[Dict[int, Dict], bytes]:
        """Saves the TruePositives of the evaluator to the given output directory.

        Returns
        -------
        Dict[int, List]
            A dictionary mapping the class id to a list of dictionaries
            containing the images and metadata of the false positives.
        """
        classwise_dict = self.error_classwise_dict(
            NonError,
            color=[ErrorColor.WHITE, ErrorColor.TP],
            axis=1,
            get_bbox_func=get_both_bboxes,
        )
        classwise_dict = dict(
            sorted(
                classwise_dict.items(),
                key=lambda x: len(x[1]),
                reverse=True,
            )
        )

        fig = self.boxplot(classwise_dict)
        # fig = encode_base64(plt.Figure())

        return classwise_dict, fig

    def inspect(self):

        for evaluation in self.evaluator.evaluations:
            for error in evaluation.instances:
                pass

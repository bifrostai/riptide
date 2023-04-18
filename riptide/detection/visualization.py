from typing import Any, Callable, Dict, Iterable, List, Tuple, Type, Union

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from torchvision.io import read_image
from torchvision.ops.boxes import box_iou
from torchvision.transforms.functional import to_pil_image

from riptide.detection.characterization import (
    compute_aspect_variance,
    compute_size_variance,
)
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
from riptide.flow import FlowVisualizer
from riptide.report.section import Content, ContentType, Section
from riptide.utils.colors import ErrorColor, gradient
from riptide.utils.image import crop_preview, encode_base64, get_bbox_stats
from riptide.utils.logging import logger
from riptide.utils.plots import (
    PALETTE_BLUE,
    PALETTE_DARKER,
    PALETTE_GREEN,
    annotate_heatmap,
    boxplot,
    heatmap,
    histogram,
    plotly_markup,
    setup_mpl_params,
)


def get_both_bboxes(error: Error, bbox_attr: str):
    return torch.stack([error.gt_bbox, error.pred_bbox])


class Inspector:
    def __init__(
        self,
        evaluators: Union[ObjectDetectionEvaluator, List[ObjectDetectionEvaluator]],
    ):
        if isinstance(evaluators, list):
            assert (
                len({e.image_dir for e in evaluators}) == 1
            ), "Models should be evaluated on the same dataset."
        else:
            evaluators = [evaluators]

        self.evaluators = evaluators
        evaluator = evaluators[0]
        self.evaluator = evaluator
        self.errorlist_dicts = [
            evaluator.get_errorlist_dict() for evaluator in evaluators
        ]

        self.num_images = evaluator.num_images
        self.image_dir = evaluator.image_dir
        # NOTE: Assumes all models have the same conf_threshold and iou_threshold
        self.conf_threshold = round(evaluator.evaluations[0].conf_threshold, 2)
        self.iou_threshold = (
            round(evaluator.evaluations[0].bg_iou_threshold, 2),
            round(evaluator.evaluations[0].fg_iou_threshold, 2),
        )

        self.summaries = [
            {
                "name": evaluator.name,
                "conf_threshold": evaluator.conf_threshold,
                "iou_thresholds": evaluator.iou_thresholds,
                **{k: round(v, 3) for k, v in evaluator.summarize().items()},
            }
            for evaluator in self.evaluators
        ]

        self.classwise_summaries = {
            evaluator.name: {
                class_idx: {k: round(v, 3) for k, v in individual_summary.items()}
                for class_idx, individual_summary in evaluator.classwise_summarize().items()
            }
            for evaluator in self.evaluators
        }

        pred_crops: List[torch.Tensor] = []
        pred_errors: List[List[Error]] = []
        gt_crops: List[torch.Tensor] = []
        gt_errors: List[List[Error]] = []
        for evaluator in evaluators:
            model_crops, model_errors = evaluator.crop_objects()
            pred_crops.extend(model_crops)
            pred_errors.append(model_errors)
            model_crops, model_errors = evaluator.crop_objects(axis=0)
            gt_crops.extend(model_crops)
            gt_errors.append(model_errors)

        self.pred_projector = CropProjector(
            name=f"Predictions",
            images=pred_crops,
            encoder_mode="preconv",
            normalize_embeddings=True,
            labels=[
                (i, error.code) if error is not None else "NON"
                for i, model_errors in enumerate(pred_errors)
                for error in model_errors
            ],
            device=torch.device("cpu"),
        )
        self.gt_projector = CropProjector(
            name=f"Ground truths",
            images=gt_crops,
            encoder_mode="preconv",
            normalize_embeddings=True,
            labels=[
                (i, error.code) if error is not None else "NON"
                for i, model_errors in enumerate(gt_errors)
                for error in model_errors
            ],
            device=torch.device("cpu"),
        )

    def _confidence_hist(
        self, confidence_list: List[float], error_name: str = "Error"
    ) -> bytes:
        setup_mpl_params()
        fig, ax = plt.subplots(figsize=(6, 3), dpi=150, constrained_layout=True)
        histogram(confidence_list, bins=41, ax=ax)
        ax.set_title(f"{error_name} Confidence")
        ax.set_xlabel("Confidence score")
        ax.set_xlim(0.45, 1.05)
        ax.set_ylabel("Number of Occurences")
        return encode_base64(fig)

    @logger()
    def summary(self) -> Section:
        content = [
            {
                "No. of Images": self.num_images,
                "Conf. Threshold": self.conf_threshold,
                "IoU Threshold": f"{self.iou_threshold[0]} - {self.iou_threshold[1]}",
            },
            [None] * len(self.evaluators),
        ]
        for i, evaluator in enumerate(self.evaluators):
            summary = evaluator.summarize()

            counts = {
                "TP": summary.get("true_positives", 0),
                "FP": summary.get("false_positives", 0),
                "FN": summary.get("false_negatives", 0),
                "MIS": summary.get("MissedError", 0),
                "BKG": summary.get("BackgroundError", 0),
                "CLS": summary.get("ClassificationError", 0),
                "LOC": summary.get("LocalizationError", 0),
                "CLL": summary.get("ClassificationAndLocalizationError", 0),
                "DUP": summary.get("DuplicateError", 0),
            }
            counts["FN"] -= counts["MIS"]
            counts["FP"] -= sum(
                counts[code] for code in ["BKG", "CLS", "LOC", "CLL", "DUP"]
            )

            opacity = 0.8
            gt_bar = [
                (ErrorColor(code).rgb(opacity, False), counts[code], code)
                for code in ["TP", "MIS", "FN"]
            ]

            pred_bar = [
                (ErrorColor(code).rgb(opacity, False), counts[code], code)
                for code in ["TP", "BKG", "CLS", "LOC", "CLL", "DUP", "FP"]
            ]

            content[1][i] = (
                evaluator.name,
                {
                    "Precision": round(summary["precision"], 2),
                    "Recall": round(summary["recall"], 2),
                    "F1": round(summary["f1"], 2),
                    "Unused": summary["unused"],
                    "Ground Truths": {
                        "total": summary["total_count"],
                        "bar": [v for v in gt_bar if v[1] > 0],
                    },
                    "Predictions": {
                        "total": summary["true_positives"] + summary["false_positives"],
                        "bar": [v for v in pred_bar if v[1] > 0],
                    },
                },
            )

        section = Section(
            id="Overview",
            contents=[Content(type=ContentType.OVERVIEW, content=content)],
        )

        return section

    @logger()
    def overview(self) -> Tuple[dict, dict, Section]:

        evaluator_summary = self.evaluator.summarize()
        overall_summary = {
            "num_images": self.evaluator.num_images,
            "conf_threshold": self.evaluator.evaluations[0].conf_threshold,
            "bg_iou_threshold": self.evaluator.evaluations[0].bg_iou_threshold,
            "fg_iou_threshold": self.evaluator.evaluations[0].fg_iou_threshold,
        }
        overall_summary.update({k: round(v, 3) for k, v in evaluator_summary.items()})

        summary_section = self.summary()

        classwise_summary = self.evaluator.classwise_summarize()
        for class_idx, individual_summary in classwise_summary.items():
            for metric, value in individual_summary.items():
                classwise_summary[class_idx][metric] = round(value, 3)

        self.overall_summary = overall_summary
        self.classwise_summary = classwise_summary

        return overall_summary, classwise_summary, summary_section

    @logger()
    def error_confidence(
        self,
        error_types: Union[Type[Error], Iterable[Type[Error]]],
        evaluator_id: int = 0,
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
                for errors in self.errorlist_dicts[evaluator_id]
                .get(error_type.__name__, dict())
                .values()
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

    def boxplot(
        self, classwise_dict: Dict[int, Tuple[str, Dict[int, List[List[dict]]]]]
    ) -> bytes:
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
        area_info = {}
        for class_idx, (_, clusters) in classwise_dict.items():
            for cluster in clusters.values():
                # TODO: change this for multi model comparison
                areas = [m["bbox_area"] for info_dicts in cluster for m in info_dicts]
                area_info[class_idx] = areas

        setup_mpl_params()
        fig, ax = plt.subplots(figsize=(6, 6), dpi=150, constrained_layout=True)
        boxplot(area_info=area_info, ax=ax)

        return encode_base64(fig)

    def error_classwise_dict(
        self,
        error_type: Type[Error],
        color: Union[str, ErrorColor, List[Union[str, ErrorColor]]],
        axis: int = 0,
        *,
        evaluator_id: int = 0,
        preview_size: int = 128,
        bbox_attr: str = None,
        label_attr: str = None,
        projector_attr: str = None,
        get_bbox_func: Callable[[Tuple[Error, str]], torch.Tensor] = None,
        get_label_func: Callable[[int], str] = None,
        add_metadata_func: Callable[[dict, Error], dict] = None,
    ) -> Dict[int, Tuple[str, Dict[int, List[List[dict]]]]]:
        """Computes a dictionary of plots for the crops of the errors, classwise.

        Arguments
        ---------
        error_type : Type[Error]
            Error type to plot crops for

        color : Union[str, ErrorColor, List[Union[str, ErrorColor]]]
            Color(s) for the bounding box(es). Can be a single color or a list of colors

        axis : int, default=0
            Axis to crop image on. 0 for ground truth, 1 for predictions

        evaluator_id : int, default=0
            Evaluator id to use

        preview_size : int, default=128
            Size of the preview image

        bbox_attr : str, default=None
            Attribute name for the bounding box. If None, attribute is determined by `axis`

        label_attr : str, default=None
            Attribute name for the label. If None, attribute is determined by `axis`

        projector_attr : str, default=None
            Attribute name for the projector. If None, attribute is determined by `axis`

        get_bbox_func : Callable[[Tuple[Error, str]], torch.Tensor], default=None
            Function to get the bounding box from the error, by default a function that returns the bounding box specified by `bbox_attr`

        get_label_func : Callable[[int], str], default=None
            Function to get the label from the class index, by default a function that returns the class index as a string

        add_metadata_func : Callable[[dict], dict], default=None
            Function to add a caption to the metadata, by default a function that returns the metadata as is

        Returns
        -------
        Dict[int, Tuple[str, List[List[dict]]]]
            Dictionary of class ids to cluster ids to list of image metadata
        """

        # region: parse parameters
        error_type_name = (
            "true_positives" if error_type is NonError else error_type.__name__
        )

        num_errors = self.summaries[evaluator_id].get(error_type_name, 0)
        if num_errors == 0:
            return dict()

        errors_list = self.errorlist_dicts[evaluator_id].get(
            error_type.__name__, dict()
        )
        errors: Dict[str, List[Error]] = {}
        for image_path, error_list in errors_list.items():
            if image_path not in errors:
                errors[image_path] = error_list
            else:
                errors[image_path].extend(error_list)

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

        get_label_func = get_label_func or (lambda label: f"Predicted: Class {label}")

        if add_metadata_func is None:

            def add_metadata_func(metadata: dict, error: Error) -> dict:
                metadata.update(
                    {
                        "caption": " | ".join(
                            [
                                metadata["image_name"],
                                f"Conf { metadata['confidence'] }",
                                f"W{ metadata['bbox_width'] }",
                                f"H{ metadata['bbox_height'] }",
                            ]
                        ),
                    }
                )

                return metadata

        projector: CropProjector = getattr(self, projector_attr)

        code = "TP" if error_type is NonError else error_type.code

        def cluster_filter(labels: List[Tuple[int, str]]) -> List[bool]:
            return [label == (evaluator_id, code) for label in labels]

        # TODO: test robustness of clustering
        clusters = projector.cluster(label_mask_func=cluster_filter)

        # endregion

        classwise_dict: Dict[int, Tuple[str, Dict[int, List[List[dict]]]]] = {}
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
                    classwise_dict[label] = (
                        get_label_func(label),
                        dict(),
                    )

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

                cluster = clusters[idx].item()
                if cluster not in classwise_dict[label][1]:
                    classwise_dict[label][1][cluster] = [[]]

                classwise_dict[label][1][cluster][0].append(
                    add_metadata_func(
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
                            "cluster": clusters[
                                idx
                            ],  # TODO: remove this, it's redundant
                        },
                        error,
                    )
                )
                idx += 1

        for label in label_set:
            class_info, clusters = classwise_dict[label]

            classwise_dict[label] = (
                class_info,
                dict(sorted(clusters.items(), key=lambda x: x[0], reverse=True)),
            )

        return classwise_dict

    def error_classwise_ranking(
        self, error_type: Type[Error], display_type: str = "bar"
    ) -> bytes:
        labels = [
            (error.gt_label, error.pred_label)
            for errors in self.errorlist_dicts[0]
            .get(error_type.__name__, dict())
            .values()
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

        if display_type == "bar":
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
                        PALETTE_GREEN,
                        PALETTE_BLUE,
                        1 - (x / max(confusion_dict.values())),
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
        else:
            row_labels, col_labels = map(set, zip(*confusion_dict.keys()))
            confusion_matrix = np.array(
                [
                    [confusion_dict.get((i, j), 0) for j in col_labels]
                    for i in row_labels
                ]
            )

            fig, ax = plt.subplots(
                figsize=(np.max(confusion_matrix.shape),) * 2,
                dpi=150,
                constrained_layout=True,
            )
            im, cbar = heatmap(
                confusion_matrix,
                row_labels,
                col_labels,
                ax=ax,
                # grid_color=PALETTE_DARKER,
                cmap="YlGn",
                cbarlabel="No. of Occurences",
            )
            texts = annotate_heatmap(im, valfmt="{x:.0f}")

        return encode_base64(fig)

    @logger()
    def background_error(self, data: dict = None) -> Dict[int, List[Dict]]:
        """Saves the BackgroundErrors (false positives) of the evaluator to the given
        output directory.

        Parameters
        ----------
        data : dict, optional
            Metadata to attach to content, by default None

        Returns
        -------
        Dict[int, List]
            A dictionary mapping the class id to a list of dictionaries
            containing the images and metadata of the false positives.
        """
        if data is None:
            data = {}
        figs = self.error_classwise_dict(BackgroundError, color=ErrorColor.BKG, axis=1)

        return Section(
            id="BackgroundError",
            title="Background Errors",
            description=f"""
            List of all the false positive detections with confidence above the <span class="code">conf_threshold={ self.overall_summary["conf_threshold"] }</span> but do not pass the <span class="code">bg_iou_threshold={ self.overall_summary["bg_iou_threshold"] }</span>.
            """,
            contents=[
                Content(
                    type=ContentType.IMAGES,
                    header="Visualizations",
                    content=figs,
                    data=data,
                ),
            ],
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

        get_label_func = lambda x: f"Ground Truth: Class {x}"

        def add_metadata_func(x: dict, error: ClassificationError) -> dict:
            x.update(
                {
                    "caption": " | ".join(
                        [
                            x["image_name"],
                            f"Pred: Class {x['pred_class']}",
                            f"Conf { x['confidence'] }",
                        ]
                    ),
                }
            )

            return x

        classwise_dict = self.error_classwise_dict(
            ClassificationError,
            color=ErrorColor.CLS,
            axis=1,
            label_attr="gt_label",
            get_label_func=get_label_func,
            add_metadata_func=add_metadata_func,
        )
        fig = self.error_classwise_ranking(ClassificationError)

        return Section(
            id="ClassificationError",
            title="Classification Errors",
            description="""
                    List of all the false positive detections with <span class="code">iou > fg_iou_threshold</span> but with predicted classes not equal to the class of the corresponding ground truth.
                    """,
            contents=[
                Content(
                    type=ContentType.PLOT,
                    header="Ranking",
                    description=(
                        "The following plot shows the distribution of classification"
                        " errors."
                    ),
                    content=dict(plot=fig),
                ),
                Content(
                    type=ContentType.IMAGES,
                    header="Visualizations",
                    description="The following is a montage of the classification.",
                    content=classwise_dict,
                ),
            ],
        )

    @logger()
    def localization_error(self) -> Tuple[Dict[int, Dict], bytes]:
        """Saves the LocalizationErrors of the evaluator to the given output directory."""

        get_label_func = lambda x: f"Ground Truth: Class {x}"

        def add_metadata_func(x: dict, error: LocalizationError) -> dict:
            x.update(
                {
                    "caption": " | ".join(
                        [
                            x["image_name"],
                            f"W{x['bbox_width']}",
                            f"H{x['bbox_height']}",
                            f"IoU {x['iou']}",
                        ]
                    ),
                }
            )

            return x

        classwise_dict = self.error_classwise_dict(
            LocalizationError,
            color=[ErrorColor.WHITE, ErrorColor.LOC],
            axis=1,
            get_bbox_func=get_both_bboxes,
            preview_size=192,
            get_label_func=get_label_func,
            add_metadata_func=add_metadata_func,
        )
        # fig = self.error_classwise_ranking(LocalizationError)

        return Section(
            id="LocalizationError",
            title="Localization Errors",
            description="""
                    The following is a montage of the localization errors.
                    """,
            contents=[
                Content(
                    type=ContentType.IMAGES,
                    header="Visualizations",
                    content=classwise_dict,
                )
            ],
        )

    @logger()
    def classification_and_localization_error(self) -> Tuple[Dict[int, Dict], bytes]:
        """Saves the ClassificationAndLocalizationErrors of the evaluator to the given output directory.

        Returns
        -------
        Dict[int, List]
            A dictionary mapping the class id to a list of dictionaries
            containing the images and metadata of the false positives.
        """

        get_label_func = lambda x: f"Ground Truth: Class {x}"

        def add_metadata_func(
            x: dict, error: ClassificationAndLocalizationError
        ) -> dict:
            x.update(
                {
                    "caption": " | ".join(
                        [
                            x["image_name"],
                            f"Pred: Class {x['pred_class']}",
                            f"Conf { x['confidence'] }",
                            f"W{x['bbox_width']}",
                            f"H{x['bbox_height']}",
                            f"IoU {x['iou']}",
                        ]
                    ),
                }
            )

            return x

        classwise_dict = self.error_classwise_dict(
            ClassificationAndLocalizationError,
            color=[ErrorColor.WHITE, ErrorColor.CLL],
            axis=1,
            get_bbox_func=get_both_bboxes,
            preview_size=192,
            label_attr="gt_label",
            get_label_func=get_label_func,
            add_metadata_func=add_metadata_func,
        )
        fig = self.error_classwise_ranking(ClassificationAndLocalizationError)

        return Section(
            id="ClassificationAndLocalizationError",
            title="Classification and Localization Errors",
            description="""
                    List of all the false positive detections with <span class="code">bg_iou_threshold < iou < fg_iou_threshold</span> and with predicted classes not equal to the class of the corresponding ground truth.
                    """,
            contents=[
                Content(
                    type=ContentType.PLOT,
                    header="Ranking",
                    description=(
                        "The following plot shows the distribution of classification"
                        " and localization errors."
                    ),
                    content=dict(plot=fig),
                ),
                Content(
                    type=ContentType.IMAGES,
                    header="Visualizations",
                    description=(
                        "The following is a montage of the classification and"
                        " localization errors."
                    ),
                    content=classwise_dict,
                ),
            ],
        )

    @logger()
    def duplicate_error(self) -> Dict[int, Dict]:
        """Saves the DuplicateErrors of the evaluator to the given output directory.

        Returns
        -------
        Dict[int, List]
            A dictionary mapping the class id to a list of dictionaries
            containing the images and metadata of the false positives.
        """

        get_label_func = lambda x: f"Ground Truth: Class {x}"

        def get_bbox_func(error: DuplicateError, attr: str):
            return torch.stack([error.gt_bbox, error.best_pred_bbox, error.pred_bbox])

        def add_metadata_func(x: dict, error: DuplicateError) -> dict:
            best_iou = round(
                box_iou(
                    error.best_pred_bbox.unsqueeze(0), error.gt_bbox.unsqueeze(0)
                ).item(),
                3,
            )
            best_conf = round(error.best_confidence, 2)
            x.update(
                {
                    "best_iou": best_iou,
                    "best_conf": best_conf,
                    "caption": " | ".join(
                        [
                            x["image_name"],
                            f"IoU, Conf ({ x['iou'] }, {x['confidence']})",
                            f"Best  ({ best_iou }, { best_conf })",
                            f"W{x['bbox_width']}",
                            f"H{x['bbox_height']}",
                        ]
                    ),
                }
            )

            return x

        classwise_dict = self.error_classwise_dict(
            DuplicateError,
            color=[ErrorColor.WHITE, ErrorColor.TP, ErrorColor.DUP],
            axis=1,
            get_bbox_func=get_bbox_func,
            preview_size=192,
            add_metadata_func=add_metadata_func,
            get_label_func=get_label_func,
        )

        return Section(
            id="DuplicateError",
            title="Duplicate Errors",
            description=f"""
                        List of all the detections with confidence above the <span class="code">conf_threshold={self.overall_summary["conf_threshold"]}</span> but lower than the confidence of another true positive prediction.
                        """,
            contents=[
                Content(
                    type=ContentType.IMAGES,
                    header="Visualizations",
                    description=(
                        "The following is a montage of the duplicate errors. White"
                        " boxes indicate ground truths, green boxes indicate best"
                        " predictions, and red boxes indicate duplicate predictions."
                    ),
                    content=classwise_dict,
                ),
            ],
        )

    @logger()
    def missed_error(self) -> Tuple[Dict[int, Dict], bytes]:
        """Saves the MissedErrors of the evaluator to the given output directory.

        Returns
        -------
        Dict[int, List]
            A dictionary mapping the class id to a list of dictionaries
            containing the images and metadata of the false positives.
        """
        get_label_func = lambda x: f"Missed: Class {x}"

        def add_metadata_func(x: dict, error: MissedError) -> dict:
            x.update(
                {
                    "caption": " | ".join(
                        [
                            x["image_name"],
                            f"W{x['bbox_width']}",
                            f"H{x['bbox_height']}",
                        ]
                    ),
                }
            )

            return x

        classwise_dict = self.error_classwise_dict(
            MissedError,
            color=ErrorColor.MIS,
            axis=0,
            get_label_func=get_label_func,
            add_metadata_func=add_metadata_func,
        )

        fig = self.boxplot(classwise_dict)

        missed_size_var = compute_size_variance(self.evaluator)
        missed_aspect_var = compute_aspect_variance(self.evaluator)

        return Section(
            id="MissedError",
            title="Missed Errors",
            description="""
                    List of all the ground truths that have no corresponding prediction.
                    """,
            contents=[
                # Content(
                #     type=ContentType.RECALL,
                #     header="Classwise Missed Errors and recall",
                #     description="Number of Missed Errors per class | number of total objects per class",
                #     content=[self.classwise_summary, "MissedError"],
                # ),
                # Content(
                #     type=ContentType.AR_SIZE,
                #     header="Aspect ratio and size variance",
                #     description="Variance of aspect ratios across Missed Errors",
                #     content=[missed_aspect_var, missed_size_var],
                # ),
                Content(
                    type=ContentType.PLOT,
                    header="Ranking",
                    description="""
                    The following plot shows the size (area) distribution of missed errors for each class. The presence of outliers in this plot indicates the misses may be caused by extreme object sizes.
                    """,
                    content=dict(plot=fig),
                ),
                Content(
                    type=ContentType.IMAGES,
                    header="Visualizations",
                    description=f"""
                    List of all the false negative detections. No prediction was made above <span class="code">conf_threshold={ self.overall_summary['conf_threshold'] }</span> that had IoU above <span class="code">bg_iou_threshold={ self.overall_summary["bg_iou_threshold"] }</span> (otherwise it would be considered a Localization Error).
                    """,
                    content=classwise_dict,
                ),
            ],
        )

    @logger()
    def true_positives(self) -> Tuple[Dict[int, Dict], bytes]:
        """Saves the TruePositives of the evaluator to the given output directory.

        Returns
        -------
        Dict[int, List]
            A dictionary mapping the class id to a list of dictionaries
            containing the images and metadata of the false positives.
        """
        get_label_func = lambda x: f"Ground Truth: Class {x}"

        def add_metadata_func(x: dict, error: NonError) -> dict:
            x.update(
                {
                    "caption": " | ".join(
                        [
                            x["image_name"],
                            f"Conf {x['confidence']}",
                            f"IoU {x['iou']}",
                        ]
                    ),
                }
            )

            return x

        classwise_dict = self.error_classwise_dict(
            NonError,
            color=[ErrorColor.WHITE, ErrorColor.TP],
            axis=1,
            get_bbox_func=get_both_bboxes,
            get_label_func=get_label_func,
            add_metadata_func=add_metadata_func,
        )

        return Section(
            id="TruePositive",
            title="True Positives",
            description=f"""
                    List of all the True Positives detections. No prediction was made below <span class="code">conf_threshold={ self.overall_summary["conf_threshold"] }</span>.
                    """,
            contents=[
                Content(
                    type=ContentType.IMAGES,
                    header="Visualizations",
                    content=classwise_dict,
                ),
            ],
        )

    @logger()
    def inspect(self) -> Dict[str, Any]:
        """Generate figures and plots for the errors.

        Returns
        -------
        Dict
            A dictionary containing the generated figures and plots.
        """

        results = dict()

        results["background_error_figs"] = self.background_error()
        results["classification_error_figs"] = self.classification_error()
        results["localization_error_figs"] = self.localization_error()
        results[
            "classification_and_localization_error_figs"
        ] = self.classification_and_localization_error()
        results["duplicate_error_figs"] = self.duplicate_error()
        results["missed_error_figs"] = self.missed_error()
        results["true_positive_figs"] = self.true_positives()

        return results

    @logger()
    def compare_background_errors(self) -> Section:
        """Saves the BackgroundErrors (false positives) of the evaluators to the given
        output directory.

        Returns
        -------
        Dict[int, List]
            A dictionary mapping the class id to a list of dictionaries
            containing the images and metadata of the false positives.
        """
        figs = [
            self.error_classwise_dict(
                BackgroundError, color=ErrorColor.BKG, axis=1, evaluator_id=idx
            )
            for idx in range(len(self.evaluators))
        ]

        return Section(
            id="BackgroundError",
            title="Background Errors",
            description=f"""
            List of all the false positive detections with confidence above the <span class="code">conf_threshold={ self.overall_summary["conf_threshold"] }</span> but do not pass the <span class="code">bg_iou_threshold={ self.overall_summary["bg_iou_threshold"] }</span>.
            """,
            contents=[
                Content(
                    type=ContentType.IMAGES,
                    header=f"Errors in {self.evaluators[i].name}",
                    content=fig,
                )
                for i, fig in enumerate(figs)
            ],
        )

    @logger()
    def flow(self) -> Section:
        """Saves the flow of the evaluators to the given output directory.

        Returns
        -------
        Dict[int, List]
            A dictionary mapping the class id to a list of dictionaries
            containing the images and metadata of the false positives.
        """
        fig = plotly_markup(
            FlowVisualizer(self.evaluators, None, self.image_dir).visualize()
        )

        return Section(
            id="Flow",
            title="Flow",
            description=f"""
            A visual representation of the flow of the ground truth statuses between models.
            """,
            contents=[
                Content(
                    type=ContentType.PLOT,
                    content=dict(plot=fig, interactive=True),
                )
            ],
        )

    @logger()
    def compare(self) -> Dict[str, Section]:
        """Generate figures and plots for the errors.

        Returns
        -------
        Dict
            A dictionary containing the generated figures and plots.
        """

        results = dict()

        _, _, results["overview"] = self.overview()
        results["background_errors"] = self.compare_background_errors()
        results["flow"] = self.flow()
        return results

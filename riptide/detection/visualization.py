import logging
import math
from typing import Callable, Dict, Iterable, List, Set, Tuple, Type, Union

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.nn.functional import conv2d
from torchvision.io import read_image
from torchvision.transforms import Grayscale
from torchvision.transforms.functional import crop
from tqdm import tqdm

from riptide.detection.confusions import Confusion
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
from riptide.utils.colors import ErrorColor
from riptide.utils.crops import (
    add_metadata,
    generate_fig,
    get_bbox_by_attr,
    get_crop_options,
)
from riptide.utils.enums import ErrorWeights
from riptide.utils.image import encode_base64
from riptide.utils.logging import logger
from riptide.utils.plots import (
    annotate_heatmap,
    boxplot,
    heatmap,
    histogram,
    plotly_markup,
    setup_mpl_params,
)

ALL_ERRORS = [
    "ClassificationError",
    "LocalizationError",
    "ClassificationAndLocalizationError",
    "DuplicateError",
    "MissedError",
    "BackgroundError",
]


def empty_section(section_id: str, title: str, description: str = None):
    return Section(
        id=section_id,
        title=title,
        contents=[Content(type=ContentType.TEXT, content=[description])],
    )


class Inspector:
    @logger("Initializing Inspector", "Initialized Inspector")
    def __init__(
        self,
        evaluators: Union[ObjectDetectionEvaluator, List[ObjectDetectionEvaluator]],
    ):
        if isinstance(evaluators, list):
            assert (
                len({e.image_dir for e in evaluators}) == 1
            ), "Models should be evaluated on the same image directory."
        else:
            evaluators = [evaluators]

        self.evaluators = evaluators
        evaluator = evaluators[0]
        self.errorlist_dicts = [
            evaluator.get_errorlist_dict() for evaluator in evaluators
        ]

        self.num_images = evaluator.num_images
        self.image_dir = evaluator.image_dir
        self.categories = evaluator.categories
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

        self.classwise_summaries = [
            {
                "name": evaluator.name,
                **{
                    class_idx: {k: round(v, 3) for k, v in individual_summary.items()}
                    for class_idx, individual_summary in evaluator.classwise_summarize().items()
                },
            }
            for evaluator in self.evaluators
        ]
        bkg_crops: List[torch.Tensor] = []
        bkg_errors: List[List[Error]] = []

        for evaluator in evaluators:
            model_crops, model_errors = evaluator.crop_objects(by_type=BackgroundError)
            bkg_crops.extend(model_crops)
            bkg_errors.append(model_errors)

        bkg_labels = [
            (i, error.pred_label)
            for i, model_errors in enumerate(bkg_errors)
            for error in model_errors
        ]

        self.gt_data = evaluators[0].get_gt_data()
        actual_labels = [(-1, label) for label in self.gt_data.gt_labels.tolist()]
        repeat_labels = [(-2, label) for label in self.gt_data.gt_labels.tolist()]
        self.projector = CropProjector(
            name=f"Crops",
            images=self.gt_data.crops + bkg_crops + self.gt_data.crops,
            encoder_mode="preconv",
            normalize_embeddings=True,
            labels=actual_labels + bkg_labels + repeat_labels,
            device=torch.device("cpu"),
        )

        self.clusters = self.projector.subcluster()

        self.overview()

        self.crops = {}
        self._generated_crops = set()

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

    def recalculate_summaries(
        self, ids: List[int] = None, *, weights: Union[dict, ErrorWeights] = None
    ) -> dict:
        summaries = (
            self.summaries
            if ids is None
            else [self.summaries[i] for i in ids if 0 <= i < len(self.summaries)]
        )

        if weights is not None:
            if isinstance(weights, ErrorWeights):
                weights = weights.weights

            tp_weight = weights.get("true_positives", 1)
            fn_weight = weights.get("false_negatives", 1)
            fp_weight = weights.get("false_positives", 1)

        for summary in summaries:
            tp = summary.get("true_positives", 0)
            fn = summary.get("false_negatives", 0)
            fp = sum(
                [
                    summary.get(error_name, 0)
                    for error_name in ALL_ERRORS
                    if error_name != "MissedError"
                ]
            )

            precision = tp / (tp + fp + 1e-7)
            recall = tp / (tp + fn + 1e-7)
            f1 = 2 * precision * recall / (precision + recall + 1e-7)

            summary.update(
                {
                    "false_positives": fp,
                    "precision": precision,
                    "recall": recall,
                    "f1": f1,
                }
            )

            if weights is not None:
                tp *= tp_weight
                fn *= fn_weight

                weighted_errors = {
                    error_name: summary.get(error_name, 0)
                    * weights.get(error_name, fp_weight)
                    if error_name != "MissedError"
                    else summary.get(error_name, 0) * weights.get(error_name, fn_weight)
                    for error_name in ALL_ERRORS
                }

                fp = sum(
                    [
                        weighted_errors.get(error_name, 0)
                        for error_name in ALL_ERRORS
                        if error_name != "MissedError"
                    ]
                )

                precision = tp / (tp + fp + 1e-7)
                recall = tp / (tp + fn + 1e-7)
                f1 = 2 * precision * recall / (precision + recall + 1e-7)

                summary.update(
                    {
                        "weighted": {
                            "true_positives": tp,
                            "false_negatives": fn,
                            "false_positives": fp,
                            "precision": precision,
                            "recall": recall,
                            "f1": f1,
                            **weighted_errors,
                        }
                    }
                )

        if weights is not None:
            return {
                "TP": tp_weight,
                "FN": fn_weight,
                "FP": fp_weight,
                "BKG": weights.get("BackgroundError", fp_weight),
                "CLS": weights.get("ClassificationError", fp_weight),
                "LOC": weights.get("LocalizationError", fp_weight),
                "CLL": weights.get("ClassificationAndLocalizationError", fp_weight),
                "MIS": weights.get("MissedError", fn_weight),
                "DUP": weights.get("DuplicateError", fp_weight),
            }

    @logger()
    def summary(
        self, ids: List[int] = None, *, display_weights: bool = False
    ) -> Section:
        """Generate a summary of the evaluation results for each model."""

        code_mapping = {
            "TP": "True Positives",
            "FN": "False Negatives",
            "FP": "False Positives",
            "BKG": "Background",
            "CLS": "Classification",
            "LOC": "Localization",
            "CLL": "Classification and Localization",
            "MIS": "Missed",
            "DUP": "Duplicate",
        }

        evaluators_and_summaries = (
            list(zip(self.evaluators, self.summaries))
            if ids is None
            else [
                (self.evaluators[i], self.summaries[i])
                for i in ids
                if 0 <= i < len(self.evaluators)
            ]
        )

        content = [
            {
                "No. of Images": (self.num_images, None),
                "No. of Objects": (len(self.gt_data), None),
                "Conf. Threshold": (self.conf_threshold, None),
                "IoU Threshold": (
                    f"{self.iou_threshold[0]} - {self.iou_threshold[1]}",
                    None,
                ),
            },
            [None] * len(evaluators_and_summaries),
        ]
        for i, (evaluator, summary) in enumerate(evaluators_and_summaries):
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

            precision = round(summary["precision"], 2)
            recall = round(summary["recall"], 2)
            f1 = round(summary["f1"], 2)

            precision_tooltip = None
            recall_tooltip = None
            f1_tooltip = None

            weights = {}

            if display_weights and "weighted" in summary:
                weighted: dict = summary["weighted"]
                precision = (
                    f" {round(weighted['precision'], 2)} <span class='text-dark"
                    f" text-xs'>| {precision}</span>"
                )
                recall = (
                    f" {round(weighted['recall'], 2)} <span class='text-dark text-xs'>|"
                    f" {recall}</span>"
                )
                f1 = (
                    f" {round(weighted['f1'], 2)} <span class='text-dark text-xs'>|"
                    f" {f1}</span>"
                )

                for code, key in zip(
                    ["TP", "FP", "CLS", "LOC", "CLL", "DUP", "MIS", "BKG"],
                    ["true_positives", "false_positives", *ALL_ERRORS],
                ):
                    weights[code] = (
                        round(weighted[key] / counts[code], 2)
                        if counts[code] > 0
                        else 0
                    )

                weights["FN"] = (
                    round(
                        weighted["false_negatives"] / (counts["FN"] + counts["MIS"]), 2
                    )
                    if (counts["FN"] + counts["MIS"]) > 0
                    else 0
                )

                precision_tooltip = "Weighted | Unweighted"
                recall_tooltip = "Weighted | Unweighted"
                f1_tooltip = "Weighted | Unweighted"

            opacity = 0.8
            gt_bar = [
                (
                    ErrorColor(code).rgb(opacity, False),
                    counts[code],
                    code_mapping[code],
                    weights.get(code),
                )
                for code in ["TP", "MIS", "FN"]
            ]

            pred_bar = [
                (
                    ErrorColor(code).rgb(opacity, False),
                    counts[code],
                    code_mapping[code],
                    weights.get(code),
                )
                for code in ["TP", "BKG", "CLS", "LOC", "CLL", "DUP", "FP"]
            ]
            content[1][i] = (
                evaluator.name,
                {
                    "Precision": (precision, precision_tooltip),
                    "Recall": (recall, recall_tooltip),
                    "F1": (f1, f1_tooltip),
                    "Unused": (
                        summary["unused"],
                        "No. of detections below conf. threshold",
                    ),
                    "Ground Truths": {
                        "total": summary["total_count"],
                        "bar": [v for v in gt_bar if v[1] > 0],
                    },
                    "Predictions": {
                        "total": summary["true_positives"] + summary["false_positives"],
                        "bar": [v for v in pred_bar if v[1] > 0],
                        "tooltip": (
                            "Similar false positive detections are suppressed from"
                            " count."
                        ),
                    },
                },
            )

        section = Section(
            id="Overview",
            contents=[Content(type=ContentType.OVERVIEW, content=content)],
        )

        return section

    @logger()
    def overview(self, evaluator_id: int = 0) -> Tuple[dict, dict, Section]:

        evaluator = self.evaluators[evaluator_id]
        evaluator_summary = evaluator.summarize()
        overall_summary = {
            "num_images": evaluator.num_images,
            "conf_threshold": evaluator.evaluations[0].conf_threshold,
            "bg_iou_threshold": evaluator.evaluations[0].bg_iou_threshold,
            "fg_iou_threshold": evaluator.evaluations[0].fg_iou_threshold,
        }
        overall_summary.update({k: round(v, 3) for k, v in evaluator_summary.items()})

        summary_section = self.summary()

        classwise_summary = evaluator.classwise_summarize()
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
        self,
        classwise_dict: Dict[int, Tuple[str, Dict[int, List[List[dict]]]]],
        *,
        threshold: bool = False,
    ) -> bytes:
        """Generate a boxplot of the area of the bounding boxes for each class.

        Parameters
        ----------
        classwise_dict : Dict[int, Tuple[str, Dict[int, List[List[dict]]]]]
            Dictionary of classwise information, generated by `error_classwise_dict`

        Returns
        -------
        bytes
            Encoded base64 image of the boxplot
        """
        # Plot the barplots of the classwise area
        area_info = {}
        for class_idx, (_, clusters) in classwise_dict.items():
            areas = []
            for cluster in clusters.values():
                # TODO: change this for multi model comparison
                areas.extend(
                    [m["bbox_area"] for info_dicts in cluster for m in info_dicts]
                )
            area_info[class_idx] = np.array(areas)

        if threshold:
            thresholds = [16, 32, 96, 288]
            quantiles = [0.0] + [t**2 for t in thresholds]
        else:
            quantiles = None

        setup_mpl_params()
        fig, ax = plt.subplots(figsize=(6, 6), dpi=150, constrained_layout=True)
        boxplot(area_info=area_info, ax=ax, quantiles=quantiles)
        ax.set_yscale("log")
        ax.set_ylabel("Area (px)")
        ax.set_xlabel("Class")

        return encode_base64(fig)

    def error_classwise_dict(
        self,
        error_type: Type[Error],
        color: Union[str, ErrorColor, List[Union[str, ErrorColor]]],
        axis: int = 0,
        *,
        evaluator_id: int = 0,
        preview_size: int = 192,
        bbox_attr: str = None,
        label_attr: str = None,
        get_bbox_func: Callable[[Tuple[Error, str]], torch.Tensor] = None,
        label_str: str = "Predicted: {label}",
        add_metadata_func: Callable[[dict, Error], dict] = None,
        clusters: torch.Tensor = None,
    ) -> Dict[int, Tuple[str, Dict[int, List[List[dict]]]]]:
        """Compute a dictionary of plots for the crops of the errors, classwise.

        Arguments
        ---------
        error_type : Type[Error]
            Error type to plot crops for

        color : Union[str, ErrorColor, List[Union[str, ErrorColor]]]
            Color(s) for the bounding box(es). Can be a single color or a list of colors

        axis : int, default=0
            Axis to perform computations. 0 for ground truths, 1 for predictions

        evaluator_id : int, default=0
            Evaluator id to use

        preview_size : int, default=128
            Size of the preview image

        bbox_attr : str, default=None
            Attribute name for the bounding box. If None, attribute is determined by `axis`

        label_attr : str, default=None
            Attribute name for the label. If None, attribute is determined by `axis`

        get_bbox_func : Callable[[Tuple[Error, str]], torch.Tensor], default=None
            Function to get the bounding box from the error, by default a function that returns the bounding box specified by `bbox_attr`

        label_str : str, default="Predicted: {label}"
            String to use for the label

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

        errors = self.errorlist_dicts[evaluator_id].get(error_type.__name__, dict())

        if not isinstance(color, list):
            color = [color]

        color = [c.colorstr if isinstance(c, ErrorColor) else c for c in color]

        attr_prefix = "gt" if axis == 0 else "pred"
        bbox_attr = bbox_attr or f"{attr_prefix}_bbox"
        label_attr = label_attr or f"{attr_prefix}_label"

        get_bbox_func = get_bbox_func or get_bbox_by_attr
        add_metadata_func = add_metadata_func or add_metadata

        if clusters is None:
            projector = self.projector
            mask = (
                [label[0] == -1 for label in projector.labels]
                if error_type is not BackgroundError
                else [label[0] == evaluator_id for label in projector.labels]
            )

            clusters = self.clusters[mask]

        else:
            assert (
                clusters.shape[0] == num_errors
            ), "Number of clusters does not match number of errors"

        # endregion

        subclusters = {}

        if error_type is BackgroundError:

            def get_cluster(idx, bkg_idx):
                return tuple(clusters[bkg_idx].tolist())

        else:

            def get_cluster(idx, bkg_idx):
                return tuple(clusters[idx].tolist())

        classwise_dict: Dict[int, Tuple[str, Dict[int, List[List[dict]]]]] = {}
        label_set = set()
        bkg_idx = 0
        count = 0
        for image_path, image_errors in errors.items():
            image_tensor = read_image(image_path)
            image_name = image_path.split("/")[-1]
            for error in image_errors:
                label: int = getattr(error, label_attr)
                label_set.add(label)

                if label not in classwise_dict:
                    classwise_dict[label] = (
                        label_str.format(
                            label=self.categories.get(label, f" Class {label}")
                        ),
                        dict(),
                    )

                cluster = get_cluster(error.idx, bkg_idx)
                if cluster[0] not in classwise_dict[label][1]:
                    classwise_dict[label][1][cluster[0]] = [[]]

                fig = generate_fig(
                    image_tensor=image_tensor,
                    image_name=image_name,
                    error=error,
                    color=color,
                    bbox_attr=bbox_attr,
                    cluster=cluster,
                    preview_size=preview_size,
                    get_bbox_func=get_bbox_func,
                    add_metadata_func=add_metadata_func,
                )

                crop_key = (evaluator_id, error)
                if evaluator_id not in self._generated_crops and crop_key in self.crops:
                    logging.debug(f"Generated duplicate crop: {crop_key}")

                self.crops[crop_key] = fig
                bkg_idx += 1

                unique_key = (*cluster, error.gt_label, error.pred_label)

                is_repeated_non_outlier = (
                    -1 not in cluster and unique_key in subclusters
                )
                is_repeated_gt = (
                    not isinstance(error, BackgroundError)
                    and unique_key in subclusters
                    and (*unique_key, error.idx) in subclusters[unique_key]["uniques"]
                )

                if is_repeated_non_outlier or is_repeated_gt:
                    subclusters[unique_key]["similar"].append(error)
                    if (*unique_key, error.idx) in subclusters[unique_key]["uniques"]:
                        count += 1
                    else:
                        subclusters[unique_key]["uniques"].add((*unique_key, error.idx))
                else:
                    subclusters[unique_key] = fig

                    classwise_dict[label][1][cluster[0]][0].append(fig)
                    count += 1

        if error_type not in [NonError, MissedError]:
            self.summaries[evaluator_id][error_type_name] = count

        for label in label_set:
            class_info, clusters_dict = classwise_dict[label]

            classwise_dict[label] = (
                class_info,
                dict(sorted(clusters_dict.items(), key=lambda x: x[0], reverse=True)),
            )

        return classwise_dict

    def error_classwise_ranking(
        self,
        error_types: List[Type[Error]],
        evaluator_id: int = 0,
        *,
        title: str = "Classwise Error Ranking",
        display_type: str = "bar",
        combine: bool = False,
        confusion: Confusion = Confusion.FALSE_POSITIVE,
        encoded: bool = True,
    ) -> bytes:
        if not isinstance(error_types, list):
            error_types = [error_types]

        confusion_matrices = [
            self.evaluators[evaluator_id].get_confusions()[error_type.code]
            for error_type in error_types
        ]

        confusion_dict = {}
        for i, confusion_matrix in enumerate(confusion_matrices):
            for pair, counts in confusion_matrix.items():
                if pair not in confusion_dict:
                    confusion_dict[pair] = [0 for _ in range(len(error_types))]
                confusion_dict[pair][i] += counts[confusion.value]

        if combine:
            confusion_dict = {k: [sum(v)] for k, v in confusion_dict.items()}

        confusion_dict = dict(
            sorted(confusion_dict.items(), key=lambda x: sum(x[1]), reverse=True)
        )

        setup_mpl_params()

        if display_type == "bar":
            fig, ax = plt.subplots(
                figsize=(5, np.ceil(len(confusion_dict) * 0.4)),
                dpi=150,
                constrained_layout=True,
            )
            for i, error_type in enumerate(error_types):
                ax.barh(
                    range(len(confusion_dict)),
                    width=[x[i] for x in confusion_dict.values()],
                    color=ErrorColor[error_type.code].hex,
                    left=[sum(x[:i]) for x in confusion_dict.values()],
                    label=error_type.code,
                )
            ax.set_yticks(range(len(confusion_dict)))
            ax.set_yticklabels(
                [
                    f"gt={self.categories.get(k[0], f'Class {k[0]}')} pred={self.categories.get(k[1], f'Class {k[1]}')}"
                    for k in confusion_dict.keys()
                ],
                minor=False,
            )
            ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0))
            ax.set_title(title)
            ax.set_xlabel("Number of Occurences")
        else:
            row_labels, col_labels = map(set, zip(*confusion_dict.keys()))
            confusion_matrix = np.array(
                [
                    [
                        confusion_dict.get((i, j), [0 for _ in error_types])
                        for j in col_labels
                    ]
                    for i in row_labels
                ]
            )

            fig, ax = plt.subplots(
                figsize=(np.max(confusion_matrix.shape),) * 2,
                dpi=150,
                constrained_layout=True,
            )
            im, cbar = heatmap(
                confusion_matrix.sum(axis=2),
                row_labels,
                col_labels,
                ax=ax,
                # grid_color=PALETTE_DARKER,
                cmap="YlGn",
                cbarlabel="No. of Occurences",
            )
            texts = annotate_heatmap(im, valfmt="{x:.0f}")

        if encoded:
            return encode_base64(fig)
        else:
            return fig, ax

    # region: Error Sections
    @logger()
    def background_error(self, *, data: dict = None, **kwargs) -> Section:
        """Generate a section visualizing the background errors in the dataset.

        Parameters
        ----------
        data : dict, optional
            Metadata to attach to content, by default None
        clusters : torch.Tensor, optional
            The cluster assignments for each error, if None then clusters are computed by `self.projector`, by default None

        kwargs : dict
            Additional keyword arguments to pass to `error_classwise_dict`

        Returns
        -------
        Section
            The section containing the visualizations
        """
        section_id = "BackgroundError"
        title = "Background Errors"
        description = f"""
            List of all the detections with confidence above the <span class="code">conf_threshold={ self.overall_summary["conf_threshold"] }</span> but do not pass the <span class="code">bg_iou_threshold={ self.overall_summary["bg_iou_threshold"] }</span>.
            """

        if self.summaries[kwargs.get("evaluator_id", 0)]["BackgroundError"] == 0:
            return empty_section(
                section_id=section_id,
                title=title,
                description=f"""
                <p>{description}</p>
                <p>No {title.lower()} were found.</p>
                """,
            )

        if data is None:
            data = {}
        data["grouped"] = data.get("grouped", True)
        data["compact"] = data.get("compact", True)

        get_crop_options(BackgroundError, kwargs)

        figs = self.error_classwise_dict(**kwargs)

        return Section(
            id=section_id,
            title=title,
            description=description,
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
    def classification_error(self, **kwargs: dict) -> Section:
        """Generate a section visualizing the classification errors in the dataset.

        Parameters
        ----------
        kwargs : dict
            Additional keyword arguments to pass to `error_classwise_dict`

        Returns
        -------
        Section
            The section containing the visualizations
        """

        section_id = "ClassificationError"
        title = "Classification Errors"
        description = """
        List of all the detections with <span class="code">iou > fg_iou_threshold</span> but with predicted classes not equal to the class of the corresponding ground truth.
        """

        if self.summaries[kwargs.get("evaluator_id", 0)]["ClassificationError"] == 0:
            return empty_section(
                section_id=section_id,
                title=title,
                description=f"""
                <p>{description}</p>
                <p>No {title.lower()} were found.</p>
                """,
            )

        get_crop_options(ClassificationError, kwargs)

        classwise_dict = self.error_classwise_dict(**kwargs)
        fig = self.error_classwise_ranking(ClassificationError)

        contents = [
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
        ]

        return Section(
            id="ClassificationError",
            title="Classification Errors",
            description="""
                    List of all the false positive detections with <span class="code">iou > fg_iou_threshold</span> but with predicted classes not equal to the class of the corresponding ground truth.
                    """,
            contents=contents,
        )

    @logger()
    def localization_error(self, **kwargs: dict) -> Section:
        """Generate a section visualizing the localization errors in the dataset.

        Parameters
        ----------
        kwargs : dict
            Additional keyword arguments to pass to `error_classwise_dict`

        Returns
        -------
        Section
            The section containing the visualizations
        """

        section_id = "LocalizationError"
        title = "Localization Errors"
        description = """
        List of all the detections with predicted classes equal to the class of the corresponding ground truth but with <span class="code">bg_iou_threshold < iou < fg_iou_threshold</span>.
        """

        if self.summaries[kwargs.get("evaluator_id", 0)]["LocalizationError"] == 0:
            return empty_section(
                section_id=section_id,
                title=title,
                description=f"""
                <p>{description}</p>
                <p>No {title.lower()} were found.</p>
                """,
            )

        get_crop_options(LocalizationError, kwargs)

        classwise_dict = self.error_classwise_dict(**kwargs)

        return Section(
            id=section_id,
            title=title,
            description=description,
            contents=[
                Content(
                    type=ContentType.IMAGES,
                    header="Visualizations",
                    content=classwise_dict,
                )
            ],
        )

    @logger()
    def classification_and_localization_error(self, **kwargs: dict) -> Section:
        """Generate a section visualizing the classification and localization errors in the dataset.

        Parameters
        ----------
        kwargs : dict
            Additional keyword arguments to pass to `error_classwise_dict`

        Returns
        -------
        Section
            The section containing the visualizations
        """

        section_id = "ClassificationAndLocalizationError"
        title = "Classification and Localization Errors"
        description = """
        List of all the detections with <span class="code">bg_iou_threshold < iou < fg_iou_threshold</span> and with predicted classes not equal to the class of the corresponding ground truth.
        """

        if (
            self.summaries[kwargs.get("evaluator_id", 0)][
                "ClassificationAndLocalizationError"
            ]
            == 0
        ):
            return empty_section(
                section_id=section_id,
                title=title,
                description=f"""
                <p>{description}</p>
                <p>No {title.lower()} were found.</p>
                """,
            )

        get_crop_options(ClassificationAndLocalizationError, kwargs)

        classwise_dict = self.error_classwise_dict(**kwargs)
        fig = self.error_classwise_ranking(ClassificationAndLocalizationError)

        return Section(
            id=section_id,
            title=title,
            description=description,
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
    def confusions(self, **kwargs: dict) -> Section:
        """Generate a section visualizing the classification type errors in the dataset.

        Parameters
        ----------
        kwargs : dict
            Additional keyword arguments to pass to `error_classwise_dict`

        Returns
        -------
        Section
            The section containing the visualizations
        """

        section_id = "Confusions"
        title = "Confusions"
        description = """
        List of all the detections with predicted classes not equal to the class of the corresponding ground truth.
        """

        num_errors = (
            self.summaries[kwargs.get("evaluator_id", 0)]["ClassificationError"]
            + self.summaries[kwargs.get("evaluator_id", 0)][
                "ClassificationAndLocalizationError"
            ]
        )

        if num_errors == 0:
            return empty_section(
                section_id=section_id,
                title=title,
                description=f"""
                <p>{description}</p>
                <p>No {title.lower()} were found.</p>
                """,
            )

        get_crop_options(ClassificationError, kwargs)

        cls_classwise_dict = self.error_classwise_dict(**kwargs)

        get_crop_options(ClassificationAndLocalizationError, kwargs)

        cll_classwise_dict = self.error_classwise_dict(**kwargs)
        fig = self.error_classwise_ranking(
            [ClassificationError, ClassificationAndLocalizationError]
        )

        # regroup by confusion pairs
        def regroup_dict(
            classwise_dict: Dict[int, Tuple[str, Dict[int, List[List[dict]]]]],
            source_dict: Dict[int, Tuple[str, Dict[int, List[List[dict]]]]],
        ) -> None:
            for _, (_, clusters) in source_dict.items():
                for cluster, error_figs in clusters.items():
                    for error_fig in error_figs[0]:
                        confusion_idx = (error_fig["gt_class"], error_fig["pred_class"])
                        if confusion_idx not in classwise_dict:
                            labels = self.categories.get(
                                confusion_idx[0], f"Class {confusion_idx[0]}"
                            ), self.categories.get(
                                confusion_idx[1], f"Class {confusion_idx[1]}"
                            )
                            classwise_dict[confusion_idx] = (
                                f"gt={labels[0]} → pred={labels[1]}",
                                dict(),
                            )
                        if cluster not in classwise_dict[confusion_idx][1]:
                            classwise_dict[confusion_idx][1][cluster] = [[]]
                        classwise_dict[confusion_idx][1][cluster][0].append(error_fig)

        classwise_dict: Dict[int, Tuple[str, Dict[int, List[List[dict]]]]] = {}
        regroup_dict(classwise_dict, cls_classwise_dict)
        regroup_dict(classwise_dict, cll_classwise_dict)

        return Section(
            id=section_id,
            title=title,
            description=description,
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
                    description=(
                        "The following is a montage of the classification errors."
                    ),
                    content=classwise_dict,
                ),
            ],
        )

    @logger()
    def duplicate_error(self, **kwargs: dict) -> Section:
        """Generate a section visualizing the duplicate errors in the dataset.

        Parameters
        ----------
        kwargs : dict
            Additional keyword arguments to pass to `error_classwise_dict`

        Returns
        -------
        Section
            The section containing the visualizations
        """

        section_id = "DuplicateError"
        title = "Duplicate Errors"
        description = f"""
            List of all the detections with confidence above the <span class="code">conf_threshold={self.overall_summary["conf_threshold"]}</span> but lower than the confidence of another true positive prediction.
            """
        if self.summaries[kwargs.get("evaluator_id", 0)]["DuplicateError"] == 0:
            return empty_section(
                section_id=section_id,
                title=title,
                description=f"""
                <p>{description}</p>
                <p>No {title.lower()} were found.</p>
                """,
            )

        get_crop_options(DuplicateError, kwargs)

        classwise_dict = self.error_classwise_dict(**kwargs)

        return Section(
            id=section_id,
            title=title,
            description=description,
            contents=[
                Content(
                    type=ContentType.IMAGES,
                    header="Visualizations",
                    description=(
                        "The following is a montage of the duplicate errors. White"
                        " boxes indicate ground truths, green boxes indicate best"
                        " predictions, and cyan boxes indicate duplicate predictions."
                    ),
                    content=classwise_dict,
                ),
            ],
        )

    @logger()
    def missed_error(self, **kwargs) -> Section:
        """Generate a section visualizing the missed errors in the dataset.

        Parameters
        ----------
        kwargs : dict
            Additional keyword arguments to pass to `error_classwise_dict`

        Returns
        -------
        Section
            The section containing the visualizations
        """

        section_id = "MissedError"
        title = "Missed Errors"
        description = """
            List of all the ground truths that have no corresponding prediction.
            """
        if self.summaries[kwargs.get("evaluator_id", 0)]["MissedError"] == 0:
            return empty_section(
                section_id=section_id,
                title=title,
                description=f"""
                <p>{description}</p>
                <p>No {title.lower()} were found.</p>
                """,
            )

        get_crop_options(MissedError, kwargs)

        classwise_dict = self.error_classwise_dict(**kwargs)
        box = self.boxplot(classwise_dict)

        sections = {
            "crowded": (
                "Crowded",
                "List of all the missed detections that overlap with other objects.",
            ),
            "low_feat": (
                "Not Enough Visual Features",
                "List of all the missed detections that are blurry or too small.",
            ),
            "occluded": (
                "Occluded",
                "List of all the missed detections that are occluded.",
            ),
            "truncated": (
                "Truncated",
                "List of all the missed detections that occur at the edge of the"
                " image.",
            ),
            "bad_label": (
                "Bad Labels",
                "List of all the missed detections that have bad labels.",
            ),
            "others": ("Others", "List of all other missed detections."),
        }

        groups: Dict[str, List[Error]] = self.missed_groups()[kwargs["evaluator_id"]]

        figs: Dict[str, Dict[int, Tuple[str, Dict[int, List[List[dict]]]]]] = {
            k: {} for k in groups
        }

        subclusters = set()

        for group, errors in groups.items():
            for error in errors:
                fig: dict = self.crops.get((kwargs["evaluator_id"], error))
                if fig is None:
                    logging.warn(f"Could not find crop for {error}")
                    continue
                fig = fig.copy()
                if error.gt_label not in figs[group]:
                    label_str = "Missed: {label}".format(
                        label=self.categories.get(
                            error.gt_label, f"Class {error.gt_label}"
                        )
                    )
                    figs[group][error.gt_label] = (label_str, {})

                cluster = fig.get("cluster")
                if -1 not in cluster and cluster in subclusters:
                    continue
                subclusters.add(cluster)

                cluster = cluster[0]

                fig["caption"] += f" | Sub {fig.get('cluster')[1]}"
                if cluster not in figs[group][error.gt_label][1]:
                    figs[group][error.gt_label][1][cluster] = [[]]

                figs[group][error.gt_label][1][cluster][0].append(fig)

        return Section(
            id=section_id,
            title=title,
            description=description,
            contents=[
                Content(
                    type=ContentType.PLOT,
                    header="Ranking",
                    description="""
                    The following plot shows the size (area) distribution of missed errors for each class. The presence of outliers in this plot indicates the misses may be caused by extreme object sizes.
                    """,
                    content=dict(plot=box),
                ),
                *[
                    Content(
                        type=ContentType.IMAGES,
                        header=sections[key][0],
                        description=sections[key][1],
                        content=fig,
                    )
                    for key, fig in figs.items()
                    if len(fig) > 0
                ],
            ],
        )

    @logger()
    def missed_groups(
        self,
        ids: List[int] = None,
        *,
        min_size: int = 32,
        var_threshold: float = 100,
    ) -> List[Dict[str, list]]:
        KERNEL = torch.tensor([[[0, 1, 0], [1, -4, 1], [0, 1, 0]]]).float().unsqueeze(0)

        errorlist_dicts = (
            self.errorlist_dicts
            if ids is None
            else [self.errorlist_dicts[i] for i in ids if i < len(self.errorlist_dicts)]
        )
        errorlist: List[Dict[str, List[MissedError]]] = [
            d["MissedError"] for d in errorlist_dicts
        ]
        groups = [
            {
                "crowded": [],
                "low_feat": [],
                "occluded": [],
                "truncated": [],
                "bad_label": [],  # subjective
                "others": [],
            }
            for _ in range(len(errorlist))
        ]
        for i, model_errors in enumerate(errorlist):
            for image_path, errors in model_errors.items():
                image_tensor = read_image(image_path)
                for error in errors:
                    count = 0
                    if error.crowd_ids().shape[0] > 0:
                        groups[i]["crowded"].append(error)
                        count += 1
                    bbox = error.gt_bbox.long()
                    image_crop: torch.Tensor = crop(
                        image_tensor, bbox[1], bbox[0], bbox[3], bbox[2]
                    )
                    image_crop = Grayscale()(image_crop).unsqueeze(0).float()
                    if (
                        torch.min(bbox[2:] - bbox[:2]) < min_size
                        or conv2d(image_crop, KERNEL).var() < var_threshold
                    ):
                        groups[i]["low_feat"].append(error)
                        count += 1

                    if (
                        bbox.min() <= min_size // 2
                        or bbox[2] >= image_tensor.shape[2] - min_size // 2
                        or bbox[3] >= image_tensor.shape[1] - min_size // 2
                    ):
                        groups[i]["truncated"].append(error)
                        count += 1

                    if count == 0:
                        groups[i]["others"].append(error)

        return groups

    @logger()
    def true_positives(self, **kwargs: dict) -> Section:
        """Generate a section visualizing the true positives in the dataset.

        Parameters
        ----------
        kwargs : dict
            Additional keyword arguments to pass to `error_classwise_dict`

        Returns
        -------
        Section
            The section containing the visualizations
        """

        section_id = "TruePositive"
        title = "True Positives"
        description = f"""
            List of all the true positive detections. No prediction was made below <span class="code">conf_threshold={ self.overall_summary["conf_threshold"] }</span>.
            """
        if self.summaries[kwargs.get("evaluator_id", 0)]["true_positives"] == 0:
            return empty_section(
                section_id=section_id,
                title=title,
                description=f"""
                <p>{description}</p>
                <p>No {title.lower()} were found.</p>
                """,
            )

        get_crop_options(NonError, kwargs)

        classwise_dict = self.error_classwise_dict(**kwargs)

        return Section(
            id=section_id,
            title=title,
            description=description,
            contents=[
                Content(
                    type=ContentType.IMAGES,
                    header="Visualizations",
                    content=classwise_dict,
                ),
            ],
        )

    # endregion

    @logger()
    def inspect(
        self,
        *,
        order: dict = None,
        weights: Union[dict, ErrorWeights] = ErrorWeights.F1,
        **kwargs: dict,
    ) -> Tuple[dict, dict]:
        """Generate figures and plots for the errors.

        Parameters
        ----------
        order : dict, default=`None`
            A dictionary containing the order in which the sections should be displayed. If not specified, error sections are sorted by decreasing weight
        weights : Union[dict, ErrorWeights], default=`ErrorWeights.PRECISION`
            A dictionary containing the weights for each error type, or an ErrorWeights enum

        Returns
        -------
        Tuple[dict, dict]
            A tuple of dictionaries containing the sections and the section names
        """
        if order is not None:
            pass
        elif isinstance(weights, ErrorWeights):
            order = weights.orders
        else:
            order = {}

        order["overview"] = order.get("overview", math.inf)
        order["TP"] = order.get("TP", -1)

        results = dict()

        func_mapping = {
            "BKG": self.background_error,
            "confusions": self.confusions,
            "LOC": self.localization_error,
            "DUP": self.duplicate_error,
            "MIS": self.missed_error,
            "TP": self.true_positives,
        }

        for code, func in func_mapping.items():
            results[code] = func(**kwargs)

        evaluator_id = kwargs.get("evaluator_id", 0)

        self.recalculate_summaries([evaluator_id], weights=weights)

        results["overview"] = self.summary([evaluator_id])

        sections = dict(
            sorted(
                results.items(),
                key=lambda x: order.get(x[0], -2),
                reverse=True,
            )
        )
        section_names = {
            "overview": ("Overview", "Overview"),
            "BKG": ("BackgroundError", "Background Errors"),
            "LOC": ("LocalizationError", "Localization Errors"),
            "confusions": ("Confusions", "Confusions"),
            "DUP": ("DuplicateError", "Duplicate Errors"),
            "MIS": ("MissedError", "Missed Errors"),
            "TP": ("TruePositive", "True Positives"),
        }

        self._generated_crops.add(evaluator_id)
        return sections, section_names

    @logger()
    def compare_background_errors(self) -> Section:
        """Generate a section comparing the background errors between models.

        Returns
        -------
        Section
            The section containing the visualizations
        """

        mask = [label[0] > -1 for label in self.projector.labels]

        figs = [
            self.error_classwise_dict(
                BackgroundError,
                color=ErrorColor.BKG,
                axis=1,
                evaluator_id=idx,
            )
            for idx in range(len(self.evaluators))
        ]

        combined_figs: Dict[int, Tuple[str, Dict[int, List[List[dict]]]]] = {}

        for idx, model_figs in enumerate(figs):
            for class_idx, (info, clusters) in model_figs.items():
                for cluster, images in clusters.items():
                    if class_idx not in combined_figs:
                        combined_figs[class_idx] = (info, {})
                    if cluster not in combined_figs[class_idx][1]:
                        combined_figs[class_idx][1][cluster] = [
                            [] for _ in range(len(self.evaluators))
                        ]
                    combined_figs[class_idx][1][cluster][idx] = images[0]

        return Section(
            id="BackgroundError",
            title="Background Errors",
            description=f"""
            List of all the false positive detections with confidence above the <span class="code">conf_threshold={ self.overall_summary["conf_threshold"] }</span> but do not pass the <span class="code">bg_iou_threshold={ self.overall_summary["bg_iou_threshold"] }</span>.
            """,
            contents=[
                Content(
                    type=ContentType.IMAGES,
                    header=f"Errors in models",
                    content=combined_figs,
                    data=dict(grouped=True),
                )
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
        fig = plotly_markup(FlowVisualizer(self.evaluators, self.image_dir).visualize())

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
    def compare_errors(
        self,
        error_type: Type[Error],
        ids: List[int] = (0, 1),
    ) -> List[Section]:
        assert (
            len(self.evaluators) > 1
        ), "Must have more than one evaluator to compare errors"
        assert len(ids) == 2, "Must have exactly two ids to compare errors"
        assert isinstance(ids, (list, tuple)), "ids must be a list or tuple"

        if error_type is BackgroundError:
            return self.compare_background_errors()

        mask = [label[0] == -1 for label in self.projector.labels]
        gt_clusters = self.clusters[mask]

        evaluators = [self.evaluators[idx] for idx in ids]

        # gt_errors keys: class_idx -> gt_id : (image_path, [errors for each model])
        gt_errors: Dict[int, Dict[int, Tuple[str, List[List[Error]]]]] = {}
        for i, evaluator in enumerate(evaluators):
            gt_data = evaluator.get_gt_data()
            for (gt_id, errors), gt_label, image_path in zip(
                gt_data.gt_errors.items(), gt_data.gt_labels, gt_data.images
            ):
                gt_label = gt_label.item()
                if gt_label not in gt_errors:
                    gt_errors[gt_label] = {}
                if gt_id not in gt_errors[gt_label]:
                    gt_errors[gt_label][gt_id] = (
                        image_path,
                        [[] for _ in range(len(ids))],
                    )
                gt_errors[gt_label][gt_id][1][i] = errors

        logging.info("Collated errors")

        ## generate crops
        # TODO: Slowest part of the pipeline. Can we parallelize this?
        # figs keys: gt_id : list of images for each model
        figs: Dict[int, List[list]] = {}
        for gt_label, errors in gt_errors.items():
            for gt_id, (image_path, error_list) in tqdm(
                errors.items(), desc=f"Generating crops (Class {gt_label})"
            ):
                if gt_id not in figs:
                    figs[gt_id] = [[] for _ in evaluators]
                image_tensor = read_image(image_path)
                image_name = image_path.split("/")[-1]
                cluster = tuple(gt_clusters[gt_id].tolist())
                for i, model_errors in zip(ids, error_list):
                    for error in model_errors:
                        crop_key = (i, error)
                        if crop_key in self.crops:
                            figs[gt_id][i].append(self.crops[crop_key])
                            continue

                        crop_options = get_crop_options(error.__class__)
                        fig = generate_fig(
                            image_tensor=image_tensor,
                            image_name=image_name,
                            error=error,
                            color=crop_options["color"],
                            bbox_attr=crop_options["bbox_attr"],
                            cluster=cluster,
                            get_bbox_func=crop_options["get_bbox_func"],
                            add_metadata_func=crop_options["add_metadata_func"],
                        )
                        self.crops[crop_key] = fig
                        figs[gt_id][i].append(fig)

                    # TODO: this sorting is repeated in Evaluation().get_pred_status(). Use that instead
                    figs[gt_id][i] = [
                        sorted(
                            figs[gt_id][i],
                            key=lambda x: (x["iou"] or 0, x["confidence"]),
                            reverse=True,
                        )[0]
                    ]

        logging.info("Generated crops")

        ## Prepare sections
        """
        Filter by flow (only show improvements/degradations)
        -> section, class_id, clusters, error_type
        """

        _, edges = FlowVisualizer(evaluators, self.image_dir).generate_graph()

        gt_ids_by_section: List[Set[int]] = [set(), set()]
        for _, (edge_gt_ids, score) in edges[["gt_ids", "score"]].iterrows():
            # skip transitions with no change
            if math.isclose(score, 0):
                continue
            section_id = int(score > 0)
            gt_ids_by_section[section_id].update(edge_gt_ids)

        section_contents = [
            dict(
                id="Degradations",
                title="Degradations",
                description=f"""
                Degradations from model {evaluators[0].name} to model {evaluators[1].name}.
                """,
                contents={},
            ),
            dict(
                id="Improvements",
                title="Improvements",
                description=f"""
                Improvements from model {evaluators[0].name} to model {evaluators[1].name}.
                """,
                contents={},
            ),
        ]
        for i, section_gt_ids in enumerate(gt_ids_by_section):
            for gt_id in section_gt_ids:
                class_idx = gt_data.gt_labels[gt_id].item()
                cluster = tuple(gt_clusters[gt_id].tolist())
                gt_figs = figs.get(gt_id)
                if gt_figs is None:
                    continue

                if class_idx not in section_contents[i]["contents"]:
                    section_contents[i]["contents"][class_idx] = {}

                contents: Dict[
                    int, Tuple[str, Dict[int, List[List[dict]]]]
                ] = section_contents[i]["contents"][class_idx]

                if cluster not in contents:
                    contents[cluster] = (f"Cluster {cluster[0]}", {})
                contents[cluster][1][gt_id] = gt_figs

        logging.info("Prepared sections")

        return [
            Section(
                id=section_content["id"],
                title=section_content["title"],
                description=section_content["description"],
                contents=[
                    Content(
                        type=ContentType.IMAGES,
                        header=f"Errors in Class {class_idx}",
                        content=section_content["contents"][class_idx],
                        data=dict(grouped=True, compact=True),
                    )
                    for class_idx in section_content["contents"]
                ],
            )
            for section_content in section_contents
        ]

    @logger()
    def compare(self) -> Dict[str, Section]:
        """Generate figures and plots for the errors.

        Returns
        -------
        Dict
            A dictionary containing the generated figures and plots.
        """

        results = dict()

        results["overview"] = self.summary()
        results["flow"] = self.flow()
        results["background_errors"] = self.compare_background_errors()
        results["degradations"], results["improvements"] = self.compare_errors(
            error_type=Error
        )

        section_names = {
            "overview": ("Overview", "Overview"),
            "flow": ("Flow", "Flow"),
            "background_errors": ("BackgroundError", "Background Errors"),
            "degradations": ("Degradations", "Degradations"),
            "improvements": ("Improvements", "Improvements"),
        }

        self._generated_crops.update({0, 1})

        return results, section_names

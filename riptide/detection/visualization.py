import logging
import math
from typing import Any, Callable, Dict, Iterable, List, Set, Tuple, Type, Union

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.nn.functional import conv2d
from torchvision.io import read_image
from torchvision.ops.boxes import box_iou
from torchvision.transforms import Grayscale
from torchvision.transforms.functional import crop

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
from riptide.utils.crops import (
    add_metadata,
    generate_fig,
    get_bbox_by_attr,
    get_both_bboxes,
    get_crop_options,
)
from riptide.utils.image import encode_base64
from riptide.utils.logging import logger
from riptide.utils.models import GTData
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
            ), "Models should be evaluated on the same dataset."
        else:
            evaluators = [evaluators]

        self.evaluators = evaluators
        evaluator = evaluators[0]
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

        self.projector = CropProjector(
            name=f"Crops",
            images=self.gt_data.crops + bkg_crops,
            encoder_mode="preconv",
            normalize_embeddings=True,
            labels=actual_labels + bkg_labels,
            device=torch.device("cpu"),
        )

        self.clusters = self.projector.cluster()

        self.overview()

        self.crops = {}

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
    def summary(self, ids: List[int] = None) -> Section:
        """Generate a summary of the evaluation results for each model."""
        evaluators = (
            self.evaluators
            if ids is None
            else [self.evaluators[min(i, len(self.evaluators) - 1)] for i in ids]
        )

        content = [
            {
                "No. of Images": self.num_images,
                "Conf. Threshold": self.conf_threshold,
                "IoU Threshold": f"{self.iou_threshold[0]} - {self.iou_threshold[1]}",
            },
            [None] * len(evaluators),
        ]
        for i, evaluator in enumerate(evaluators):
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
        self, classwise_dict: Dict[int, Tuple[str, Dict[int, List[List[dict]]]]]
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
        preview_size: int = 192,
        bbox_attr: str = None,
        label_attr: str = None,
        get_bbox_func: Callable[[Tuple[Error, str]], torch.Tensor] = None,
        get_label_func: Callable[[int], str] = None,
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

        errors = self.errorlist_dicts[evaluator_id].get(error_type.__name__, dict())

        if not isinstance(color, list):
            color = [color]

        color = [c.colorstr if isinstance(c, ErrorColor) else c for c in color]

        attr_prefix = "gt" if axis == 0 else "pred"
        bbox_attr = bbox_attr or f"{attr_prefix}_bbox"
        label_attr = label_attr or f"{attr_prefix}_label"

        get_bbox_func = get_bbox_func or get_bbox_by_attr
        add_metadata_func = add_metadata_func or add_metadata
        get_label_func = get_label_func or (lambda label: f"Predicted: Class {label}")

        if clusters is None:
            projector = self.projector

            # code = "TP" if error_type is NonError else error_type.code

            mask = (
                [label[0] == -1 for label in projector.labels]
                if error_type is not BackgroundError
                else [label[0] != -1 for label in projector.labels]
            )

            # clusters = self.clusters[mask]
            clusters = projector.cluster(mask=mask)

        else:
            assert (
                clusters.shape[0] == num_errors
            ), "Number of clusters does not match number of errors"

        # endregion

        if error_type is BackgroundError:

            def get_cluster(idx, bkg_idx):
                return clusters[bkg_idx].item()

        else:

            def get_cluster(idx, bkg_idx):
                return clusters[idx].item()

        classwise_dict: Dict[int, Tuple[str, Dict[int, List[List[dict]]]]] = {}
        label_set = set()
        bkg_idx = 0
        for image_path, image_errors in errors.items():
            image_tensor = read_image(image_path)
            image_name = image_path.split("/")[-1]
            for error in image_errors:
                label: int = getattr(error, label_attr)
                label_set.add(label)

                if label not in classwise_dict:
                    classwise_dict[label] = (
                        get_label_func(label),
                        dict(),
                    )

                cluster = get_cluster(error.idx, bkg_idx)
                if cluster not in classwise_dict[label][1]:
                    classwise_dict[label][1][cluster] = [[]]

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
                if crop_key in self.crops:
                    logging.warn(f"Duplicate crop: {crop_key}")

                self.crops[crop_key] = fig

                classwise_dict[label][1][cluster][0].append(fig)
                bkg_idx += 1

        for label in label_set:
            class_info, clusters_dict = classwise_dict[label]

            classwise_dict[label] = (
                class_info,
                dict(sorted(clusters_dict.items(), key=lambda x: x[0], reverse=True)),
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

    # region: Error Sections
    @logger()
    def background_error(
        self, data: dict = None, clusters: torch.Tensor = None, **kwargs
    ) -> Section:
        """Generate a section visualizing the background errors in the dataset.

        Parameters
        ----------
        data : dict, optional
            Metadata to attach to content, by default None
        clusters : torch.Tensor, optional
            The cluster assignments for each error, if None then clusters are computed by `error_classwise_dict`, by default None

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

        if self.overall_summary[section_id] == 0:
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

        kwargs.update(
            dict(
                error_type=BackgroundError,
                color=ErrorColor.BKG,
                axis=1,
                clusters=clusters,
            )
        )

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
    def classification_error(
        self, **kwargs: dict
    ) -> Tuple[Dict[int, List[Dict]], bytes]:
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

        if self.overall_summary[section_id] == 0:
            return empty_section(
                section_id=section_id,
                title=title,
                description=f"""
                <p>{description}</p>
                <p>No {title.lower()} were found.</p>
                """,
            )

        get_label_func = lambda x: f"Ground Truth: Class {x}"

        def add_metadata_func(x: dict, error: ClassificationError) -> dict:
            x.update(
                {
                    "caption": " | ".join(
                        [
                            x["image_name"],
                            f"Pred: Class {x['pred_class']}",
                            f"Conf { x['confidence'] }",
                            f"Cluster { x['cluster'] }",
                        ]
                    ),
                }
            )

            return x

        kwargs.update(
            dict(
                error_type=ClassificationError,
                color=ErrorColor.CLS,
                axis=1,
                label_attr="gt_label",
                get_label_func=get_label_func,
                add_metadata_func=add_metadata_func,
            )
        )

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
    def localization_error(self, **kwargs: dict) -> Tuple[Dict[int, Dict], bytes]:
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

        if self.overall_summary[section_id] == 0:
            return empty_section(
                section_id=section_id,
                title=title,
                description=f"""
                <p>{description}</p>
                <p>No {title.lower()} were found.</p>
                """,
            )

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
                            f"Cluster { x['cluster'] }",
                        ]
                    ),
                }
            )

            return x

        kwargs.update(
            dict(
                error_type=LocalizationError,
                color=[ErrorColor.WHITE, ErrorColor.LOC],
                axis=1,
                get_bbox_func=get_both_bboxes,
                get_label_func=get_label_func,
                add_metadata_func=add_metadata_func,
            )
        )

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
    def classification_and_localization_error(
        self, **kwargs: dict
    ) -> Tuple[Dict[int, Dict], bytes]:
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

        if self.overall_summary[section_id] == 0:
            return empty_section(
                section_id=section_id,
                title=title,
                description=f"""
                <p>{description}</p>
                <p>No {title.lower()} were found.</p>
                """,
            )

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
                            f"Cluster { x['cluster'] }",
                        ]
                    ),
                }
            )

            return x

        kwargs.update(
            dict(
                error_type=ClassificationAndLocalizationError,
                color=[ErrorColor.WHITE, ErrorColor.CLL],
                axis=1,
                get_bbox_func=get_both_bboxes,
                label_attr="gt_label",
                get_label_func=get_label_func,
                add_metadata_func=add_metadata_func,
            )
        )

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
    def duplicate_error(self, **kwargs: dict) -> Dict[int, Dict]:
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
        if self.overall_summary[section_id] == 0:
            return empty_section(
                section_id=section_id,
                title=title,
                description=f"""
                <p>{description}</p>
                <p>No {title.lower()} were found.</p>
                """,
            )

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
                            f"Cluster { x['cluster'] }",
                        ]
                    ),
                }
            )

            return x

        kwargs.update(
            dict(
                error_type=DuplicateError,
                color=[ErrorColor.WHITE, ErrorColor.TP, ErrorColor.DUP],
                axis=1,
                get_bbox_func=get_bbox_func,
                add_metadata_func=add_metadata_func,
                get_label_func=get_label_func,
            )
        )

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
                        " predictions, and red boxes indicate duplicate predictions."
                    ),
                    content=classwise_dict,
                ),
            ],
        )

    @logger()
    def missed_error(self, **kwargs: dict) -> Tuple[Dict[int, Dict], bytes]:
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
        if self.overall_summary[section_id] == 0:
            return empty_section(
                section_id=section_id,
                title=title,
                description=f"""
                <p>{description}</p>
                <p>No {title.lower()} were found.</p>
                """,
            )

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

        kwargs.update(
            dict(
                error_type=MissedError,
                color=ErrorColor.MIS,
                axis=0,
                get_label_func=get_label_func,
                add_metadata_func=add_metadata_func,
                evaluator_id=1,
            )
        )

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
            "unseen": ("Unseen", "List of all other missed detections."),
        }

        groups: Dict[str, List[Error]] = self.missed_groups()[kwargs["evaluator_id"]]

        figs: Dict[str, Dict[int, Tuple[str, Dict[int, List[List[dict]]]]]] = {
            k: {} for k in groups
        }

        for group, errors in groups.items():
            for error in errors:
                fig: dict = self.crops.get((kwargs["evaluator_id"], error))
                if fig is None:
                    logging.warn(f"Could not find crop for {error}")
                    continue
                if error.gt_label not in figs[group]:
                    figs[group][error.gt_label] = (
                        f"Missed: Class {error.gt_label}",
                        {},
                    )

                cluster = fig.get("cluster")
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
    ):
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
                "unseen": [],
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
                        # TODO: plot crowd
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
                        groups[i]["unseen"].append(error)

        return groups

    @logger()
    def true_positives(self, **kwargs: dict) -> Tuple[Dict[int, Dict], bytes]:
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
        if self.overall_summary["true_positives"] == 0:
            return empty_section(
                section_id=section_id,
                title=title,
                description=f"""
                <p>{description}</p>
                <p>No {title.lower()} were found.</p>
                """,
            )

        get_label_func = lambda x: f"Ground Truth: Class {x}"

        def add_metadata_func(x: dict, error: NonError) -> dict:
            x.update(
                {
                    "caption": " | ".join(
                        [
                            x["image_name"],
                            f"Conf {x['confidence']}",
                            f"IoU {x['iou']}",
                            f"Cluster { x['cluster'] }",
                        ]
                    ),
                }
            )

            return x

        kwargs.update(
            dict(
                error_type=NonError,
                color=[ErrorColor.WHITE, ErrorColor.TP],
                axis=1,
                get_bbox_func=get_both_bboxes,
                get_label_func=get_label_func,
                add_metadata_func=add_metadata_func,
            )
        )

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
    def confusions(self, **kwargs: dict) -> Tuple[Dict[int, List[Dict]], bytes]:
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

        section_id = "Confusions"
        title = "Confusions"
        description = """
        Confusions are the errors where the predicted class is different from the ground truth class. The following plot shows the number of confusions per class.
        """

        if self.overall_summary.get(section_id, -1) == 0:
            return empty_section(
                section_id=section_id,
                title=title,
                description=f"""
                <p>{description}</p>
                <p>No {title.lower()} were found.</p>
                """,
            )

        get_label_func = lambda x: f"Ground Truth: Class {x}"

        def add_metadata_func(x: dict, error: ClassificationError) -> dict:
            x.update(
                {
                    "caption": " | ".join(
                        [
                            x["image_name"],
                            f"Pred: Class {x['pred_class']}",
                            f"Conf { x['confidence'] }",
                            f"Cluster { x['cluster'] }",
                        ]
                    ),
                }
            )

            return x

        kwargs.update(
            dict(
                error_type=ClassificationError,
                color=ErrorColor.CLS,
                axis=1,
                label_attr="gt_label",
                get_label_func=get_label_func,
                add_metadata_func=add_metadata_func,
            )
        )

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
    def inspect(self) -> Tuple[Dict[str, Any], Dict[str, str]]:
        """Generate figures and plots for the errors.

        Returns
        -------
        Dict
            A dictionary containing the generated figures and plots.
        """

        results = {
            "overview": self.summary([0]),
            "distractions": self.background_error(),  # TODO: order by cluster size
            "confusions": self.classification_error(),
            "misses": self.missed_error(),
            "duplicates": self.duplicate_error(),
            "true_positives": self.true_positives(),
        }

        section_names = {
            "Overview": "Overview",
            "Distractions": "Distractions",
            "Confusions": "Confusions",
            "Misses": "Misses",
            "Duplicates": "Duplicates",
            "TruePositive": "True Positives",
        }

        return results, section_names

    @logger()
    def inspect(self) -> Dict[str, Any]:
        """Generate figures and plots for the errors.

        Returns
        -------
        Dict
            A dictionary containing the generated figures and plots.
        """

        results = dict()

        results["overview"] = self.summary([0])
        results["BKG"] = self.background_error()
        results["CLS"] = self.classification_error()
        results["LOC"] = self.localization_error()
        results["CLL"] = self.classification_and_localization_error()
        results["DUP"] = self.duplicate_error()
        results["MIS"] = self.missed_error()
        results["TP"] = self.true_positives()

        return results

    @logger()
    def compare_background_errors(self) -> Section:
        """Generate a section comparing the background errors between models.

        Returns
        -------
        Section
            The section containing the visualizations
        """

        mask = [label[0] != -1 for label in self.projector.labels]

        clusters = self.projector.cluster(mask=mask)
        submasks = [
            [label[0] == idx for label in self.projector.labels if label[0] != -1]
            for idx in range(len(self.evaluators))
        ]

        figs = [
            self.error_classwise_dict(
                BackgroundError,
                color=ErrorColor.BKG,
                axis=1,
                evaluator_id=idx,
                clusters=clusters[submasks[idx]],
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
        gt_clusters = self.projector.cluster(mask=mask)

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

        gt_ids = gt_data.gt_ids.tolist()

        logging.info("Collated errors")

        ## generate crops
        # TODO: Slowest part of the pipeline. Can we parallelize this?
        # figs keys: gt_id : list of images for each model
        figs: Dict[int, List[list]] = {}
        for errors in gt_errors.values():
            for gt_id, (image_path, error_list) in errors.items():
                if gt_id not in figs:
                    figs[gt_id] = [[] for _ in evaluators]
                image_tensor = read_image(image_path)
                image_name = image_path.split("/")[-1]
                cluster = gt_clusters[gt_ids.index(gt_id)].item()
                for i, model_errors in enumerate(error_list):
                    for error in model_errors:
                        crop_options = get_crop_options(error.__class__)

                        figs[gt_id][i].append(
                            generate_fig(
                                image_tensor=image_tensor,
                                image_name=image_name,
                                error=error,
                                color=crop_options["color"],
                                bbox_attr=crop_options["bbox_attr"],
                                cluster=cluster,
                                get_bbox_func=crop_options["get_bbox_func"],
                                add_metadata_func=crop_options["add_metadata_func"],
                            )
                        )

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
                gt_ref = gt_ids.index(gt_id)
                class_idx = gt_data.gt_labels[gt_ref].item()
                cluster = gt_clusters[gt_ref].item()
                gt_figs = figs.get(gt_id)
                if gt_figs is None:
                    continue

                if class_idx not in section_contents[i]["contents"]:
                    section_contents[i]["contents"][class_idx] = {}

                contents: Dict[
                    int, Tuple[str, Dict[int, List[List[dict]]]]
                ] = section_contents[i]["contents"][class_idx]

                if cluster not in contents:
                    contents[cluster] = (f"Cluster {cluster}", {})
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

        return results

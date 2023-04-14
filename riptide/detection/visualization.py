from typing import Any, Callable, Dict, Iterable, List, Tuple, Type, Union

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from torchvision.io import read_image
from torchvision.ops.boxes import box_iou
from torchvision.transforms.functional import to_pil_image

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
from riptide.utils.image import (
    convex_hull,
    crop_preview,
    encode_base64,
    get_bbox_stats,
    get_padded_bbox_crop,
)
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

        self.summaries = {
            evaluator.name: {
                "conf_threshold": evaluator.conf_threshold,
                "iou_thresholds": evaluator.iou_thresholds,
                **{k: round(v, 3) for k, v in evaluator.summarize().items()},
            }
            for evaluator in self.evaluators
        }

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

    def boxplot(self, classwise_dict: Dict[int, Tuple[str, List[Dict]]]) -> bytes:
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
        for class_idx, (_, info_dict) in classwise_dict.items():
            areas = [m["bbox_area"] for m in info_dict]
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
        evaluator_ids: List[int] = 0,
        preview_size=128,
        bbox_attr: str = None,
        label_attr: str = None,
        projector_attr: str = None,
        get_bbox_func: Callable[[Tuple[Error, str]], torch.Tensor] = None,
        get_label_func: Callable[[int], str] = None,
        add_metadata_func: Callable[[dict, Error], dict] = None,
    ) -> Dict[int, Tuple[str, List[Dict]]]:
        """Computes a dictionary of plots for the crops of the errors, classwise.

        Arguments
        ---------
        error_types : Union[Error, Iterable[Error]]
            Error types to plot confidence for

        color : Union[str, ErrorColor, List[Union[str, ErrorColor]]]
            Color(s) for the bounding box(es). Can be a single color or a list of colors

        axis : int, default=0
            Axis to crop image on. 0 for ground truth, 1 for predictions

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
        Dict[str, Dict[str, bytes]]
            Dictionary of plots for each error type, classwise
        """
        if not isinstance(evaluator_ids, (list, tuple)):
            evaluator_ids = [evaluator_ids]

        summaries = list(self.summaries.values())
        num_errors = sum(
            [summaries[idx].get(error_type.__name__, 0) for idx in evaluator_ids]
        )

        if num_errors == 0:
            return dict()

        errors_list = [
            self.errorlist_dicts[idx].get(error_type.__name__, dict())
            for idx in evaluator_ids
        ]
        errors: Dict[str, List[List[Error]]] = {}
        for i, error_dict in enumerate(errors_list):
            for key, error_list in error_dict.items():
                if key not in errors:
                    errors[key] = [[] for _ in range(len(evaluator_ids))]
                errors[key][i].extend(error_list)

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
        cluster_labels = {(i, code) for i in evaluator_ids}

        if len(cluster_labels) == 1:

            def cluster_filter(labels: List[Tuple[int, str]]) -> List[bool]:
                return [label in cluster_labels for label in labels]

            clusters = [projector.cluster(label_mask_func=cluster_filter)]
        else:
            clusters = projector.match_clusters(cluster_labels)

        classwise_dict: Dict[int, Tuple[str, Tuple[List[Dict], ...]]] = {}
        label_set = set()
        idx = 0
        for image_path, image_errors in errors.items():
            image_tensor = read_image(image_path)
            image_name = image_path.split("/")[-1]
            for model_id, model_errors in enumerate(image_errors):
                for error in model_errors:
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
                            tuple([] for _ in evaluator_ids),
                        )

                    confidence = (
                        round(error.confidence, 2)
                        if error.confidence is not None
                        else None
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

                    classwise_dict[label][1][model_id].append(
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
                                "cluster": clusters[model_id][idx],
                            },
                            error,
                        )
                    )
                    idx += 1

        for label in label_set:
            for model_id in range(len(evaluator_ids)):
                classwise_dict[label][1][model_id].sort(
                    key=lambda x: x["cluster"], reverse=True
                )

        classwise_dict = dict(
            sorted(
                classwise_dict.items(),
                key=lambda x: sum([len(y) for y in x[1][1]]),
                reverse=True,
            )
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

        return Section(
            id="BackgroundError",
            title="Background Errors",
            description=f"""
            List of all the false positive detections with confidence above the <span class="code">conf_threshold={ self.overall_summary["conf_threshold"] }</span> but do not pass the <span class="code">bg_iou_threshold={ self.overall_summary["bg_iou_threshold"] }</span>.
            """,
            contents=[
                Content(type=ContentType.IMAGES, header="Visualizations", content=figs)
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
                    The following plot shows the distribution of classification errors.
                    """,
            contents=[
                Content(
                    type=ContentType.PLOT,
                    header="Ranking",
                    content=dict(plot=fig),
                ),
                Content(
                    type=ContentType.IMAGES,
                    header="Visualizations",
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
        fig = self.error_classwise_ranking(LocalizationError)

        # return classwise_dict, fig

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

    @logger()
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
        )
        classwise_dict = dict(
            sorted(
                classwise_dict.items(),
                key=lambda x: len(x[1][1]),
                reverse=True,
            )
        )

        fig = self.boxplot(classwise_dict)

        return classwise_dict, fig

    @logger()
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
                key=lambda x: len(x[1][1]),
                reverse=True,
            )
        )

        fig = self.boxplot(classwise_dict)

        return classwise_dict, fig

    @logger()
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
                key=lambda x: len(x[1][1]),
                reverse=True,
            )
        )

        fig = self.boxplot(classwise_dict)
        # fig = encode_base64(plt.Figure())

        return classwise_dict, fig

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
        (
            results["classification_and_localization_error_figs"],
            results["classification_and_localization_error_plot"],
        ) = self.classification_and_localization_error()
        results["duplicate_error_figs"], _ = self.duplicate_error()
        results["missed_error_figs"], results["missed_error_plot"] = self.missed_error()
        (
            results["true_positive_figs"],
            results["true_positive_plot"],
        ) = self.true_positives()

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
                BackgroundError, color=ErrorColor.BKG, axis=1, evaluator_ids=idx
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

        results["overview"] = self.summary()
        results["background_errors"] = self.compare_background_errors()
        results["flow"] = self.flow()
        return results

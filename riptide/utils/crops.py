from typing import Callable, Type

import torch
from torchvision.ops.boxes import box_iou

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
from riptide.utils.colors import ErrorColor


def get_bbox_by_attr(error: Error, bbox_attr: str) -> torch.Tensor:
    t = getattr(error, bbox_attr)
    assert isinstance(t, torch.Tensor), f"{bbox_attr} is not a tensor"
    return t.unsqueeze(0)


def get_both_bboxes(error: Error, bbox_attr: str):
    return torch.stack([error.gt_bbox, error.pred_bbox])


def add_metadata(metadata: dict, error: Error) -> dict:
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


def label_func_generator(pre: str = "Class ", post: str = "") -> Callable[[int], str]:
    def func(label: int):
        return f"{pre}{label}{post}"

    return func


def get_crop_options(error_type: Type[Error], **kwargs) -> dict:
    """Returns the base crop options for the given error type

    Parameters
    ----------
    error_type : Type[Error]
        The error type to get the crop options for

    Returns
    -------
    dict
        The crop options for the given error type
    """

    if error_type is BackgroundError:
        kwargs.update(
            dict(
                error_type=BackgroundError,
                color=ErrorColor.BKG.hex,
                axis=1,
                bbox_attr="pred_bbox",
                label_attr="pred_label",
                projector_attr="pred_projector",
                get_bbox_func=get_bbox_by_attr,
                add_metadata_func=add_metadata,
                get_label_func=label_func_generator("Predicted: Class "),
            )
        )
    elif error_type is ClassificationError:

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

        kwargs.update(
            dict(
                error_type=ClassificationError,
                color=ErrorColor.CLS.hex,
                axis=1,
                bbox_attr="pred_bbox",
                label_attr="gt_label",
                projector_attr="pred_projector",
                get_bbox_func=get_bbox_by_attr,
                add_metadata_func=add_metadata_func,
                get_label_func=label_func_generator("Ground Truth: Class "),
            )
        )
    elif error_type is LocalizationError:

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
                color=[ErrorColor.WHITE.hex, ErrorColor.LOC.hex],
                axis=1,
                bbox_attr="pred_bbox",
                label_attr="pred_label",
                projector_attr="pred_projector",
                get_bbox_func=get_both_bboxes,
                add_metadata_func=add_metadata_func,
                get_label_func=label_func_generator("Ground Truth: Class "),
            )
        )
    elif error_type is ClassificationAndLocalizationError:

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
                color=[ErrorColor.WHITE.hex, ErrorColor.CLL.hex],
                axis=1,
                bbox_attr="pred_bbox",
                label_attr="gt_label",
                projector_attr="pred_projector",
                get_bbox_func=get_both_bboxes,
                add_metadata_func=add_metadata_func,
                get_label_func=label_func_generator("Ground Truth: Class "),
            )
        )
    elif error_type is DuplicateError:

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
                color=[ErrorColor.WHITE.hex, ErrorColor.TP.hex, ErrorColor.DUP.hex],
                axis=1,
                bbox_attr="pred_bbox",
                label_attr="pred_label",
                projector_attr="pred_projector",
                get_bbox_func=get_bbox_func,
                add_metadata_func=add_metadata_func,
                get_label_func=label_func_generator("Ground Truth: Class "),
            )
        )
    elif error_type is MissedError:

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
                color=ErrorColor.MIS.hex,
                axis=0,
                bbox_attr="gt_bbox",
                label_attr="gt_label",
                projector_attr="gt_projector",
                get_bbox_func=get_bbox_by_attr,
                add_metadata_func=add_metadata_func,
                get_label_func=label_func_generator("Missed: Class "),
            )
        )

    else:

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
                color=[ErrorColor.WHITE.hex, ErrorColor.TP.hex],
                axis=1,
                bbox_attr="pred_bbox",
                label_attr="pred_label",
                projector_attr="pred_projector",
                get_bbox_func=get_both_bboxes,
                add_metadata_func=add_metadata_func,
                get_label_func=label_func_generator("Ground Truth: Class "),
            )
        )

    return kwargs

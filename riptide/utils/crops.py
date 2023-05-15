from typing import Any, Callable, List, Type, Union

import torch
from PIL import Image
from torchvision.ops.boxes import box_iou
from torchvision.transforms.functional import to_pil_image

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
from riptide.utils.image import crop_preview, encode_base64, get_bbox_stats


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
                    f"Cluster { metadata['cluster'] }",
                ]
            ),
        }
    )

    return metadata


def label_func_generator(pre: str = "Class ", post: str = "") -> Callable[[int], str]:
    def func(label: int):
        return f"{pre}{label}{post}"

    return func


def get_crop_options(error_type: Type[Error], kwargs: dict = None) -> dict:
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

    if kwargs is None:
        kwargs = dict()

    if error_type is BackgroundError:
        kwargs.update(
            dict(
                error_type=BackgroundError,
                color=ErrorColor.BKG.hex,
                axis=1,
                bbox_attr="pred_bbox",
                label_attr="pred_label",
                get_bbox_func=get_bbox_by_attr,
                add_metadata_func=add_metadata,
                get_label_func=label_func_generator("Predicted: "),
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
                color=[ErrorColor.WHITE.hex, ErrorColor.CLS.hex],
                axis=1,
                bbox_attr="pred_bbox",
                label_attr="gt_label",
                get_bbox_func=get_both_bboxes,
                add_metadata_func=add_metadata_func,
                get_label_func=label_func_generator("Actual: "),
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
                get_bbox_func=get_both_bboxes,
                add_metadata_func=add_metadata_func,
                get_label_func=label_func_generator("Actual: "),
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
                get_bbox_func=get_both_bboxes,
                add_metadata_func=add_metadata_func,
                get_label_func=label_func_generator("Actual: "),
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
                get_bbox_func=get_bbox_func,
                add_metadata_func=add_metadata_func,
                get_label_func=label_func_generator("Actual: "),
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
                get_bbox_func=get_bbox_by_attr,
                add_metadata_func=add_metadata_func,
                get_label_func=label_func_generator("Missed: "),
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
                get_bbox_func=get_both_bboxes,
                add_metadata_func=add_metadata_func,
                get_label_func=label_func_generator("Actual: "),
            )
        )

    return kwargs


def generate_fig(
    image_tensor: torch.Tensor,
    image_name: str,
    error: Error,
    color: Union[str, ErrorColor, List[Union[str, ErrorColor]]],
    bbox_attr: str,
    get_bbox_func: Callable[[Error, str], Any],
    add_metadata_func: Callable[[dict, Error], dict],
    *,
    cluster: Any = 0,
    preview_size: int = 192,
) -> dict:
    bbox: torch.Tensor = getattr(error, bbox_attr)

    if bbox is not None:
        width, height, area = get_bbox_stats(bbox)
        bboxes = get_bbox_func(error, bbox_attr)

        crop_tensor = (
            crop_preview(image_tensor, bboxes, color)
            if isinstance(bboxes, torch.Tensor)
            else image_tensor
        )
        crop: Image.Image = to_pil_image(crop_tensor)
        encoded_crop = encode_base64(crop.resize((preview_size, preview_size)))
    else:
        width, height, area = (None, None, None)
        encoded_crop = None

    confidence = round(error.confidence, 2) if error.confidence is not None else None
    iou = (
        round(
            box_iou(error.pred_bbox.unsqueeze(0), error.gt_bbox.unsqueeze(0)).item(),
            3,
        )
        if error.pred_bbox is not None and error.gt_bbox is not None
        else None
    )

    data = {
        "type": error.code,
        "image_name": image_name,
        "image_base64": encoded_crop,
        "pred_class": error.pred_label,
        "gt_class": error.gt_label,
        "confidence": confidence,
        "bbox_width": width,
        "bbox_height": height,
        "bbox_area": area,
        "iou": iou,
        "cluster": cluster,
        "similar": [],
    }

    return add_metadata_func(data, error)

import os
from typing import Dict, Tuple, Type

import pandas as pd
import torch
import ujson as json


class DictLoader:
    def __init__(self, dict_file: str, image_dir: str):
        self.dict_file = dict_file
        self.image_dir = image_dir
        self.targets_dict_file = None
        self.predictions_dict_file = None

    @classmethod
    def from_dict_files(
        cls,
        targets_dict_file: str,
        predictions_dict_file: str,
        image_dir: str,
    ) -> "DictLoader":
        loader = cls(targets_dict_file, image_dir)
        loader.targets_dict_file = targets_dict_file
        loader.predictions_dict_file = predictions_dict_file
        return loader

    def process_boxes(self, gt_dict: dict, pred_dict: dict) -> Tuple[torch.Tensor, ...]:
        pred_bboxes = (
            pred_dict["boxes"] if len(pred_dict["boxes"]) > 0 else torch.empty((0, 4))
        )
        pred_scores = (
            pred_dict["scores"] if len(pred_dict["scores"]) > 0 else torch.Tensor([])
        )
        pred_labels = (
            pred_dict["labels"] if len(pred_dict["labels"]) > 0 else torch.Tensor([])
        )
        gt_bboxes = gt_dict["boxes"]
        gt_labels = gt_dict["labels"]
        return (
            pred_bboxes,
            pred_scores,
            pred_labels,
            gt_bboxes,
            gt_labels,
        )

    def load(
        self,
        evaluation_cls: Type,
        evaluator_cls: Type,
        conf_threshold: float = 0.5,
        **kwargs
    ):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if self.targets_dict_file is None and self.predictions_dict_file is None:
            results_dict = torch.load(self.dict_file, map_location=device)
            targets_dict = results_dict["targets"]
            predictions_dict = results_dict["predictions"]
        else:
            targets_dict = torch.load(self.targets_dict_file, map_location=device)
            predictions_dict = torch.load(
                self.predictions_dict_file, map_location=device
            )
        evaluations = []
        start = 0
        for file_name, gt in targets_dict.items():
            pred = predictions_dict[file_name]
            (
                pred_bboxes,
                pred_scores,
                pred_labels,
                gt_bboxes,
                gt_labels,
            ) = self.process_boxes(gt, pred)
            evaluation = evaluation_cls(
                image_path=os.path.join(self.image_dir, file_name),
                pred_bboxes=pred_bboxes.cpu(),
                pred_scores=pred_scores.cpu(),
                pred_labels=pred_labels.cpu(),
                gt_bboxes=gt_bboxes.cpu(),
                gt_labels=gt_labels.cpu(),
                gt_offset=start,
                conf_threshold=conf_threshold,
            )
            evaluations.append(evaluation)
            start += len(gt_labels)

        return evaluator_cls(evaluations, image_dir=self.image_dir, **kwargs)


class COCOLoader:
    def __init__(
        self,
        annotations_file: str,
        predictions_file: str,
        image_dir: str,
    ):
        self.annotations_file = annotations_file
        self.predictions_file = predictions_file
        self.image_dir = image_dir

    def load(
        self,
        evaluation_cls: Type,
        evaluator_cls: Type,
        conf_threshold: float = 0.5,
        obb: bool = False,
        **kwargs
    ):
        with open(self.annotations_file, "r") as f:
            j = json.load(f)
            target_annotations = pd.DataFrame(j["annotations"])
            target_images = pd.DataFrame(j["images"])
            categories = pd.DataFrame(j["categories"])
            categories.index = categories["id"]
            categories = categories.drop(columns=["id"]).to_dict(orient="index")
            categories = {k: v["name"] for k, v in categories.items()}
        with open(self.predictions_file, "r") as f:
            j = json.load(f)
            prediction_annotations = pd.DataFrame(j["annotations"])
            prediction_images = pd.DataFrame(j["images"])

        target_annotations["file_name"] = target_annotations["image_id"].apply(
            lambda x: target_images[target_images["id"] == x]["file_name"].values[0]
        )

        prediction_annotations["file_name"] = prediction_annotations["image_id"].apply(
            lambda x: prediction_images[prediction_images["id"] == x][
                "file_name"
            ].values[0]
        )

        results_dict = {}
        for id, file_name, bbox, category_id in target_annotations[
            ["id", "file_name", "bbox", "category_id"]
        ].values:
            if file_name not in results_dict:
                results_dict[file_name] = {
                    "targets": {"ids": [], "boxes": [], "labels": []},
                    "predictions": {"boxes": [], "scores": [], "labels": []},
                }
            results_dict[file_name]["targets"]["ids"].append(id)
            results_dict[file_name]["targets"]["boxes"].append(bbox)
            results_dict[file_name]["targets"]["labels"].append(category_id)

        for file_name, bbox, score, category_id in prediction_annotations[
            ["file_name", "bbox", "score", "category_id"]
        ].values:
            if file_name not in results_dict:
                results_dict[file_name] = {
                    "targets": {"ids": [], "boxes": [], "labels": []},
                    "predictions": {"boxes": [], "scores": [], "labels": []},
                }
            results_dict[file_name]["predictions"]["boxes"].append(bbox)
            results_dict[file_name]["predictions"]["scores"].append(score)
            results_dict[file_name]["predictions"]["labels"].append(category_id)

        evaluations = []
        start = 0
        gt_ids_map = []
        for file_name, pred_gt_dict in results_dict.items():
            coco_ids = torch.tensor(pred_gt_dict["targets"]["ids"])
            gt_ids_map.append(coco_ids)
            gt_bboxes = torch.tensor(pred_gt_dict["targets"]["boxes"])
            if not obb:
                if len(gt_bboxes) == 0:
                    gt_bboxes = torch.empty((0, 4))
                gt_bboxes[:, 2:] = gt_bboxes[:, :2] + gt_bboxes[:, 2:]
            else:
                if len(gt_bboxes) == 0:
                    gt_bboxes = torch.empty((0, 8))

            gt_labels = torch.tensor(pred_gt_dict["targets"]["labels"])

            pred_bboxes = torch.tensor(pred_gt_dict["predictions"]["boxes"])
            if not obb:
                if len(pred_bboxes) == 0:
                    pred_bboxes = torch.empty((0, 4))
                pred_bboxes[:, 2:] = pred_bboxes[:, :2] + pred_bboxes[:, 2:]
            else:
                if len(pred_bboxes) == 0:
                    pred_bboxes = torch.empty((0, 8))

            pred_scores = torch.tensor(pred_gt_dict["predictions"]["scores"])
            pred_labels = torch.tensor(pred_gt_dict["predictions"]["labels"])
            evaluation = evaluation_cls(
                image_path=os.path.join(self.image_dir, file_name),
                pred_bboxes=pred_bboxes,
                pred_scores=pred_scores,
                pred_labels=pred_labels,
                gt_bboxes=gt_bboxes,
                gt_labels=gt_labels,
                gt_offset=start,
                conf_threshold=conf_threshold,
                obb=obb,
            )
            evaluations.append(evaluation)
            start += len(gt_labels)

        return evaluator_cls(
            evaluations,
            image_dir=self.image_dir,
            gt_ids_map=torch.cat(gt_ids_map),
            categories=categories,
            **kwargs
        )

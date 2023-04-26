from __future__ import annotations

from typing import Dict, List

import torch
from pydantic import BaseModel

from riptide.detection.errors import Error


class GTData(BaseModel):
    """Data class for GT data"""

    crops: List[torch.Tensor]
    gt_ids: torch.Tensor
    gt_labels: torch.Tensor
    gt_errors: Dict[int, List[Error]]
    images: List[str]

    class Config:
        arbitrary_types_allowed = True

    @classmethod
    def combine(cls, *gt_data: GTData) -> GTData:
        """Combine multiple GTData objects into one"""
        try:
            gt_ids = set(gt_data[0].gt_ids.tolist())
            for gt in gt_data[1:]:
                curr_gt_ids = set(gt.gt_ids.tolist())
                assert gt_ids == curr_gt_ids
        except AssertionError:
            raise ValueError("GTData objects must have the same gt_ids")

        crops = []
        gt_ids = []
        gt_labels = []
        gt_errors = {}
        images = []
        for gt in gt_data:
            crops.extend(gt.crops)
            gt_ids.extend(gt.gt_ids)
            gt_labels.extend(gt.gt_labels)
            images.extend(gt.images)
            for gt_id, errors in gt.gt_errors.items():
                if gt_id not in gt_errors:
                    gt_errors[gt_id] = []
                gt_errors[gt_id].extend(errors)
        return cls(
            crops=crops,
            gt_ids=torch.stack(gt_ids),
            gt_labels=torch.stack(gt_labels),
            gt_errors=gt_errors,
            images=images,
        )

from __future__ import annotations

from typing import Dict, List

import torch
from pydantic import BaseModel

from riptide.detection.errors import Error


class GTData(BaseModel):
    """Data class for GT data"""

    crops: List[torch.Tensor]
    gt_labels: torch.Tensor
    gt_errors: Dict[int, List[Error]]
    images: List[str]

    class Config:
        arbitrary_types_allowed = True

    @classmethod
    def combine(cls, *gt_data: GTData) -> GTData:
        """Combine multiple GTData objects into one"""
        # assert that the length of all gt_data is the same
        assert (
            len(set([len(gt) for gt in gt_data])) == 1
        ), "Only GTData with the same number of ground truths can be combined"

        crops = []
        gt_labels = []
        gt_errors: Dict[int, list] = {}
        images = []
        for gt in gt_data:
            crops.extend(gt.crops)
            gt_labels.extend(gt.gt_labels)
            images.extend(gt.images)
            for gt_id, errors in gt.gt_errors.items():
                if gt_id not in gt_errors:
                    gt_errors[gt_id] = []
                gt_errors[gt_id].extend(errors)
        return cls(
            crops=crops,
            gt_labels=torch.stack(gt_labels),
            gt_errors=gt_errors,
            images=images,
        )

    def __len__(self) -> int:
        return len(self.crops)

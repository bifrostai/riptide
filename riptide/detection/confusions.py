from enum import Enum
from typing import List, Union


class Confusion(Enum):
    FALSE_NEGATIVE = 0
    TRUE_POSITIVE = 1
    FALSE_POSITIVE = 2
    UNUSED = 3

    @property
    def code(self):
        mapping = {
            Confusion.FALSE_NEGATIVE: "FN",
            Confusion.TRUE_POSITIVE: "TP",
            Confusion.FALSE_POSITIVE: "FP",
            Confusion.UNUSED: "UN",
        }
        return mapping[self]


class Confusions:
    """Confusions assigned to predictions or ground truths."""

    def __init__(
        self, evaluation: "ImageEvaluation", num_preds: int, num_gts: int
    ) -> None:
        self.evaluation = evaluation
        self.num_preds = num_preds
        self.num_gts = num_gts
        self.pred_confusions = [None] * num_preds
        self.gt_confusions = [None] * num_gts

    @property
    def is_complete(self) -> bool:
        return (
            None not in self.pred_confusions
            and None not in self.gt_confusions
            and all(isinstance(pred, Confusion) for pred in self.pred_confusions)
            and all(isinstance(gt, Confusion) for gt in self.gt_confusions)
        )

    def assign_prediction_confusion(self, pred_idx: int, confusion: Confusion) -> None:
        if confusion is Confusion.FALSE_NEGATIVE:
            raise Exception("A prediction confusion is either TP, FP, or UNUSED")
        if self.pred_confusions[pred_idx] is not None:
            raise Exception(
                f"Tried to assign {confusion} to an already assigned confusion "
                f"{self.pred_confusions[pred_idx]}"
            )
        self.pred_confusions[pred_idx] = confusion

    def assign_gt_confusion(self, gt_idx: int, confusion: Confusion) -> None:
        if confusion is Confusion.FALSE_POSITIVE or confusion is Confusion.UNUSED:
            raise Exception("A gt confusion is either TP or FN")
        if self.gt_confusions[gt_idx] is not None:
            raise Exception(
                f"Tried to assign {confusion} to an already assigned confusion "
                f"{self.gt_confusions[gt_idx]}"
            )
        self.gt_confusions[gt_idx] = confusion

    def check_prediction_confusion(self, pred_idx: int) -> Union[Confusion, None]:
        return self.pred_confusions[pred_idx]

    def check_gt_confusion(self, gt_idx: int) -> Union[Confusion, None]:
        return self.gt_confusions[gt_idx]

    def get_prediction_confusions(self) -> List[Confusion]:
        if None in self.pred_confusions:
            raise Exception(
                "The prediction confusions list has not been fully initialized: "
                f"{self.pred_confusions}"
            )
        return self.pred_confusions

    def get_gt_confusions(self) -> List[Confusion]:
        if None in self.gt_confusions:
            raise Exception(
                "The ground truth confusions list has not been fully initialized: "
                f"{self.gt_confusions}"
            )
        return self.gt_confusions

    def assert_valid_confusions(self) -> None:
        assert len(
            [c for c in self.gt_confusions if c is Confusion.TRUE_POSITIVE]
        ) == len([c for c in self.gt_confusions if c is Confusion.TRUE_POSITIVE]), (
            self.evaluation,
            self.pred_confusions,
            self.gt_confusions,
        )
        assert (
            len([c for c in self.pred_confusions if c is Confusion.TRUE_POSITIVE])
            + len([c for c in self.pred_confusions if c is Confusion.FALSE_POSITIVE])
            + len([c for c in self.pred_confusions if c is Confusion.UNUSED])
            == self.num_preds
        ), (
            self.evaluation,
            self.pred_confusions,
        )
        assert (
            len([c for c in self.gt_confusions if c is Confusion.TRUE_POSITIVE])
            + len([c for c in self.gt_confusions if c is Confusion.FALSE_NEGATIVE])
            == self.num_gts
        ), (
            self.evaluation,
            self.gt_confusions,
        )

    def get_true_positives(self) -> int:
        self.assert_valid_confusions()
        return len([c for c in self.pred_confusions if c is Confusion.TRUE_POSITIVE])

    def get_false_positives(self) -> int:
        self.assert_valid_confusions()
        return len([c for c in self.pred_confusions if c is Confusion.FALSE_POSITIVE])

    def get_false_negatives(self) -> int:
        self.assert_valid_confusions()
        return len([c for c in self.gt_confusions if c is Confusion.FALSE_NEGATIVE])

    def get_unused(self) -> int:
        self.assert_valid_confusions()
        return len([c for c in self.pred_confusions if c is Confusion.UNUSED])

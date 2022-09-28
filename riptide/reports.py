import os

from jinja2 import Environment, FileSystemLoader

from riptide.detection.characterization import (
    compute_aspect_variance,
    compute_size_variance,
)
from riptide.detection.errors import (
    BackgroundError,
    ClassificationAndLocalizationError,
    ClassificationError,
    DuplicateError,
    LocalizationError,
    MissedError,
)
from riptide.detection.visualization import (
    inspect_background_error,
    inspect_classification_error,
    inspect_error_confidence,
    inspect_missed_error,
)

ERROR_TYPES = [
    BackgroundError,
    ClassificationError,
    LocalizationError,
    ClassificationAndLocalizationError,
    DuplicateError,
    MissedError,
]


class HtmlReport:
    def __init__(self, evaluator: "Evaluator"):
        self.evaluator = evaluator
        self.env = Environment(loader=FileSystemLoader("static"), autoescape=True)
        self.template = self.env.get_template("template.html")

    def get_error_info(self) -> dict:
        error_info = {}
        for error_type in ERROR_TYPES:
            error_name = error_type.__name__
            error_info[error_name] = {}
            error_info[error_name]["confidence_hist"] = inspect_error_confidence(
                self.evaluator, error_type
            )
        return error_info

    def render(self):
        section_names = [
            "Overview",
            "BackgroundError",
            "ClassificationError",
            "LocalizationError",
            "ClassificationAndLocalizationError",
            "DuplicateError",
            "MissedError",
        ]

        # Summary data
        summary = {
            "num_images": self.evaluator.num_images,
            "conf_threshold": self.evaluator.evaluations[0].conf_threshold,
            "bg_iou_threshold": self.evaluator.evaluations[0].bg_iou_threshold,
            "fg_iou_threshold": self.evaluator.evaluations[0].fg_iou_threshold,
        }
        summary.update({k: round(v, 3) for k, v in self.evaluator.summarize().items()})

        classwise_summary = self.evaluator.classwise_summarize()
        for class_idx, individual_summary in classwise_summary.items():
            for metric, value in individual_summary.items():
                classwise_summary[class_idx][metric] = round(value, 3)

        # BackgroundError data - classwise false positives
        # TODO: Visual grouping using MeP
        background_error_figs = inspect_background_error(self.evaluator)

        background_error_figs = dict(
            sorted(
                background_error_figs.items(),
                key=lambda x: len(x[1]),
                reverse=True,
            )
        )

        # ClassificationError data - classwise confusion
        (
            classification_error_figs,
            classification_error_plot,
        ) = inspect_classification_error(self.evaluator)

        # MissedError data - classwise false negatives
        # TODO: Visual grouping using MeP
        missed_size_var = compute_size_variance(self.evaluator)
        missed_aspect_var = compute_aspect_variance(self.evaluator)

        missed_error_figs, missed_error_plot = inspect_missed_error(self.evaluator)

        missed_error_figs = dict(
            sorted(
                missed_error_figs.items(),
                key=lambda x: len(x[1]),
                reverse=True,
            )
        )

        output = self.template.render(
            title="Riptide",
            section_names=section_names,
            summary=summary,
            classwise_summary=classwise_summary,
            error_info=self.get_error_info(),
            background_error_figs=background_error_figs,
            classification_error_figs=classification_error_figs,
            classification_error_plot=classification_error_plot,
            missed_error_figs=missed_error_figs,
            missed_error_plot=missed_error_plot,
            missed_size_var=missed_size_var,
            missed_aspect_var=missed_aspect_var,
        )
        os.makedirs("output", exist_ok=True)
        with open("output/report.html", "w") as f:
            f.writelines(output)

import os
import shutil
from typing import List

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
    inspect_classification_and_localization_error,
    inspect_classification_error,
    inspect_duplicate_error,
    inspect_error_confidence,
    inspect_localization_error,
    inspect_missed_error,
    inspect_true_positives,
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

    def get_suggestions(
        self, overall_summary: dict, classwise_summary: dict, **kwargs
    ) -> List[dict]:
        suggestions = []
        overall_errors = {
            k: v for k, v in overall_summary.items() if k.endswith("Error")
        }
        worst_error, worst_error_value = max(overall_errors.items(), key=lambda x: x[1])
        if worst_error == "MissedError":
            worst_class_idx, classwise_errors = max(
                classwise_summary.items(), key=lambda x: x[1]["MissedError"]
            )
            suggestions.append(
                {
                    "title": f"Top Error: MissedError ({worst_error_value})",
                    "content": (
                        "You have a lot of missed detections in class"
                        f" {worst_class_idx} ({classwise_errors['MissedError']})."
                    ),
                }
            )
        return suggestions

    def get_error_info(self) -> dict:
        error_info = {}
        for error_type in ERROR_TYPES:
            error_name = error_type.__name__
            error_info[error_name] = {}
            error_info[error_name]["confidence_hist"] = inspect_error_confidence(
                self.evaluator, error_type
            )
        return error_info

    def render(self, output_dir: str):
        section_names = [
            "Overview",
            "BackgroundError",
            "ClassificationError",
            "LocalizationError",
            "ClassificationAndLocalizationError",
            "DuplicateError",
            "MissedError",
            "TruePositive",
        ]

        # Summary data
        print("Creating summaries...")
        evaluator_summary = self.evaluator.summarize()
        overall_summary = {
            "num_images": self.evaluator.num_images,
            "conf_threshold": self.evaluator.evaluations[0].conf_threshold,
            "bg_iou_threshold": self.evaluator.evaluations[0].bg_iou_threshold,
            "fg_iou_threshold": self.evaluator.evaluations[0].fg_iou_threshold,
        }
        overall_summary.update({k: round(v, 3) for k, v in evaluator_summary.items()})

        classwise_summary = self.evaluator.classwise_summarize()
        for class_idx, individual_summary in classwise_summary.items():
            for metric, value in individual_summary.items():
                classwise_summary[class_idx][metric] = round(value, 3)

        # BackgroundError data - classwise false positives
        # TODO: Visual grouping using MeP
        print("Visualizing BackgroundErrors...")
        background_error_figs = inspect_background_error(self.evaluator)

        background_error_figs = dict(
            sorted(
                background_error_figs.items(),
                key=lambda x: len(x[1]),
                reverse=True,
            )
        )

        # ClassificationError data - classwise confusion
        print("Visualizing ClassificationErrors...")
        (
            classification_error_figs,
            classification_error_plot,
        ) = inspect_classification_error(self.evaluator)

        # LocalizationError data - classwise confusion
        print("Visualizing LocalizationErrors...")
        (
            localization_error_figs,
            localization_error_plot,
        ) = inspect_localization_error(self.evaluator)

        # ClassificationAndLocalizationError data - classwise confusion
        print("Visualizing ClassificationAndLocalizationError...")
        (
            classification_and_localization_error_figs,
            classification_and_localization_error_plot,
        ) = inspect_classification_and_localization_error(self.evaluator)

        # DuplicateError data - classwise confusion
        print("Visualizing DuplicateErrors...")
        (
            duplicate_error_figs,
            duplicate_error_plot,
        ) = inspect_duplicate_error(self.evaluator)

        # MissedError data - classwise false negatives
        # TODO: Visual grouping using MeP
        print("Visualizing MissedErrors...")
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

        # True Positives data - to bring a balanced and unbiased view to the dataset
        print("Visualizing TruePositives...")
        (
            true_positive_figs,
            true_positive_plot,
        ) = inspect_true_positives(self.evaluator)

        # Infobox suggestions
        infoboxes = self.get_suggestions(
            overall_summary,
            classwise_summary,
            missed_size_var=missed_size_var,
            missed_aspect_var=missed_aspect_var,
        )

        print("Rendering output...")
        output = self.template.render(
            title="Riptide",
            section_names=section_names,
            summary=overall_summary,
            classwise_summary=classwise_summary,
            infoboxes=infoboxes,
            error_info=self.get_error_info(),
            background_error_figs=background_error_figs,
            classification_error_figs=classification_error_figs,
            classification_error_plot=classification_error_plot,
            localization_error_figs=localization_error_figs,
            localization_error_plot=localization_error_plot,
            classification_and_localization_error_figs=classification_and_localization_error_figs,
            classification_and_localization_error_plot=classification_and_localization_error_plot,
            duplicate_error_figs=duplicate_error_figs,
            # duplicate_error_plot=duplicate_error_plot,
            missed_error_figs=missed_error_figs,
            missed_error_plot=missed_error_plot,
            missed_size_var=missed_size_var,
            missed_aspect_var=missed_aspect_var,
            true_positive_figs=true_positive_figs,
            true_positive_plot=true_positive_plot,
        )
        os.makedirs(output_dir, exist_ok=True)
        with open(f"{output_dir}/report.html", "w") as f:
            f.writelines(output)

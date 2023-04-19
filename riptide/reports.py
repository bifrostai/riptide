import logging
import os
import shutil
from typing import Dict, List

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
from riptide.detection.visualization import Inspector

ERROR_TYPES = [
    BackgroundError,
    ClassificationError,
    LocalizationError,
    ClassificationAndLocalizationError,
    DuplicateError,
    MissedError,
]


class HtmlReport:
    def __init__(self, evaluators: "Evaluator"):
        if not isinstance(evaluators, list):
            evaluators = [evaluators]
        self.evaluators = evaluators
        self.evaluator = evaluators[0]
        self.env = Environment(loader=FileSystemLoader("static"), autoescape=True)
        self.template = self.env.get_template("template.html")
        self.inspector = Inspector(evaluators)

    def get_suggestions(
        self, overall_summary: Dict, classwise_summary: Dict, **kwargs
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

    def get_error_info(self) -> Dict:
        confidence_hists: Dict[str, bytes] = self.inspector.error_confidence(
            ERROR_TYPES
        )
        return {
            error_name: {"confidence_hist": confidence_hist}
            for error_name, confidence_hist in confidence_hists.items()
        }

    def render(
        self,
        output_dir: str,
        fname: str = "report.html",
        template: str = "template.html",
    ):
        inspector = self.inspector
        section_names = {
            "Overview": "Overview",
            "BackgroundError": "Background Errors",
            "ClassificationError": "Classification Errors",
            "LocalizationError": "Localization Errors",
            "ClassificationAndLocalizationError": (
                "Classification And Localization Errors"
            ),
            "DuplicateError": "Duplicate Errors",
            "MissedError": "Missed Errors",
            "TruePositive": "True Positives",
        }

        # Summary data
        overall_summary, classwise_summary, summary_section = inspector.overview()

        # Error data - figures and plots for each error type
        error_fig_plots = inspector.inspect()

        # MissedError data - classwise false negatives
        missed_size_var = compute_size_variance(self.evaluators[0])
        missed_aspect_var = compute_aspect_variance(self.evaluators[0])

        # Infobox suggestions
        infoboxes = self.get_suggestions(
            overall_summary,
            classwise_summary,
            missed_size_var=missed_size_var,
            missed_aspect_var=missed_aspect_var,
        )

        error_info = self.get_error_info()

        logging.info("Rendering output...")
        output = self.env.get_template(template).render(
            title="Riptide",
            section_names=section_names,
            summary=overall_summary,
            summary_section=summary_section,
            classwise_summary=classwise_summary,
            infoboxes=infoboxes,
            error_info=error_info,
            missed_size_var=missed_size_var,
            missed_aspect_var=missed_aspect_var,
            **error_fig_plots,
        )
        os.makedirs(output_dir, exist_ok=True)
        fout = os.path.join(output_dir, fname)
        with open(fout, "w") as f:
            f.writelines(output)
        logging.info(f"Rendered output to {fout}")

    def compare(
        self,
        output_dir: str,
        fname: str = "compare.html",
        template: str = "comparison.html",
    ):
        inspector = self.inspector
        section_names = {
            "Overview": "Overview",
            "Flow": "Flow",
            "BackgroundError": "Background Errors",
        }

        sections = inspector.compare()

        logging.info("Rendering output...")
        output = self.env.get_template(template).render(
            title="Riptide",
            section_names=section_names,
            **sections,
        )
        os.makedirs(output_dir, exist_ok=True)
        with open(os.path.join(output_dir, fname), "w") as f:
            f.writelines(output)

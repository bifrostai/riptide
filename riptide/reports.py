from io import BytesIO
import logging
import os
import requests
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
        self.env = Environment(loader=FileSystemLoader("static"), autoescape=True)
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
        output_dir: str | None = None,
        fname: str = "report.html",
        template: str = "evaluation.html",
        *,
        evaluator_id: str = 0,
    ):
        inspector = self.inspector

        # Summary data
        overall_summary, classwise_summary, _ = inspector.overview()

        # Error data - figures and plots for each error type
        sections, section_names = inspector.inspect(evaluator_id=evaluator_id)

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

        logging.info("Rendering output...")
        output = self.env.get_template(template).render(
            title="Riptide",
            section_names=section_names,
            sections=sections,
            summary=overall_summary,
            classwise_summary=classwise_summary,
            infoboxes=infoboxes,
            missed_size_var=missed_size_var,
            missed_aspect_var=missed_aspect_var,
        )

        if output_dir is None:
            file_stream = BytesIO(output.encode("utf-8"))
            files = {"file": (fname, file_stream, "text/plain")}
            instance_url = os.environ.get("BIFROST_INSTANCE_URL")
            upload_key = os.environ.get("RIPTIDE_UPLOAD_KEY")
            url = f"https://riptide.{instance_url}/api/upload/{upload_key}"
            response = requests.post(
                url,
                files=files,
            )
            if response.status_code != 200:
                raise Exception(
                    f"Failed to upload report to {url}: {response.status_code} {response.text}"
                )
            logging.info(f"Uploaded report to {response.json().get('url')}")

        else:
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
        for idx in [0, 1]:
            if idx not in inspector._generated_crops:
                inspector.inspect(evaluator_id=idx)

        sections, section_names = inspector.compare()

        logging.info("Rendering output...")
        output = self.env.get_template(template).render(
            title="Riptide",
            section_names=section_names,
            sections=sections,
        )
        os.makedirs(output_dir, exist_ok=True)
        fout = os.path.join(output_dir, fname)
        with open(fout, "w") as f:
            f.writelines(output)
        logging.info(f"Rendered output to {fout}")

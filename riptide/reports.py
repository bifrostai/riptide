import os

from jinja2 import Environment, FileSystemLoader

from riptide.detection.visualization import (
    inspect_background_error,
    inspect_classification_error,
)


class HtmlReport:
    def __init__(self, evaluator: "Evaluator"):
        self.evaluator = evaluator
        self.env = Environment(loader=FileSystemLoader("static"), autoescape=True)
        self.template = self.env.get_template("template.html")

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
        summary = {k: round(v, 3) for k, v in self.evaluator.summarize().items()}
        background_error_figs = inspect_background_error(self.evaluator)
        background_error_figs = dict(
            sorted(
                background_error_figs.items(),
                key=lambda x: len(x[1]),
                reverse=True,
            )
        )
        (
            classification_error_figs,
            classification_error_plot,
        ) = inspect_classification_error(self.evaluator)
        output = self.template.render(
            title="Riptide",
            section_names=section_names,
            summary=summary,
            background_error_figs=background_error_figs,
            classification_error_figs=classification_error_figs,
            classification_error_plot=classification_error_plot,
        )
        os.makedirs("output", exist_ok=True)
        with open("output/report.html", "w") as f:
            f.writelines(output)

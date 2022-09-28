import argparse

from riptide.detection.evaluation import ObjectDetectionEvaluator
from riptide.reports import HtmlReport

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-targets", "-t", type=str)
    parser.add_argument("-predictions", "-p", type=str)
    parser.add_argument("-image_dir", "-i", type=str)
    parser.add_argument("-conf_threshold", "-c", type=float, default=0.5)

    args = parser.parse_args()

    evaluator: ObjectDetectionEvaluator = ObjectDetectionEvaluator.from_dicts(
        targets_dict_file=args.targets,
        predictions_dict_file=args.predictions,
        image_dir=args.image_dir,
        conf_threshold=args.conf_threshold,
    )

    HtmlReport(evaluator).render()

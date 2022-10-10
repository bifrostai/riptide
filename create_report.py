import argparse

from riptide.detection.evaluation import ObjectDetectionEvaluator
from riptide.reports import HtmlReport

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-targets", "-t", type=str, required=True)
    parser.add_argument("-predictions", "-p", type=str, required=True)
    parser.add_argument("-image_dir", "-i", type=str, required=True)
    parser.add_argument("-conf_threshold", "-c", type=float, required=False)
    # summary.json file to retrieve best_conf_threshold
    parser.add_argument("-summary_file", "-s", type=str, required=False)

    args = parser.parse_args()

    evaluator: ObjectDetectionEvaluator = ObjectDetectionEvaluator.from_dicts(
        targets_dict_file=args.targets,
        predictions_dict_file=args.predictions,
        image_dir=args.image_dir,
        conf_threshold=args.conf_threshold,
        summary_file=args.summary_file,
    )

    HtmlReport(evaluator).render()

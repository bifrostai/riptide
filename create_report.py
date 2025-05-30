import argparse
from dotenv import load_dotenv

from riptide.detection.evaluation import ObjectDetectionEvaluator
from riptide.reports import HtmlReport

# Load environment variables from .env file
load_dotenv()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-targets", "-t", type=str, required=True)
    parser.add_argument("-predictions", "-p", type=str, required=True)
    parser.add_argument("-image_dir", "-i", type=str, required=True)
    parser.add_argument("-conf_threshold", "-c", type=float, default=0.5)

    args = parser.parse_args()

    evaluator: ObjectDetectionEvaluator = ObjectDetectionEvaluator.from_coco(
        targets_file=args.targets,
        predictions_file=args.predictions,
        image_dir=args.image_dir,
        conf_threshold=args.conf_threshold,
    )

    HtmlReport(evaluator).render()

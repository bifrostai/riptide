# Using Riptide
Riptide is built upon the [TIDE](https://arxiv.org/abs/2008.08115 "TIDE: A General Toolbox for Identifying Object Detection Errors") framework, the modern standard for object detection evaluation. It is designed to help you quickly identify and understand the types of errors your model is making.


## Installation
```bash
git clone github.com/bifrost-core/riptide
poetry install
```

## Getting Started
For any quantitative evaluation, you need: <br>
- Ground truths (targets) <br>
- Predictions

Riptide allows you to quickly evaluate object detection models using the following API:
```python
from riptide.detection.evaluation import ObjectDetectionEvaluator

evaluator = ObjectDetectionEvaluator.from_dict(pt_dict_file) # .pt results
evaluator = ObjectDetectionEvaluator.from_coco(coco_pred_file, coco_gt_file) # coco predictions and targets
```

### Report Generation
Riptide supports HTML report generation using `jinja2`. Here is a minimal example of generating a report from an evaluator object:

```python
from riptide.detection.evaluation import ObjectDetectionEvaluator
from riptide.reports import HtmlReport

evaluator = ObjectDetectionEvaluator.from_dicts(
    targets_dict_file="targets.pt",
    predictions_dict_file="predictions.pt",
    image_dir="path/to/images",
    conf_threshold=0.5,
)
print(evaluator.summarize())
report = HtmlReport(evaluator).render("path/to/output/folder")
```
Read More: [:material-book: Evaluation Report](../guides/evaluation/index.md#evaluation-report)

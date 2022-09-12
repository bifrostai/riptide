# Riptide

## Installation
```
poetry install
```

## Getting Started
For any quantitative evaluation, you need:
- Ground truths (targets)
- Predictions

Riptide allows you to quickly evaluate object detection models using the following API:
```
from riptide.detection.evaluation import ObjectDetectionEvaluator

evaluator = ObjectDetectionEvaluator.from_dict(pt_dict_file) # .pt results
evaluator = ObjectDetectionEvaluator.from_coco(coco_pred_file, coco_gt_file) # coco predictions and targets
```

## Understanding Evaluations
To obtain a summary of the predictions, use `summarize()`. This gives a breakdown of each error type, the confusion counts, and the precision, recall and F1 score.
```
evaluator.summarize()
```

To obtain a classwise summary of the predictions, use `classwise_summarize()`. This gives a classwise breakdown of the above.
```
evaluator.classwise_summarize()
```

## Inspecting Individual Images
To diagnose the error for a single image, you can access the `evaluations` attribute of the evaluator.
```
print(evaluator.evaluations[0])

>>> ObjectDetectionEvaluation(pred=2, gt=1)
myimage_000.png
                gt0
================================================
        cls     3       cfu     err     score
================================================
pred0   3       0.94    [TP]     ---    0.98
pred1   3       0.47    [UN]     ---    0.43
================================================
        cfu     [TP]
        err      --- 
```
This indicates that in this image, there was one ground truth (`gt0`), two predictions (`pred0` and `pred1`). There was one true positive (TP), and one **unused prediction** (UN), since its score (0.43) is below the confidence threshold above which predictions are considered (default: 0.5). If this were to be lowered, then `pred1` would be considered a false positive (FP), and further classified into one of the error types (in this case, a LocalizationError):
```
                gt0
================================================
        cls     3       cfu     err     score
================================================
pred0   3       0.94    [TP]     ---    0.98
pred1   3       0.47    [FP]    [LOC]   0.43
================================================
        cfu     [TP]
        err      --- 
```
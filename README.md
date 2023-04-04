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

### Report Generation
Riptide supports HTML report generation using `jinja2`. Here is a minimal example of generating a report from an evaluator object:

```
from riptide.detection.evaluation import ObjectDetectionEvaluator
from riptide.reports import HtmlReport

evaluator = ObjectDetectionEvaluator.from_dicts(
    targets_dict_file="targets.pt",
    predictions_dict_file="predictions.pt",
    image_dir="maritime",
    conf_threshold=0.5,
)
print(evaluator.summarize())
report = HtmlReport(evaluator).render("path/to/output/folder")
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

## Inspecting the changes in status of Ground Truths across different models
To evaluate if the changes made to the model (e.g. changing model hyperparameters, changing the training datset) were successful in solving the problem (e.g. Did changing XXX help to resolve these Localization Errors? What are their status now?), we can use the `models_gt_flow` functions to identify the change in state of ground truth across two different models (note that both models need to use the same ground truth images and annotations).

The magnitude of the change in status of ground truths can be visualized overall in terms of a sankey diagram, and the individual ground truth in a specific flow (e.g. LOC to TP) can be visualized in terms of a montage.

### NOTE

Note that the localization error of each model as reported by the gt_flow class will be less than or equal to the actual number of localization errors. This is because the same ground truth can have both a localization error and a True Positive box drawn for it. However, in the current implementation, a ground truth bounding box can only be of a single state at any time. Therefore, if a ground truth bounding box only has a localization error drawn around the ground truth, then it obviously has a status of 'LOC', but if has both a localization error and True Positive drawn around it, then it will be considered a 'TP' by virtue of that correct bounding box drawn around it.

This has implications in the visualization of the flows -- there may have been a great decrease in the LOC errors, but it may not show up in the flow as those ground truths that have a LOC error and a TP is already considered a TP, therefore no new change in status is observed by gt_flow class

Below is an example of how the sankey diagram and montage of a flow can be created. Currently this code has only been tested to work for Single Class GT flows; more functionalities should be added if we want to analyse gt flows within a single class in multi class models

```
from riptide.models_gt_flow import *

## define the relevant paths and thresholds here

print("Instantiate gt_flow class...")
flow = gt_flow(
    INITIAL_PRED_PATH,
    INITIAL_PRED_CONF_THRESHOLD,
    SECOND_PRED_PATH,
    SECOND_PRED_CONF_THRESHOLD,
    TARGET_PATH,
    IMG_DIR,
    COCO_ANNOTATIONS)

print("Instantiate Sankey Diagram...")
flow.instantiate_sankey_diagram(
    title_text= "Ground Truth Sankey Diagram for Single Class")

print("Showing Sankey Diagram...")
flow.sankey.show()

print("Saving Sankey Diagram...")
flow.save_sankey_diagram(
    output_path= "sankey.png")

possible_statuses = ["LOC", "MIS", "TP"]

for i in range(len(possible_statuses)):
    for j in range(len(possible_statuses)):
        m1_status = possible_statuses[i]
        m2_status = possible_statuses[j]
        print(f"Saving Flow Montage for Model 1 {m1_status} to Model 2 {m2_status}...")
        flow.save_flow(
            m1_status= m1_status,
            m2_status= m2_status,
            output_path= f"{m1_status}_{m2_status}.png")
```

## Understanding Error Types
There are three threshold values:
- Background IoU threshold `bg_iou_threshold`: Detections smaller than this level are not considered
- Foreground IoU threshold `fg_iou_threshold`: Detections must be >= this level to be considered **correct**
- Confidence threshold `conf_threshold`: Detections must be >= this confidence to be considered

### BackgroundError
- Is above `conf_threshold` and does not meet the `bg_iou_threshold` with any ground truth
- Counts as a false positive to the predicted class

### ClassificationError
- Is above `conf_threshold` and `fg_iou_threshold`, but the class label is incorrect
- Counts as a false positive to the predicted class
- Counts as a false negative to the ground truth class (missed it)

### LocalizationError
- Is above `conf_threshold` and is between `bg_iou_threshold` and `fg_iou_threshold`, and the class label is **correct**
- Counts as a false positive to the predicted class
- Counts as a false negative to the ground truth class (missed it)

### ClassificationAndLocalizationError
- Is above `conf_threshold` and is between `bg_iou_threshold` and `fg_iou_threshold`, and the class label is **incorrect**
- Counts as a false positive to the predicted class
- Counts as a false negative to the ground truth class (missed it)

### DuplicateError
- Is above `conf_threshold` and `fg_iou_threshold`, but a simultaneous valid prediction has been made (true positive), which has a higher IoU than this one
- Counts as a false positive to the predicted class

### MissedError
- No prediction was made above `conf_threshold` that had IoU above `bg_iou_threshold` (otherwise it would be considered a `LocalizationError`)
- Counts as a false negative to the ground truth class

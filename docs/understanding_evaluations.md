# Understanding Evaluations
To obtain a summary of the predictions, use `summarize()`. This gives a breakdown of each error type, the confusion counts, and the precision, recall and F1 score.
```python
evaluator.summarize()
```

To obtain a classwise summary of the predictions, use `classwise_summarize()`. This gives a classwise breakdown of the above.
```python
evaluator.classwise_summarize()
```

## Inspecting Individual Images
To diagnose the error for a single image, you can access the `evaluations` attribute of the evaluator. Each evaluation corresponds to a single image, and contains the ground truths and predictions for that image.
```python
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

### Visualizing detections
To visualize the detections for an evaluation, you can use the `draw_image_errors()` method. This will return a PyTorch tensor of the image with the ground truths and predictions drawn on it.
```python
from IPython.display import display
from torchvision.transforms.functional import to_pil_image

evaluation = evaluator.evaluations[0]
image = evaluation.draw_image_errors()
display(to_pil_image(image))
```

## Evaluation Report
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

## Sections in the Evaluation Report
The report is divided into the following sections:

### Overview
This section provides a summary of the performance of the model, in terms of the number of ground truths, predictions, and the error distribution for each model.

### Error Visualization
These sections provide visualizations of the errors for each error type. The errors are grouped by error type, class, and perceptual similarity, in order. Perceptual similarity is determined by computing cluster labels for the feature embeddings of ground truths and background errors, using the [HDBSCAN algorithm](https://github.com/scikit-learn-contrib/hdbscan). The cluster labels are then used to group the errors into perceptually similar groups.

Predictions are grouped into the following categories:
- **Missed Errors (MIS)**: Ground truths that were not detected by the model.
- **Background Errors (BKG)**: Predictions that do not correspond to any ground truth.
- **Confusions (CLS + CLL)**: Predictions that correspond to a ground truth, but are classified as a different class.
- **Localization Errors (LOC)**: Predictions that correspond to a ground truth, but have poor localization.
- **Duplicate Errors (DUP)**: Predictions that correspond to a ground truth, but are duplicate detections.

For more information on the error types, see [Understanding Error Types](error_types.md).

### Visualization of True Positives
This section provides visualizations of the true positives for each class. The true positives are grouped by class, and perceptual similarity, in order.

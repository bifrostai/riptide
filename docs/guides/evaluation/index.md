# Understanding Evaluations

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
report = HtmlReport(evaluator).render("path/to/output/folder")
```

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
```pycon
>>> print(evaluator.evaluations[0])
ObjectDetectionEvaluation(pred=2, gt=1)
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
This indicates that in this image, there was one ground truth (`gt0`) and two predictions (`pred0` and `pred1`). There was one true positive (TP), and one **unused prediction** (UN), since its score (0.43) is below the confidence threshold above which predictions are considered (default: 0.5). If this were to be lowered, then `pred1` would be considered a false positive (FP), and further classified into one of the error types&mdash; in this case, a Localization Error (LOC):
```pycon
>>> print(evaluator.evaluations[0])
ObjectDetectionEvaluation(pred=2, gt=1)
myimage_000.png
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

### Visualizing Detections
To visualize the detections for an evaluation, you can use the `draw_image_errors()` method. This will return a PyTorch tensor of the image with the ground truths and predictions drawn on it.
```python
from IPython.display import display
from torchvision.transforms.functional import to_pil_image

evaluation = evaluator.evaluations[0]
image = evaluation.draw_image_errors()
display(to_pil_image(image))
```

## Evaluation Report

Below, we will go through the different sections of the report.

### Overview
This section provides a summary of the performance of the model, in terms of the number of ground truths, predictions, and the error distribution for each model.

Below is an example of the overview section:

<section id="example-overview" class="example">
<h2>Overview</h2>
<div class="summary-container">
    <div class="summary-row"><div class="summary-item">
        <div class="summary-metric-title gradient">
            No. of Images
        </div>
        <div class="summary-metric-value">80</div>
    </div>
    <div class="summary-item">
        <div class="summary-metric-title gradient">
            No. of Objects
        </div>
        <div class="summary-metric-value">113</div>
    </div>
    <div class="summary-item">
        <div class="summary-metric-title gradient">
            Conf. Threshold
        </div>
        <div class="summary-metric-value">0.5</div>
    </div>
    <div class="summary-item">
        <div class="summary-metric-title gradient">
            IoU Threshold
        </div>
        <div class="summary-metric-value">0.1 - 0.5</div>
    </div>
</div>
<div class="summary-row">
    <div class="summary-item">
        <div class="summary-metric-title gradient">
            Precision
        </div>
        <div class="summary-metric-value">0.32</div>
    </div>
    <div class="summary-item">
        <div class="summary-metric-title gradient">
            Recall
        </div>
        <div class="summary-metric-value">0.62</div>
    </div>
    <div class="summary-item">
        <div class="summary-metric-title gradient">
                F1
        </div>
        <div class="summary-metric-value">0.43</div>
    </div>
    <div class="summary-item">
        <div class="summary-metric-title gradient">
            Unused
        </div>
        <div class="summary-metric-value">50</div>
    </div>
    <div class="break"></div>
    <div class="summary-item" style="flex-grow: 1">
        <span class="summary-metric-title gradient">Ground Truths</span>
        <div class="summary-metric-value">
            <div class="summary-bar">
                <span class="summary-bar-value">113</span>
                <div class="summary-bar-item first" style="--bar-width: 70; --bar-color: rgba(0, 255, 0, 0.8);">
                    <span class="label">70</span>
                    <span class="tooltiptext">True Positives</span>
                </div>
                <div class="summary-bar-item" style="--bar-width: 16; --bar-color: rgba(154, 205, 50, 0.8);">
                    <span class="label">16</span>
                    <span class="tooltiptext">Missed</span>
                </div>
                <div class="summary-bar-item last" style="--bar-width: 27; --bar-color: rgba(255, 0, 0, 0.8);">
                    <span class="label">27</span>
                    <span class="tooltiptext">False Negatives</span>
                </div>
            </div>
        </div>
    </div>
    <div class="break"></div>
    <div class="summary-item" style="flex-grow: 1">
        <span class="summary-metric-title gradient">Predictions</span>
        <div class="summary-metric-value">
             <div class="summary-bar">
                <span class="summary-bar-value">216</span>
                <div class="summary-bar-item first" style="--bar-width: 70; --bar-color: rgba(0, 255, 0, 0.8);">
                    <span class="label">70</span>
                    <span class="tooltiptext">True Positives</span>
                </div>
                <div class="summary-bar-item" style="--bar-width: 24; --bar-color: rgba(255, 0, 255, 0.8);">
                    <span class="label">24</span>
                    <span class="tooltiptext">Background</span>
                </div>
                <div class="summary-bar-item" style="--bar-width: 16; --bar-color: rgba(220, 20, 60, 0.8);">
                    <span class="label">16</span>
                    <span class="tooltiptext">Classification</span>
                </div>
                <div class="summary-bar-item" style="--bar-width: 21; --bar-color: rgba(255, 69, 0, 0.8);">
                    <span class="label">21</span>
                    <span class="tooltiptext">Classification and Localization</span>
                </div>
                <div class="summary-bar-item" style="--bar-width: 74; --bar-color: rgba(255, 215, 0, 0.8);">
                    <span class="label">74</span>
                    <span class="tooltiptext">Localization</span>
                </div>
                <div class="summary-bar-item last" style="--bar-width: 11; --bar-color: rgba(0, 255, 255, 0.8);">
                    <span class="label">11</span>
                    <span class="tooltiptext">Duplicate</span>
                </div>
            </div>
        </div>
    </div>
</div>
</section>

???note "A Note on Precision"
    One common problem of evalutating computer vision models is that datasets often contain a large number of highly similar images, especially if a dataset is derived from videos. Consequently, failure cases in a model often occur multiple times in a dataset.

    This is not a problem for recall, as a model should be penalized for each missed target, regardless of whether it is similar to other targets. However, the conventional definition of precision is not an accurate representation of a model's performance, as a model can be penalized multiple times for the same underlying error.

    To account for this, Riptide suppresses the number of false positives by only counting similar false positives once, based on [perceptual similarity](similarity.md). This is a more accurate representation of precision, as a model is not repeatedly penalized for making multiple predictions for similar objects.


### Error Visualization
These sections provide visualizations of the errors for each error type. The errors are grouped by error type, class, and [perceptual similarity](similarity.md), in order.

Predictions are grouped into the following categories:

- **Missed Errors (MIS)**{id="error-mis"}: Ground truths that were not detected by the model.
- **Background Errors (BKG)**{id="error-bkg"}: Predictions that do not correspond to any ground truth.
- **Confusions (CLS + CLL)**{id="error-confusions"}: Predictions that correspond to a ground truth, but are classified as a different class.
- **Localization Errors (LOC)**{id="error-loc"}: Predictions that correspond to a ground truth, but have poor localization.
- **Duplicate Errors (DUP)**{id="error-dup"}: Predictions that correspond to a ground truth, but are duplicate detections.

For more information on the error types, see [Understanding Error Types](../error_types.md).

### Visualization of True Positives
This section provides visualizations of the true positives for each class. The true positives are grouped by class, and [perceptual similarity](similarity.md), in order.

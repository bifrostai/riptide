# Comparing Models
> (new in v0.3.0)

Riptide supports comparing the performance of two models. Here is a minimal example of generating a comparison report, given two `Evaluator` objects:

```python
from riptide.detection.evaluation import ObjectDetectionEvaluator
from riptide.reports import HtmlReport

evaluator1 = ObjectDetectionEvaluator.from_dicts(
    targets_dict_file="targets.pt",
    predictions_dict_file="predictions2.pt",
    image_dir="path/to/images",
    conf_threshold=0.5,
)

evaluator2 = ObjectDetectionEvaluator.from_dicts(
    targets_dict_file="targets.pt",
    predictions_dict_file="predictions2.pt",
    image_dir="path/to/images",
    conf_threshold=0.5,
)

report = HtmlReport([evaluator1, evaluator2]).compare("path/to/output/folder")
```

## Sections in the Comparison Report
The comparison report is divided into the following sections:
### Overview
This section provides a summary of the performance of the two models, in terms of the number of ground truths, predictions, and the error distribution for each model.

### Flow
This section visualizes the flow of ground truths across the two models, in terms of the error types. The flow is visualized in terms of a sankey diagram, and the individual ground truth in a specific flow (e.g. LOC to TP) can be visualized in terms of a montage.
[Read More - Comparing Flow](#comparing-flow)

### Background Errors
This section visualizes the background errors for each model, grouped by visual similarity.

### Degradations and Improvements
These sections visualize the ground truths that have degraded or improved in status across the two models. Aside from background errors, all other predictions (or misses) are associated with a ground truth. Therefore, we can quantify the level of improvement or degradation in the prediction of a ground truth by comparing the error types of the associated prediction or miss across the two models.

**Degradations** are ground truths that were correctly predicted by the first model, but incorrectly predicted by the second model.

**Improvements** are ground truths that were incorrectly predicted by the first model, but correctly predicted by the second model.


## Comparing Flow
To evaluate if the changes made to the model (e.g. changing model hyperparameters, changing the training datset) were successful in solving the problem (e.g. Did changing XXX help to resolve these Localization Errors? What are their status now?), we can use the module `riptide.flow` to identify the change in state of ground truth across two different models.

The magnitude of the change in status of ground truths can be visualized overall in terms of a sankey diagram, and the individual ground truth in a specific flow (e.g. LOC to TP) can be visualized in terms of a montage.

Below is an example of how the sankey diagram and montage of a flow can be created.

```python
from riptide.detection.evaluation import Evaluator
from riptide.flow import FlowVisualizer

## Instantiate Evaluator instances for each model and define path to image directory
evaluators: list[Evaluator]
IMAGE_DIR: str = "/path/to/images"

## Instantiate FlowVisualizer
flow = FlowVisualizer(evaluators, IMAGE_DIR)

## Show figure
flow.visualize().show()
```

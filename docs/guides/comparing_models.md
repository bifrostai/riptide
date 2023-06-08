---
plotly: true
---

# Comparing Models {.subtitled}
> new in v0.3.0

<br>

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
> See also: [Evaluation Report - Overview](evaluation/index.md#overview)

This section provides a summary of the performance of the two models, in terms of the number of ground truths, predictions, and the error distribution for each model.


### Flow
> Read more: [Comparing Flow](#comparing-flow)

This section visualizes the relative performance of the two models in predicting the ground truths. To do so, we assign a status to each ground truth for each model.
!!! question "Ground Truth Status"
    The status of a ground truth is either:

    1. A prediction (non-)error with the highest IoU (i.e. TP, LOC, CLS, CLL)
    2. A missed error (MIS)

With the assigned statuses, we can visualize the change in status of each ground truth across the two models. In Riptide, we do so using a Sankey diagram.

### Background Errors
This section visualizes the background errors for each model, grouped by [perceptual similarity](evaluation/similarity.md).

### Degradations and Improvements
These sections visualize the ground truths that have degraded or improved in status across the two models. Aside from background errors, all other predictions (or misses) are associated with a ground truth. Therefore, we can quantify the level of improvement or degradation in the prediction of a ground truth by comparing the error types of the associated prediction or miss across the two models.

**Degradations** are ground truths that were correctly predicted by the first model, but incorrectly predicted by the second model.

**Improvements** are ground truths that were incorrectly predicted by the first model, but correctly predicted by the second model.

> See also: [Interpreting the Flow Diagram](#interpreting-the-flow-diagram)


## Comparing Flow
To evaluate if the changes made to the model (e.g. changing model hyperparameters, changing the training datset) were successful in solving the problem (e.g. Did changing XXX help to resolve these Localization Errors? What are their status now?), we can use the module `riptide.flow` to identify the change in state of ground truth across two different models.

The magnitude of the change in status of ground truths can be visualized overall in terms of a Sankey diagram, and the individual ground truth in a specific flow (e.g. LOC to TP) can be visualized in terms of a montage.

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
### Interpreting the Flow Diagram
<div class="example">
    <div id="a7f1c339-0e59-46d5-8d59-42866a48b3a9" class="plotly-graph-div" style="height:100%; width:100%;"></div>            <script type="text/javascript">                                    window.PLOTLYENV=window.PLOTLYENV || {};                                    if (document.getElementById("a7f1c339-0e59-46d5-8d59-42866a48b3a9")) {                    Plotly.newPlot(                        "a7f1c339-0e59-46d5-8d59-42866a48b3a9",                        [{"link":{"color":["rgba(0, 255, 0, 0.3)","rgba(0, 255, 0, 0.3)","rgba(0, 255, 0, 0.3)","rgba(0, 255, 0, 0.3)","rgba(0, 255, 0, 0.3)","rgba(255, 215, 0, 0.3)","rgba(255, 215, 0, 0.3)","rgba(255, 215, 0, 0.3)","rgba(255, 215, 0, 0.3)","rgba(255, 215, 0, 0.3)","rgba(220, 20, 60, 0.3)","rgba(220, 20, 60, 0.3)","rgba(220, 20, 60, 0.3)","rgba(220, 20, 60, 0.3)","rgba(220, 20, 60, 0.3)","rgba(255, 69, 0, 0.3)","rgba(255, 69, 0, 0.3)","rgba(255, 69, 0, 0.3)","rgba(255, 69, 0, 0.3)","rgba(255, 69, 0, 0.3)","rgba(154, 205, 50, 0.3)","rgba(154, 205, 50, 0.3)","rgba(154, 205, 50, 0.3)","rgba(154, 205, 50, 0.3)","rgba(154, 205, 50, 0.3)"],"label":["0.00","0.00","1.00","0.00","0.00","1.00","-1.00","-1.00","0.00"],"source":[0,0,0,0,0,1,1,1,1,1,2,2,2,2,2,3,3,3,3,3,4,4,4,4,4],"target":[5,6,7,8,9,5,6,7,8,9,5,6,7,8,9,5,6,7,8,9,5,6,7,8,9],"value":[82,10,0,0,15,38,15,0,0,0,12,0,23,0,0,0,0,30,0,0,20,15,0,0,10]},"node":{"color":["rgba(0, 255, 0, 0.8)","rgba(255, 215, 0, 0.8)","rgba(220, 20, 60, 0.8)","rgba(255, 69, 0, 0.8)","rgba(154, 205, 50, 0.8)","rgba(0, 255, 0, 0.8)","rgba(255, 215, 0, 0.8)","rgba(220, 20, 60, 0.8)","rgba(255, 69, 0, 0.8)","rgba(154, 205, 50, 0.8)"],"label":["Model 1 TP","Model 1 LOC","Model 1 CLS","Model 1 CLL","Model 1 MIS","Model 2 TP","Model 2 LOC","Model 2 CLS","Model 2 CLL","Model 2 MIS"],"line":{"color":"black","width":0.1},"pad":15,"thickness":20},"type":"sankey"}],                        {"font":{"size":10},"margin":{"b":10,"l":20,"r":20,"t":30},"paper_bgcolor":"rgba(0,0,0,0)","plot_bgcolor":"rgba(0,0,0,0)","template":{"data":{"barpolar":[{"marker":{"line":{"color":"white","width":0.5},"pattern":{"fillmode":"overlay","size":10,"solidity":0.2}},"type":"barpolar"}],"bar":[{"error_x":{"color":"rgb(36,36,36)"},"error_y":{"color":"rgb(36,36,36)"},"marker":{"line":{"color":"white","width":0.5},"pattern":{"fillmode":"overlay","size":10,"solidity":0.2}},"type":"bar"}],"carpet":[{"aaxis":{"endlinecolor":"rgb(36,36,36)","gridcolor":"white","linecolor":"white","minorgridcolor":"white","startlinecolor":"rgb(36,36,36)"},"baxis":{"endlinecolor":"rgb(36,36,36)","gridcolor":"white","linecolor":"white","minorgridcolor":"white","startlinecolor":"rgb(36,36,36)"},"type":"carpet"}],"choropleth":[{"colorbar":{"outlinewidth":1,"tickcolor":"rgb(36,36,36)","ticks":"outside"},"type":"choropleth"}],"contourcarpet":[{"colorbar":{"outlinewidth":1,"tickcolor":"rgb(36,36,36)","ticks":"outside"},"type":"contourcarpet"}],"contour":[{"colorbar":{"outlinewidth":1,"tickcolor":"rgb(36,36,36)","ticks":"outside"},"colorscale":[[0.0,"#440154"],[0.1111111111111111,"#482878"],[0.2222222222222222,"#3e4989"],[0.3333333333333333,"#31688e"],[0.4444444444444444,"#26828e"],[0.5555555555555556,"#1f9e89"],[0.6666666666666666,"#35b779"],[0.7777777777777778,"#6ece58"],[0.8888888888888888,"#b5de2b"],[1.0,"#fde725"]],"type":"contour"}],"heatmapgl":[{"colorbar":{"outlinewidth":1,"tickcolor":"rgb(36,36,36)","ticks":"outside"},"colorscale":[[0.0,"#440154"],[0.1111111111111111,"#482878"],[0.2222222222222222,"#3e4989"],[0.3333333333333333,"#31688e"],[0.4444444444444444,"#26828e"],[0.5555555555555556,"#1f9e89"],[0.6666666666666666,"#35b779"],[0.7777777777777778,"#6ece58"],[0.8888888888888888,"#b5de2b"],[1.0,"#fde725"]],"type":"heatmapgl"}],"heatmap":[{"colorbar":{"outlinewidth":1,"tickcolor":"rgb(36,36,36)","ticks":"outside"},"colorscale":[[0.0,"#440154"],[0.1111111111111111,"#482878"],[0.2222222222222222,"#3e4989"],[0.3333333333333333,"#31688e"],[0.4444444444444444,"#26828e"],[0.5555555555555556,"#1f9e89"],[0.6666666666666666,"#35b779"],[0.7777777777777778,"#6ece58"],[0.8888888888888888,"#b5de2b"],[1.0,"#fde725"]],"type":"heatmap"}],"histogram2dcontour":[{"colorbar":{"outlinewidth":1,"tickcolor":"rgb(36,36,36)","ticks":"outside"},"colorscale":[[0.0,"#440154"],[0.1111111111111111,"#482878"],[0.2222222222222222,"#3e4989"],[0.3333333333333333,"#31688e"],[0.4444444444444444,"#26828e"],[0.5555555555555556,"#1f9e89"],[0.6666666666666666,"#35b779"],[0.7777777777777778,"#6ece58"],[0.8888888888888888,"#b5de2b"],[1.0,"#fde725"]],"type":"histogram2dcontour"}],"histogram2d":[{"colorbar":{"outlinewidth":1,"tickcolor":"rgb(36,36,36)","ticks":"outside"},"colorscale":[[0.0,"#440154"],[0.1111111111111111,"#482878"],[0.2222222222222222,"#3e4989"],[0.3333333333333333,"#31688e"],[0.4444444444444444,"#26828e"],[0.5555555555555556,"#1f9e89"],[0.6666666666666666,"#35b779"],[0.7777777777777778,"#6ece58"],[0.8888888888888888,"#b5de2b"],[1.0,"#fde725"]],"type":"histogram2d"}],"histogram":[{"marker":{"line":{"color":"white","width":0.6}},"type":"histogram"}],"mesh3d":[{"colorbar":{"outlinewidth":1,"tickcolor":"rgb(36,36,36)","ticks":"outside"},"type":"mesh3d"}],"parcoords":[{"line":{"colorbar":{"outlinewidth":1,"tickcolor":"rgb(36,36,36)","ticks":"outside"}},"type":"parcoords"}],"pie":[{"automargin":true,"type":"pie"}],"scatter3d":[{"line":{"colorbar":{"outlinewidth":1,"tickcolor":"rgb(36,36,36)","ticks":"outside"}},"marker":{"colorbar":{"outlinewidth":1,"tickcolor":"rgb(36,36,36)","ticks":"outside"}},"type":"scatter3d"}],"scattercarpet":[{"marker":{"colorbar":{"outlinewidth":1,"tickcolor":"rgb(36,36,36)","ticks":"outside"}},"type":"scattercarpet"}],"scattergeo":[{"marker":{"colorbar":{"outlinewidth":1,"tickcolor":"rgb(36,36,36)","ticks":"outside"}},"type":"scattergeo"}],"scattergl":[{"marker":{"colorbar":{"outlinewidth":1,"tickcolor":"rgb(36,36,36)","ticks":"outside"}},"type":"scattergl"}],"scattermapbox":[{"marker":{"colorbar":{"outlinewidth":1,"tickcolor":"rgb(36,36,36)","ticks":"outside"}},"type":"scattermapbox"}],"scatterpolargl":[{"marker":{"colorbar":{"outlinewidth":1,"tickcolor":"rgb(36,36,36)","ticks":"outside"}},"type":"scatterpolargl"}],"scatterpolar":[{"marker":{"colorbar":{"outlinewidth":1,"tickcolor":"rgb(36,36,36)","ticks":"outside"}},"type":"scatterpolar"}],"scatter":[{"fillpattern":{"fillmode":"overlay","size":10,"solidity":0.2},"type":"scatter"}],"scatterternary":[{"marker":{"colorbar":{"outlinewidth":1,"tickcolor":"rgb(36,36,36)","ticks":"outside"}},"type":"scatterternary"}],"surface":[{"colorbar":{"outlinewidth":1,"tickcolor":"rgb(36,36,36)","ticks":"outside"},"colorscale":[[0.0,"#440154"],[0.1111111111111111,"#482878"],[0.2222222222222222,"#3e4989"],[0.3333333333333333,"#31688e"],[0.4444444444444444,"#26828e"],[0.5555555555555556,"#1f9e89"],[0.6666666666666666,"#35b779"],[0.7777777777777778,"#6ece58"],[0.8888888888888888,"#b5de2b"],[1.0,"#fde725"]],"type":"surface"}],"table":[{"cells":{"fill":{"color":"rgb(237,237,237)"},"line":{"color":"white"}},"header":{"fill":{"color":"rgb(217,217,217)"},"line":{"color":"white"}},"type":"table"}]},"layout":{"annotationdefaults":{"arrowhead":0,"arrowwidth":1},"autotypenumbers":"strict","coloraxis":{"colorbar":{"outlinewidth":1,"tickcolor":"rgb(36,36,36)","ticks":"outside"}},"colorscale":{"diverging":[[0.0,"rgb(103,0,31)"],[0.1,"rgb(178,24,43)"],[0.2,"rgb(214,96,77)"],[0.3,"rgb(244,165,130)"],[0.4,"rgb(253,219,199)"],[0.5,"rgb(247,247,247)"],[0.6,"rgb(209,229,240)"],[0.7,"rgb(146,197,222)"],[0.8,"rgb(67,147,195)"],[0.9,"rgb(33,102,172)"],[1.0,"rgb(5,48,97)"]],"sequential":[[0.0,"#440154"],[0.1111111111111111,"#482878"],[0.2222222222222222,"#3e4989"],[0.3333333333333333,"#31688e"],[0.4444444444444444,"#26828e"],[0.5555555555555556,"#1f9e89"],[0.6666666666666666,"#35b779"],[0.7777777777777778,"#6ece58"],[0.8888888888888888,"#b5de2b"],[1.0,"#fde725"]],"sequentialminus":[[0.0,"#440154"],[0.1111111111111111,"#482878"],[0.2222222222222222,"#3e4989"],[0.3333333333333333,"#31688e"],[0.4444444444444444,"#26828e"],[0.5555555555555556,"#1f9e89"],[0.6666666666666666,"#35b779"],[0.7777777777777778,"#6ece58"],[0.8888888888888888,"#b5de2b"],[1.0,"#fde725"]]},"colorway":["#1F77B4","#FF7F0E","#2CA02C","#D62728","#9467BD","#8C564B","#E377C2","#7F7F7F","#BCBD22","#17BECF"],"font":{"color":"rgb(36,36,36)"},"geo":{"bgcolor":"white","lakecolor":"white","landcolor":"white","showlakes":true,"showland":true,"subunitcolor":"white"},"hoverlabel":{"align":"left"},"hovermode":"closest","mapbox":{"style":"light"},"paper_bgcolor":"white","plot_bgcolor":"white","polar":{"angularaxis":{"gridcolor":"rgb(232,232,232)","linecolor":"rgb(36,36,36)","showgrid":false,"showline":true,"ticks":"outside"},"bgcolor":"white","radialaxis":{"gridcolor":"rgb(232,232,232)","linecolor":"rgb(36,36,36)","showgrid":false,"showline":true,"ticks":"outside"}},"scene":{"xaxis":{"backgroundcolor":"white","gridcolor":"rgb(232,232,232)","gridwidth":2,"linecolor":"rgb(36,36,36)","showbackground":true,"showgrid":false,"showline":true,"ticks":"outside","zeroline":false,"zerolinecolor":"rgb(36,36,36)"},"yaxis":{"backgroundcolor":"white","gridcolor":"rgb(232,232,232)","gridwidth":2,"linecolor":"rgb(36,36,36)","showbackground":true,"showgrid":false,"showline":true,"ticks":"outside","zeroline":false,"zerolinecolor":"rgb(36,36,36)"},"zaxis":{"backgroundcolor":"white","gridcolor":"rgb(232,232,232)","gridwidth":2,"linecolor":"rgb(36,36,36)","showbackground":true,"showgrid":false,"showline":true,"ticks":"outside","zeroline":false,"zerolinecolor":"rgb(36,36,36)"}},"shapedefaults":{"fillcolor":"black","line":{"width":0},"opacity":0.3},"ternary":{"aaxis":{"gridcolor":"rgb(232,232,232)","linecolor":"rgb(36,36,36)","showgrid":false,"showline":true,"ticks":"outside"},"baxis":{"gridcolor":"rgb(232,232,232)","linecolor":"rgb(36,36,36)","showgrid":false,"showline":true,"ticks":"outside"},"bgcolor":"white","caxis":{"gridcolor":"rgb(232,232,232)","linecolor":"rgb(36,36,36)","showgrid":false,"showline":true,"ticks":"outside"}},"title":{"x":0.05},"xaxis":{"automargin":true,"gridcolor":"rgb(232,232,232)","linecolor":"rgb(36,36,36)","showgrid":false,"showline":true,"ticks":"outside","title":{"standoff":15},"zeroline":false,"zerolinecolor":"rgb(36,36,36)"},"yaxis":{"automargin":true,"gridcolor":"rgb(232,232,232)","linecolor":"rgb(36,36,36)","showgrid":false,"showline":true,"ticks":"outside","title":{"standoff":15},"zeroline":false,"zerolinecolor":"rgb(36,36,36)"}}},"title":{"text":""}},                        {"displayModeBar": false, "responsive": true, "displaylogo": false}                    )                };                            </script>
</div>

Hovering over the nodes in the figure, we see the following breakdown of errors.

| Model | TP  | LOC | CLS | CLL | MIS |
| ----- | --- | --- | --- | --- | --- |
| 1     | 107 |  53 |  35 |  30 |  45 |
| 2     | 152 |  40 |  53 |   0 |  25 |

Furthermore, hovering over the edges, we can make the following observations:

1. **LOC → TP**: 38 targets were poorly localized by Model 1 but correctly detected in Model 2, indicating that Model 2 is better at localizing these targets.
2. **MIS → TP**: 20 targets that were missed by Model 1 were correctly detected in Model 2.
3. **CLL → CLS**: All [CLL](error_types.md#error-cll "Classification and Localization") errors in Model 1 became [CLS](error_types.md#error-cls "Classification") errors in Model 2, indicating that Model 2 is better at localizing the corresponding targets, but is still unable to classify them correctly.
4. **Improvements and Degradations**:
    - Incoming edges of _Model 2 TP_: There are $38+20+12 = 70$ improvements from Model 1 to Model 2.
    - Outgoing edges of _Model 1 TP_: There are $15+10=25$ degradations from Model 1 to Model 2.

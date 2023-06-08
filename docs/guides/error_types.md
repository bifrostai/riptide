# Understanding Error Types {.subtitled}
> Read more: [Alchemy Handbook - Analyzing Failures](https://bifrost-core.gitlab.io/alchemy-handbook/object-detection/analyzing-failures/)

<br>

[TIDE](https://arxiv.org/abs/2008.08115) is the modern standard for object detection analysis. The simplest way to understand the TIDE framework is through its three threshold values:

| Threshold | Default | Description |
| --- | --- | --- |
| Confidence $t_c$ | 0.5 | Predictions with a confidence below this level will be unused |
| Background IoU $\text{IoU}_{b}$ | 0.1 | Predictions with $\text{IoU}_{\max}$ less than this level are considered background predictions |
| Foreground IoU $\text{IoU}_{f}$ | 0.5 | Predictions must have $\text{IoU}_{\max}$ greater than this level to be considered correct |

where $\text{IoU}_{\max}$ is the maximum IoU of the prediction with any ground truth in the image.

## :material-volcano: Background Errors {id="error-bkg"}
???+ info "Background Error (BKG)"
    - $\text{confidence} \geq t_c$ and $\text{IoU}_{\max} < \text{IoU}_{b}$
    - Counts as a false positive to the predicted class

## :material-shape: Classification Errors {id="error-cls"}
???+ info "Classification Error (CLS)"
    - $\text{confidence} \geq t_c$ and $\text{IoU}_{\max} \geq \text{IoU}_{f}$, but the class label is incorrect
    - Counts as a false positive to the predicted class
    - Counts as a false negative to the ground truth class (missed it)

## :material-square-opacity: Localization Errors {id="error-loc"}
???+ info "Localization Error (LOC)"
    - $\text{confidence} \geq t_c$ and $\text{IoU}_{b} \leq \text{IoU}_{\max} < \text{IoU}_{f}$, and the class label is **correct**
    - Counts as a false positive to the predicted class
    - Counts as a false negative to the ground truth class (missed it)

## :material-shape: Classification and Localization Errors {id="error-cll"}
???+ info "Classification And Localization Error (CLL)"
    - $\text{confidence} \geq t_c$ and $\text{IoU}_{b} \leq \text{IoU}_{\max} < \text{IoU}_{f}$, and the class label is **incorrect**
    - Counts as a false positive to the predicted class
    - Counts as a false negative to the ground truth class (missed it)

## :material-content-duplicate: Duplicate Errors {id="error-dup"}
???+ info "Duplicate Error (DUP)"
    -  $\text{confidence} \geq t_c$ and $\text{IoU}_{\max} \geq \text{IoU}_{f}$, but a simultaneous valid prediction has been made (true positive), which has a higher $\text{IoU}_{\max}$ than this one
    - Counts as a false positive to the predicted class

## :material-square-off-outline: Missed Errors {id="error-mis"}
???+ info "Missed Error (MIS)"
    - No predictions were made with $\text{confidence} \geq t_c$ and $\text{IoU}_{\max} \geq \text{IoU}_{b}$ (otherwise it would be considered a Localization Error)
    - Counts as a false negative to the ground truth class

### Subgroups
To categorize how the missed errors are manifested, we can further subdivide the missed errors into the following subgroups:

#### **Crowded**
The bounding box for the associated ground truth overlaps with another object.

- This is determined by calculating the IoU of the ground truth with all other ground truths in the image, and if the maximum IoU is above `iou_threshold`, then the ground truth is considered crowded.
#### **Occluded**
The associated ground truth is occluded by the environment.
#### **Truncated**
The bounding box for the associated ground truth is truncated by or located close to the image boundary.

- That is, at least one of the four corners of the bounding box within `min_size // 2` pixels of the image boundary.
#### **Not Enough Visual Features**
The associated ground truth is either too small or too blur to be reasonably detected.

- A ground truth is too small if at least one of its dimensions is below `min_size`.
- A ground truth is considered blur if the variance of the Laplacian of the object crop is below `var_threshold`. This is adapted from [PyImageSearch - Blur Detection with OpenCV](https://www.pyimagesearch.com/2015/09/07/blur-detection-with-opencv/).
#### **Other**
The associated ground truth is not occluded, crowded, or truncated.

### Thresholds
The default values of the thresholds above are as follows:
```python
iou_threshold = 0.4
min_size = 32
var_threshold = 100
```

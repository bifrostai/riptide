# Understanding Error Types
> Read more: [Alchemy Handbook - Analyzing Failures](https://bifrost-core.gitlab.io/alchemy-handbook/object-detection/analyzing-failures/)

There are three threshold values:
- Background IoU threshold `bg_iou_threshold`: Detections smaller than this level are not considered
- Foreground IoU threshold `fg_iou_threshold`: Detections must be >= this level to be considered **correct**
- Confidence threshold `conf_threshold`: Detections must be >= this confidence to be considered


## BackgroundError
- Is above `conf_threshold` and does not meet the `bg_iou_threshold` with any ground truth
- Counts as a false positive to the predicted class

## ClassificationError
- Is above `conf_threshold` and `fg_iou_threshold`, but the class label is incorrect
- Counts as a false positive to the predicted class
- Counts as a false negative to the ground truth class (missed it)

## LocalizationError
- Is above `conf_threshold` and is between `bg_iou_threshold` and `fg_iou_threshold`, and the class label is **correct**
- Counts as a false positive to the predicted class
- Counts as a false negative to the ground truth class (missed it)

## ClassificationAndLocalizationError
- Is above `conf_threshold` and is between `bg_iou_threshold` and `fg_iou_threshold`, and the class label is **incorrect**
- Counts as a false positive to the predicted class
- Counts as a false negative to the ground truth class (missed it)

## DuplicateError
- Is above `conf_threshold` and `fg_iou_threshold`, but a simultaneous valid prediction has been made (true positive), which has a higher IoU than this one
- Counts as a false positive to the predicted class

## MissedError
- No prediction was made above `conf_threshold` that had IoU above `bg_iou_threshold` (otherwise it would be considered a `LocalizationError`)
- Counts as a false negative to the ground truth class

### Subgroups
To categorize how the missed errors are manifested, we can further subdivide the missed errors into the following subgroups:
- **Crowded**: The bounding box for the associated ground truth overlaps with another object.
    - This is determined by calculating the IoU of the ground truth with all other ground truths in the image, and if the maximum IoU is above `iou_threshold`, then the ground truth is considered crowded.
- **Occluded**: The associated ground truth is occluded by the environment.
- **Truncated**: The bounding box for the associated ground truth is truncated by the image boundary.
    - That is, at least one of the four corners of the bounding box within `min_size // 2` pixels of the image boundary.
- **Not Enough Visual Features**: The associated ground truth is either too small or too blur to be reasonably detected.
    - A ground truth is too small if at least one of its dimensions is below `min_size`.
    - A ground truth is considered blur if the variance of the Laplacian of the object crop is below `var_threshold`. This is adapted from [PyImageSearch - Blur Detection with OpenCV](https://www.pyimagesearch.com/2015/09/07/blur-detection-with-opencv/).
- **Other**: The associated ground truth is not occluded, crowded, or truncated.

The default values of the thresholds above are as follows:
```python
iou_threshold = 0.4
min_size = 32
var_threshold = 100
```

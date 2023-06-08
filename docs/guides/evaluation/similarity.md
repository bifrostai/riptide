Perceptual similarity is determined by computing cluster labels for the feature embeddings of ground truths and background errors, using the [HDBSCAN algorithm](https://github.com/scikit-learn-contrib/hdbscan). The cluster labels are then used to group the errors into perceptually similar groups.

## Understanding Grouped Errors

When errors are grouped, the first processed error is used as a representative for the group. Each group only contains errors of the same type.

A badge is displayed on the representative error to indicate the number of <u>u</u>nique targets and <u>t</u>otal number of errors within the group.

<div class="example infobox class-image-container row">
    <div class="class-image-wrapper">
        <img class="class-image" src="https://picsum.photos/id/906/200/200?grayscale&blur=2" width="150" height="150">
        <span class="data-tooltip">(1)</span>
    </div>
    <div class="class-image-wrapper">
        <img class="class-image" src="https://picsum.photos/id/906/200/200?grayscale&blur=2"  width="150" height="150">
        <span class="badge">T</span>
        <span class="data-tooltip">(2)</span>
    </div>
    <div class="class-image-wrapper">
        <img class="class-image" src="https://picsum.photos/id/906/200/200?grayscale&blur=2"  width="150" height="150">
        <span class="badge">U | T</span>
        <span class="data-tooltip">(3)</span>
    </div>
</div>

1. Single detection
2. Multiple detections, single target
3. Multiple detections, multiple targets

### Duplicity Ratio
For each error group, we can define the metric

$$\text{duplicity} = \frac{t}{u+1}, $$

where $t$ is the number of targets and $u$ is the number of unique targets. The $+1$ is introduced to distinguish groups with $t=1$ and $t=u.$

A high duplicity ratio indicates that for a group of similar targets, a model is making multiple erroneous predictions for the same target. This could be caused by the following:

- **Crowded Scenes:** The model is unable to identify the target in a crowded scene. This can manifest as [Confusions](../error_types.md#error-confusions "CLS + CLL") or [Localization Errors](../error_types.md#error-loc "LOC").


!!! tip "Model Improvement"
    Suppressing the predictions in a group with high duplicity ratio will improve the model's precision.

!!! note
    As of v0.4.0, this metric is only used to identify the most repeated localization errors.

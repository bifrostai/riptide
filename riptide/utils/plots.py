from typing import Any, Dict, List, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
from matplotlib.colorbar import Colorbar
from matplotlib.image import AxesImage
from matplotlib.ticker import StrMethodFormatter
from plotly.offline import plot

# region: Helper functions for plotting heatmaps
# Based on https://matplotlib.org/stable/gallery/images_contours_and_fields/image_annotated_heatmap.html


def heatmap(
    data: Union[np.ndarray, Dict[Tuple, Any]],
    row_labels: List[str] = None,
    col_labels: List[str] = None,
    ax: plt.Axes = None,
    axis_labels: Tuple[str, str] = ("Actual", "Predicted"),
    cbar_kw: dict = None,
    cbarlabel: str = "",
    grid_color: str = "w",
    **kwargs,
) -> Tuple[AxesImage, Colorbar]:
    """Create a heatmap from a numpy array and two lists of labels.

    Parameters
    ----------
    data: np.ndarray
        A 2D numpy array of shape (M, N).

    row_labels: list
        A list or array of length M with the labels for the rows.

    col_labels: list
        A list or array of length N with the labels for the columns.

    ax: matplotlib.axes.Axes, optional
        A `matplotlib.axes.Axes` instance to which the heatmap is plotted.  If
        not provided, use current axes or create a new one.

    cbar_kw: dict, optional
        A dictionary with arguments to `matplotlib.Figure.colorbar`.

    cbarlabel: str, optional
        The label for the colorbar.

    **kwargs
        All other arguments are forwarded to `imshow`.
    """
    if isinstance(data, dict):
        row_labels, col_labels = map(set, zip(*data.keys()))
        data: np.ndarray = np.array(
            [[data.get((i, j), 0) for j in col_labels] for i in row_labels]
        )
    else:
        assert isinstance(
            data, np.ndarray
        ), f"Expected data to be a dict or np.ndarray, got {type(data)}"
        assert len(data.shape) == 2, f"Expected data to be a 2D array, got {data.shape}"
        assert row_labels is not None, "Expected row_labels to be provided"
        assert col_labels is not None, "Expected col_labels to be provided"

    if ax is None:
        ax = plt.gca()

    if cbar_kw is None:
        cbar_kw = {}

    # Plot the heatmap
    im = ax.imshow(data, **kwargs)

    # Create colorbar
    cbar = ax.figure.colorbar(im, ax=ax, **cbar_kw)
    cbar.ax.set_ylabel(cbarlabel, rotation=-90, va="bottom")

    # Show all ticks and label them with the respective list entries.
    ax.set_xticks(np.arange(data.shape[1]), labels=col_labels)
    ax.set_yticks(np.arange(data.shape[0]), labels=row_labels)

    # Let the horizontal axes labeling appear on top.
    ax.tick_params(top=True, bottom=False, labeltop=True, labelbottom=False)

    # Add axis labels
    ax.set_xlabel(axis_labels[0])
    ax.set_ylabel(axis_labels[1])
    ax.xaxis.set_label_position("top")

    # # Rotate the tick labels and set their alignment.
    # plt.setp(ax.get_xticklabels(), rotation=-30, ha="right", rotation_mode="anchor")

    # Turn spines off and create white grid.
    ax.spines[:].set_visible(False)

    ax.set_xticks(np.arange(data.shape[1] + 1) - 0.5, minor=True)
    ax.set_yticks(np.arange(data.shape[0] + 1) - 0.5, minor=True)
    ax.grid(which="minor", color=grid_color, linestyle="-", linewidth=3)
    ax.tick_params(which="minor", bottom=False, left=False)

    return im, cbar


def annotate_heatmap(
    im: AxesImage,
    data=None,
    valfmt="{x:.2f}",
    textcolors=("black", "white"),
    threshold=None,
    **textkw,
):
    """A function to annotate a heatmap.

    Parameters
    ----------
    im
        The AxesImage to be labeled.
    data
        Data used to annotate.  If None, the image's data is used.  Optional.
    valfmt
        The format of the annotations inside the heatmap.  This should either
        use the string format method, e.g. "$ {x:.2f}", or be a
        `matplotlib.ticker.Formatter`.  Optional.
    textcolors
        A pair of colors.  The first is used for values below a threshold,
        the second for those above.  Optional.
    threshold
        Value in data units according to which the colors from textcolors are
        applied.  If None (the default) uses the middle of the colormap as
        separation.  Optional.
    **kwargs
        All other arguments are forwarded to each call to `text` used to create
        the text labels.
    """

    if not isinstance(data, (list, np.ndarray)):
        data = im.get_array()

    # Normalize the threshold to the images color range.
    if threshold is not None:
        threshold = im.norm(threshold)
    else:
        threshold = im.norm(data.max()) / 2.0

    # Set default alignment to center, but allow it to be
    # overwritten by textkw.
    kw = dict(horizontalalignment="center", verticalalignment="center")
    kw.update(textkw)

    # Get the formatter in case a string is supplied
    if isinstance(valfmt, str):
        valfmt = StrMethodFormatter(valfmt)

    # Loop over the data and create a `Text` for each "pixel".
    # Change the text's color depending on the data.
    texts = []
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            kw.update(color=textcolors[int(im.norm(data[i, j]) > threshold)])
            text = im.axes.text(j, i, valfmt(data[i, j], None), **kw)
            texts.append(text)

    return texts


# endregion


def plotly_markup(content: go.Figure) -> str:
    layout = dict(
        template="simple_white",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=20, r=20, t=30, b=10),
    )
    content.update_layout(layout)
    return plot(
        content,
        include_plotlyjs=False,
        output_type="div",
        config=dict(
            modeBarButtonsToRemove=["toggleHover"], responsive=True, displaylogo=False
        ),
    )

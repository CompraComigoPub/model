"""
Plot Utilities for the models .
"""
from io import BytesIO
from typing import List, Optional, Tuple, Union
from matplotlib.pyplot import (
    figure,
    grid,
    legend,
    plot,
    show,
    xlabel,
    ylabel,
    savefig,
    scatter,
)
from numpy import ndarray
from PIL import Image, PngImagePlugin


def plot_series(
    series: Union[Tuple[ndarray, ndarray], List[Tuple[ndarray, ndarray]]],
    predictions: Optional[
        Union[Tuple[ndarray, ndarray], List[Tuple[ndarray, ndarray]]]
    ],
    pformat: str = "-",
    series_labels: Optional[Tuple[str, str]] = None,
    predictions_labels: Optional[Tuple[str, str]] = None,
    plt_labels: Optional[List[str]] = None,
    figsize: Tuple[int] = (12, 8),
    fontsize: int = 12,
    do_show: bool = True,
) -> PngImagePlugin.PngImageFile:
    """
    Visualizes time series data

    Parameters
    ----------
    series : Union[Tuple[ndarray, ndarray], List[Tuple[ndarray, ndarray]]]
        Time series values for each time step .
    predictions: Optional[Union[Tuple[ndarray, ndarray], List[Tuple[ndarray, ndarray]]]]
        Predictions series values .
    pformat : str
        Plot format .
    plt_labels : Optional[Tuple[str, str]]
        Labels for the plot. Defaults to None .
    series_labels: Optional[Tuple[str, str]]
        Labels for the lines. Defaults to None .
    predictions_labels: Optional[Tuple[str, str]]
        Labels for the predictions. Defaults to None .
    figsize : Tuple[int]
        Size of the figure. Defaults to (12, 8) .
    fontsize : int
        Size of the figure font. Defaults to 12 .
    do_show: bool
        Whether to show or not .

    Returns
    -------
    PngImagePlugin.PngImageFile
        PIL image .

    """
    # Creates plot with time series data
    figure(figsize=figsize)
    if type(series) is list:
        for i, series_num in enumerate(series):
            plot(series_num[0], series_num[1], pformat)
    else:
        plot(series[0], series[1], pformat)
    if series_labels and (len(series_labels) > 0):
        legend(fontsize=fontsize, labels=series_labels)

    # Creates plot with predictions
    if type(predictions) is list:
        for i, series_num in enumerate(predictions):
            scatter(series_num[0], series_num[1], pformat)
    else:
        scatter(predictions[0], predictions[1], pformat)
    if predictions_labels and (len(predictions_labels) > 0):
        legend(fontsize=fontsize, labels=predictions_labels)

    # Labels for the axis
    if plt_labels:
        xlabel(plt_labels[0])
        ylabel(plt_labels[1])
    else:
        xlabel("Time")
        ylabel("Value")

    grid(True)

    buf = BytesIO()
    savefig(buf)
    buf.seek(0)
    im = Image.open(buf)

    # Shows plot
    if do_show:
        show()
    # Returns image
    return im

"""
IO utils to handle data from/to cloud and local filesystems .
"""

from typing import Callable, Union, List, Optional, Tuple
from s3fs import S3FileSystem
from numpy import ndarray
from matplotlib.pyplot import (
    figure,
    grid,
    gcf,
    legend,
    plot,
    xlabel,
    ylabel,
    scatter,
    close,
    cla,
    clf,
)
from PIL import Image, PngImagePlugin
from io import BytesIO
from tensorflow.data import Dataset
from tensorflow_datasets import as_numpy


def dynamic_open(path: str, mode: str) -> Callable:
    """
    Reads data from path .

    Parameters
    ----------
    path : str
        Path to data .
    mode : str
        Variable mode for the opener
        (for example, r for read, w for write ...) .

    Returns
    -------
    Callable
        Correct "open function" for the path .

    """
    filesystem, stripped_path = get_filesystem(path)
    if filesystem == "aws":
        return S3FileSystem().open(stripped_path, mode)
    else:
        return open(stripped_path, mode)


def get_filesystem(path: str) -> str:
    """
    Get filesystem from path .

    Parameters
    ----------
    path : str
        Path to the file .

    Returns
    -------
    str
        Filesystem from the path without prefix  .

    """
    if path.startswith("s3://"):
        filesystem = "aws"
        path = path[5:]
    else:
        filesystem = "local"
    return filesystem, path


def get_file_format(path: str) -> str:
    """
    Get file format from path .

    Parameters
    ----------
    path : str
        Path to the file .

    Returns
    -------
    str
        File format from the suffix of the path .

    """
    if path.endswith("csv"):
        return "csv"
    elif path.endswith("parquet"):
        return "parquet"


def plot_series(
    time: ndarray,
    series: Union[ndarray, List[ndarray]],
    pformat: str = "-",
    start: Optional[int] = 0,
    end: Optional[int] = -1,
    labels: Optional[List[str]] = None,
    figsize: Tuple[int] = (12, 8),
    fontsize: int = 12,
) -> None:
    """
    Visualizes time series data .

    Parameters
    ----------
    time : ndarray
        Time series timesteps .
    series : Union[ndarray, List[ndarray]]
        Time series values for each time step .
    pformat : str
        Plot format .
    start : int
        Index of first timestep to plot. Defaults to 0 .
    end : Optional[int]
        Index of last timestep to plot. Defaults to -1 .
    labels: Optional[List[str]]
        List of tags for the line. Defaults to None .
    figsize : Tuple[int]
        Size of the figure. Defaults to (12, 8) .
    fontsize : int
        Size of the figure font. Defaults to 12 .

    Returns
    -------
    None

    """
    # Creates plot with time series data
    figure(figsize=figsize)
    if type(series) is tuple:
        for series_num in series:
            plot(time[start:end], series_num[start:end], pformat)
    else:
        plot(time[start:end], series[start:end], pformat)

    # Labels for the axis
    xlabel("Time")
    ylabel("Value")
    if labels:
        legend(fontsize=fontsize, labels=labels)

    grid(True)
    legend(loc="upper right")

    # Image
    buf = BytesIO()
    gcf().savefig(buf)
    buf.seek(0)
    im = Image.open(buf)

    close()

    return im


def plot_all(
    series_lines: List[Tuple[ndarray, ndarray]],
    series_points: Optional[List[Tuple[ndarray, ndarray]]] = None,
    labels_lines: Optional[List[str]] = None,
    labels_points: Optional[List[str]] = None,
    xy_label: Optional[Tuple[str]] = None,
    figsize: Tuple[int] = (12, 8),
    fontsize: int = 12,
) -> PngImagePlugin.PngImageFile:
    """
    Visualizes time series data .

    Parameters
    ----------
    series_lines: List[Tuple[ndarray, ndarray]]
        Time series line values .
    series_points: Optional[List[Tuple[ndarray, ndarray]]]
        Time series point values .
    labels_lines: Optional[List[str]]
        List of tags for the line. Defaults to None .
    labels_points: Optional[List[str]]
        List of tags for the points. Defaults to None .
    xy_label: Optional[Tuple[str]]
        Labels for xy axis .
    figsize : Tuple[int]
        Size of the figure. Defaults to (12, 8) .
    fontsize : int
        Size of the figure font. Defaults to 12 .

    Returns
    -------
    PngImagePlugin.PngImageFile

    """
    # Creates plot with time series data
    fig = figure(figsize=figsize)

    # Plots lines
    for i, series in enumerate(series_lines):
        plot(series[0], series[1], label=labels_lines[i])
    # Plots points
    if series_points:
        for i, series in enumerate(series_points):
            scatter(series[0], series[1], label=labels_points[i])

    # Labels for the axis
    if not xy_label:
        xlabel("Time")
        ylabel("Value")
    else:
        xlabel(xy_label[0])
        ylabel(xy_label[1])

    grid(True)
    legend(loc="upper right")

    # Image
    buf = BytesIO()
    gcf().savefig(buf)
    buf.seek(0)
    im = Image.open(buf)

    cla()
    clf()
    close("all")
    close(fig)

    return im


def convert_tensorflow_dataset_to_numpy(dataset: Dataset) -> List[ndarray]:
    """
    Performs conversion from tensorflow dataset to numpy array(s) .

    Parameters
    ----------
    dataset: Dataset
        Time series in a tensorflow dataset.

    Returns
    -------
    List[ndarray]
        Time series as numpy array(s) .

    """
    datasets = []
    for val in as_numpy(dataset):
        for x in val:
            datasets.append([])
        break
    for i, val in enumerate(as_numpy(dataset)):
        for j, x in enumerate(val):
            datasets[j].append(x)
    return datasets

"""
Implementation of DataHandler class .
"""

from typing import Any, Dict, Optional, Union, List, Tuple

from pandas import DataFrame, read_csv, read_parquet
from yaml import dump, safe_load
from tensorflow.data import Dataset

from model_compra_comigo.data_handler.utils import (
    dynamic_open,
    get_file_format,
    plot_series,
    plot_all,
    convert_tensorflow_dataset_to_numpy,
)
from PIL import PngImagePlugin

from numpy import ndarray


class DataHandler:
    """
    Handler for IO related functionalities. It should allow the
    user to load and save data locally, or in the cloud .

    Attributes
    ----------
    supported_file_formats : List[str]
        Supported filetypes to save/read to/from .

    Methods
    -------
    read_dataset(path: str) -> DataFrame
        Reads dataset from path .
    save_dataset(
        dataset: Union[DataFrame, ndarray],
        path: str,
        file_format: str = "csv",
        columns: Optional[List[str]] = [],
    ) -> None
        Saves dataset .
    read_yaml(path: str) -> Dict[str, Any]
        Reads yaml from path .
    save_yaml(
        data: Dict[str, Any],
        path: str
    ) -> None
        Saves yaml to path .
    convert_tensorflow_dataset_to_numpy(
        dataset: Dataset
    ) -> List[ndarray]
        Performs conversion from tensorflow dataset to numpy array(s) .
    generate_windowed_dataset(
        data: ndarray,
        window_size: int,
        batch_size: int,
        shuffle: bool,
        shuffle_buffer_size: int,
        nforecasts: int = 1
    ) -> Dataset
        Generates a windowed dataset from timeseries .

    """

    # Supported filetypes to save/read to/from .
    supported_file_formats = ["csv", "parquet"]
    supported_data_types = ["pandas", "numpy"]

    @staticmethod
    def read_dataset(
        path: str,
        data_type: str = "pandas",
        target_column: Optional[Union[str, List[str]]] = "target",
    ) -> Union[DataFrame, ndarray]:
        """
        Reads dataset from path .

        Parameters
        ----------
        path : str
            Path to dataset .
        data_type : str
            Type of the data to be read. Currently supports pandas and
            numpy .
        target_column : str
            Column name for the target of the data to be read. Defaults to target.
            It can separate more than one target for multiple objective tasks,
            if a list of variables/strings is specified instead of a single one.
            This variable is used only when the data type is numpy .

        Raises
        ------
        ValueError
            If format passed is invalid .
        NotImplementedError
            If format passed was not implemented. This should not
            occur, because it would imply a format not implemented
            is in the supported_file_formats list .

        Returns
        -------
        Union[DataFrame, ndarray, Tuple[ndarray]]
            Dataset read from path .

        """
        file_format = get_file_format(path)
        if data_type not in DataHandler.supported_data_types:
            raise ValueError(
                f"Type {data_type} not in list of supported data types: {DataHandler.supported_data_types}"
            )
        supported_file_formats = DataHandler.supported_file_formats
        if file_format not in supported_file_formats:
            raise ValueError(
                f"File format not supported. Given: {file_format}. ",
                f"Expected one of: {supported_file_formats}",
            )
        elif file_format == "csv":
            with dynamic_open(path=path, mode="rb") as fh:
                data = read_csv(fh)
                data.drop(data.filter(regex="Unnamed"), axis=1, inplace=True)
        elif file_format == "parquet":
            data = read_parquet(path)
        else:
            raise NotImplementedError(
                f"Type {file_format} is not currently supported, ",
                "but is on the way!",
            )
        if data_type == "numpy":
            X, y = (
                data.loc[:, data.columns != "target"].to_numpy(),
                data["target"].to_numpy(),
            )
            return X, y
        return data

    @staticmethod
    def save_dataset(
        dataset: Union[DataFrame, ndarray],
        path: str,
        file_format: str = "csv",
        columns: Optional[List[str]] = [],
    ) -> None:
        """
        Saves dataset .

        Parameters
        ----------
        dataset : Union[DataFrame, ndarray]
            Dataset to be saved .
        path: str
            Path to save the dataset .
        file_format: str
            File format of the dataset to be saved. Defaults to csv .
        columns: Optional[List[str]] = []
            Columns of the values to be saved. It is required if dataset
            is a numpy array .

        Raises
        ------
        ValueError
            If format passed is invalid .
        NotImplementedError
            If format passed was not implemented. This should not occur,
            because it would imply a format not implemented is in the
            supported_file_formats list .

        Returns
        -------
        None

        """
        supported_file_formats = DataHandler.supported_file_formats
        file_format = get_file_format(path)
        if file_format not in supported_file_formats:
            raise ValueError(
                "File format not supported. ",
                f"Given: {file_format}. Expected one of: {supported_file_formats}",
            )
        elif isinstance(dataset, DataFrame):
            if file_format == "csv":
                with dynamic_open(path=path, mode="wb") as fh:
                    dataset.to_csv(fh)
            elif file_format == "parquet":
                with dynamic_open(path=path, mode="wb") as fh:
                    dataset.to_parquet(fh)
            else:
                raise NotImplementedError(
                    f"Type {file_format} is not currently supported, but is on the way!"
                )
        elif isinstance(dataset, ndarray):
            if dataset.shape[1] != len(columns):
                raise ValueError(
                    "You must specify column names. Please include 'target' as well ."
                )
            dataset = DataFrame(dataset, columns=columns)
            if file_format == "csv":
                with dynamic_open(path=path, mode="wb") as fh:
                    dataset.to_csv(fh)
            elif file_format == "parquet":
                with dynamic_open(path=path, mode="wb") as fh:
                    dataset.to_parquet(fh)
            else:
                raise NotImplementedError(
                    f"Type {file_format} is not currently supported, but is on the way!"
                )
        return

    @staticmethod
    def read_yaml(path: str) -> Dict[str, Any]:
        """
        Reads yaml from path .

        Parameters
        ----------
        path : str
            Path to yaml .

        Returns
        -------
        Dict[str, Any]
            Definition read from path .

        """
        with dynamic_open(path=path, mode="r") as fh:
            data = safe_load(fh)
        return data

    @staticmethod
    def save_yaml(data: Dict[str, Any], path: str) -> None:
        """
        Saves yaml to path .

        Parameters
        ----------
        data : Dict[str, Any]
            Dataset to be saved .
        path: str
            Destination to save the yaml .

        Returns
        -------
        None

        """
        with dynamic_open(path=path, mode="w") as fh:
            return dump(data, fh)

    @staticmethod
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
        return convert_tensorflow_dataset_to_numpy(
            dataset=dataset,
        )

    @staticmethod
    def generate_windowed_dataset(
        data: ndarray,
        window_size: int,
        batch_size: int,
        shuffle_buffer_size: int,
        shuffle: bool = False,
        nforecast: int = 1,
        prefetch: int = 1,
    ) -> Dataset:
        """
        Generates a windowed dataset from timeseries .

        Parameters
        ----------
        data: ndarray
            Time series .
        window_size: int
            Window size for the prediction .
        batch_size: int
            Number of batches .
        shuffle_buffer_size: int
            Buffer size for the shuffle .
        shuffle: bool
            Whether to shuffle or not .
        nforecast: int
            Number of steps to look into the future. Defaults to 1 .
        prefetch: int
            Number to prefetch . Defaults to 1 .

        Returns
        -------
        Dataset
            Time series dataset .

        """
        dataset = (
            Dataset.from_tensor_slices(data)
            .window(window_size + nforecast, shift=1, drop_remainder=True)
            .flat_map(lambda window: window.batch(window_size + nforecast))
            .map(lambda window: (window[:-nforecast], window[-nforecast:]))
        )
        if shuffle:
            dataset = (
                dataset.shuffle(shuffle_buffer_size)
                .batch(batch_size)
                .prefetch(prefetch)
            )
            return dataset
        elif batch_size >= 1:
            dataset = dataset.batch(batch_size).prefetch(prefetch)
        return dataset

    @staticmethod
    def generate_windowed_data_forecast(
        data: ndarray,
        window_size: int,
        batch_size: int,
        prefetch: int = 1,
    ) -> Dataset:
        """
        Generates a windowed dataset from timeseries for forecasting .

        Parameters
        ----------
        data: ndarray
            Time series .
        window_size: int
            Window size for the prediction .
        batch_size: int
            Number of batches .
        prefetch: int
            Number to prefetch . Defaults to 1 .

        Returns
        -------
        Dataset
            Time series dataset .

        """
        dataset = (
            Dataset.from_tensor_slices(data)
            .window(window_size, shift=1, drop_remainder=True)
            .flat_map(lambda window: window.batch(window_size))
            .batch(batch_size)
            .prefetch(prefetch)
        )
        return dataset

    @staticmethod
    def plot_series(
        time: ndarray,
        series: Union[ndarray, List[ndarray]],
        pformat: str = "-",
        start: Optional[int] = 0,
        end: Optional[int] = -1,
        labels: Optional[List[str]] = None,
        figsize: Tuple[int] = (12, 8),
        fontsize: int = 12,
    ) -> PngImagePlugin.PngImageFile:
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
        labels : Optional[str]
            List of tags for the line. Defaults to None .
        figsize : Tuple[int]
            Size of the figure. Defaults to (12, 8) .
        fontsize : int
            Size of the figure font. Defaults to 12 .

        Returns
        -------
        None

        """
        return plot_series(
            time=time,
            series=series,
            pformat=pformat,
            start=start,
            end=end,
            labels=labels,
            figsize=figsize,
            fontsize=fontsize,
        )

    @staticmethod
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
        legend_lines: Optional[List[str]]
            List of tags for the line. Defaults to None .
        legend_points: Optional[List[str]]
            List of tags for the points. Defaults to None .
        xy_label: Optional[Tuple[str]]
            Labels for xy axis .
        figsize : Tuple[int]
            Size of the figure. Defaults to (12, 8) .
        fontsize : int
            Size of the figure font. Defaults to 12 .

        Returns
        -------
        None

        """
        return plot_all(
            series_lines=series_lines,
            series_points=series_points,
            labels_lines=labels_lines,
            labels_points=labels_points,
            xy_label=xy_label,
            figsize=figsize,
            fontsize=fontsize,
        )

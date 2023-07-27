"""
Implementation of Visualizer class. It is intended to provide visual aid
to select the best model and evaluate results visually .
"""

from typing import List


from model_compra_comigo.data_handler.utils import (
    plot_all,
)
from numpy import (
    ndarray,
    minimum,
    maximum,
)


class Visualizer:
    """
    Provides visual aid to select the best model and evaluate results visually .

    Attributes
    ----------
    None

    Methods
    -------
    create_gif(
        time_data: List,
        series: ndarray,
        forecast: ndarray,
        batch_size: int,
        window_size: int,
        nforecast: int,
        gif_window: int = 70,
        method: str = "default"
    ) -> List
        Creates a gif for visual evaluation of the performance of the model .

    """

    @staticmethod
    def create_gif(
        time_data: List,
        series: ndarray,
        forecast: ndarray,
        batch_size: int,
        window_size: int,
        nforecast: int,
        gif_window: int = 70,
        method: str = "default",
    ) -> List:
        """
        Creates a gif for visual evaluation of the performance of the model .

        Parameters
        ----------
        time_data: List
            Time .
        series: ndarray
            Time series as is .
        forecast: ndarray
            Forecast .
        batch_size: int
            Batch size .
        window_size: int
            Window size .
        nforecast: int
            Number of points for the forecast .
        gif_window: int
            Window of focus. Defaults to 70 .
        method: str
            Method for predict .

        Returns
        -------
        List
            List of images .

        """
        plots = []
        n = forecast.shape[0]
        for j in range(0, n):
            tdata = list(
                time_data[
                    minimum(window_size + j, len(time_data)) : minimum(
                        window_size + nforecast + j, len(time_data)
                    )
                ]
            )
            if not tdata:
                tdata = [time_data[-1] + 1]
            if len(tdata) < nforecast:
                tdata = tdata + list(
                    range(tdata[-1] + 1, tdata[-1] + nforecast + 1 - len(tdata))
                )
            low, upw = maximum(0, j - gif_window), j + gif_window
            plot = plot_all(
                series_lines=[
                    (time_data[low:upw], series[low:upw]),
                    (time_data[j : window_size + j], series[j : window_size + j]),
                ],
                series_points=[(tdata, forecast[j])],
                labels_lines=["series", "window"],
                labels_points=["forecast"],
                xy_label=["Time", "Value"],
            )
            plots.append(plot)
        return plots

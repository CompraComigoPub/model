"""
Implementation of a PreviousOne Naive model .
"""
from numpy import ones, convolve, ndarray, maximum, concatenate, apply_along_axis, mean
from tensorflow.data import Dataset
from tensorflow.keras.metrics import mean_squared_error, mean_absolute_error


class PreviousOneModel:
    """
    Implementation of a Naive Model that returns the last value .

    Methods
    -------
    predict_batch(
        data: Dataset,
        window_size: int,
        nforecast: int = 1
    ) -> ndarray
        Previous One prediction in a dataset (batch) .
    predict(
        data: ndarray,
        window_size: int,
        nforecast: int = 1
    ) -> ndarray
        Previous One prediction in a dataset (batch) .

    """

    def __init__(
        self,
    ):
        """
        Constructor for PreviousOneModel .

        """
        return

    def predict_batch(
        self, data: Dataset, window_size: int, nforecast: int = 1
    ) -> ndarray:
        """
        Previous One prediction in a dataset (batch) .

        Parameters
        ----------
        data: Dataset
            Time series as a tensorflow dataset .
        window_size: int
            Moving average window size .
        nforecast: int
            Number of steps to forecast. Defaults to 1 .

        Returns
        -------
        ndarray
            Time series predictions .

        """
        predictions = []
        for example in data:
            predictions.append(
                apply_along_axis(
                    func1d=self.predict,
                    axis=-1,
                    arr=example.numpy(),
                    nforecast=nforecast,
                )
            )
        result = concatenate(predictions, axis=0)
        return result

    def predict(
        self,
        data: ndarray,
        window_size: int = 1,
        nforecast: int = 1,
    ) -> ndarray:
        """
        Previous One prediction in a dataset (batch) .

        Parameters
        ----------
        data: ndarray
            Data points .
        nforecast: int
            Number of steps to forecast. Defaults to 1 .

        Returns
        -------
        ndarray
            Time series predictions .

        """
        return data[-1] * ones(nforecast)

    def evaluate(self, data: Dataset, window_size: int, nforecast: int = 1) -> ndarray:
        """
        Previous One evaluation .

        Parameters
        ----------
        data: Dataset
            Time series as a tensorflow dataset .
        window_size: int
            Moving average window size .
        nforecast: int
            Number of steps to forecast. Defaults to 1 .

        Returns
        -------
        ndarray
            Time series predictions .

        """
        mse = []
        mae = []
        for example_window, correct_forecast in data:
            forecast = apply_along_axis(
                func1d=self.predict,
                axis=-1,
                arr=example_window.numpy(),
                nforecast=nforecast,
            )
            mse.append(mean_squared_error(forecast, correct_forecast).numpy())
            mae.append(mean_absolute_error(forecast, correct_forecast).numpy())
        mse_c = concatenate(mse, axis=0)
        mae_c = concatenate(mae, axis=0)
        return {"mse": mean(mse_c), "mae": mean(mae_c)}

"""
Implementation of a Moving Average Naive model .
"""
from tensorflow.data import Dataset
from numpy import ones, convolve, ndarray, maximum, concatenate, apply_along_axis


class MovingAverageModel:
    """
    Implementation of a Moving Average Naive Model .

    Methods
    -------
    predict(
        data: Dataset,
        window_size: int,
        nforecast: int = 1
    ) -> ndarray
        Moving average prediction .
    predict_batch(
        data: Dataset,
        window_size: int,
        nforecast: int = 1
    ) -> ndarray
        Moving average prediction in a dataset (batch) .

    """

    def __init__(
        self,
    ):
        """
        Constructor for MovingAverageModel .

        """
        return

    def predict(self, data: Dataset, window_size: int, nforecast: int = 1) -> ndarray:
        """
        Moving average prediction .

        Parameters
        ----------
        data: ndarray
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
        nforecast = maximum(1, int(nforecast))
        prediction = convolve(data, ones(window_size), "valid") / float(window_size)
        if nforecast == 1:
            return prediction
        else:
            return prediction * ones(nforecast)

    def predict_batch(
        self, data: Dataset, window_size: int, nforecast: int = 1
    ) -> ndarray:
        """
        Moving average prediction in a dataset (batch) .

        Parameters
        ----------
        data: ndarray
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
                    window_size=window_size,
                    nforecast=nforecast,
                )
            )
        result = concatenate(predictions, axis=0)
        return result

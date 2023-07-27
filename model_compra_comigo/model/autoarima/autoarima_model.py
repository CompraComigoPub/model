"""
Implementation of a AutoArima model .
"""
from numpy import ones, convolve, ndarray


class AutoArima:
    """
    Implementation of a AutoArima Model .

    """

    def __init__(
        self,
    ):
        """
        Constructor for AutoArima .

        """
        return

    def fit(
        self,
        series: ndarray,
        window_size: int,
    ) -> ndarray:
        """
        Moving average prediction .

        Parameters
        ----------
        series: ndarray
            Time series .
        window_size: int
            Moving average window size .

        Returns
        -------
        ndarray
            Time series predictions .

        """
        prediction = convolve(series, ones(window_size), "valid") / float(window_size)
        return prediction

    def predict(
        self,
        series: ndarray,
        window_size: int,
    ) -> ndarray:
        """
        Moving average prediction .

        Parameters
        ----------
        series: ndarray
            Time series .
        window_size: int
            Moving average window size .

        Returns
        -------
        ndarray
            Time series predictions .

        """
        prediction = convolve(series, ones(window_size), "valid") / float(window_size)
        return prediction

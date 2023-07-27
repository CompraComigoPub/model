"""
Data Simulator. It simulates the creation of synthetic time series data. Based on open ai course's on tensorflow
"""

from typing import Optional, Tuple, Dict, Any, Union

from numpy import arange, cos, exp, ndarray, pi, where
from numpy.random import RandomState, sample


class DataSimulator:
    """
    Simulates the creation of synthetic time series data .

    Attributes
    ----------
    seed : int
        Basic seed to be used by packages such as numpy .

    Methods
    -------
    generate(
        time_range: int,
        trend_slope: Optional[float],
        c: Optional[float],
        seasonality_period: int,
        seasonality_amplitude: float,
        seasonality_phase: float,
        seasonality_time_threshold: float,
        seasonality_ncos: float,
        seasonality_nexp: float,
        noise_scaling_factor: float,
        autocorrelation_amplitude: float,
        autocorrelation_phi: float,
        disable_trend: bool = False,
        disable_seasonality: bool = False,
        disable_noise: bool = False,
        disable_autocorrelation: bool = False,
    ) -> ndarray
        Generates adaptable timeseries .
    add_trend(data: ndarray, slope: float) -> ndarray
        Adds trend to data with slope .
    add_seasonality(
        time: ndarray,
        period: int,
        amplitude: float,
        phase: float,
        time_threshold: float,
        ncos: float,
        nexp: float,
    ) -> ndarray
        Adds seasonality to time series .
    seasonal_pattern(
        index_pattern: ndarray,
        time_threshold: float,
        ncos: float,
        nexp: float,
    ) -> ndarray
        Seasonal pattern .
    add_noise(time: ndarray, scaling_factor: float) -> ndarray
        Adds gaussian noise .
    add_autocorrelation(
        time: ndarray,
        amplitude: float,
        autocorrelation_phi: float,
    ) -> ndarray
        Generates autocorrelated data .
    plot_series(
        time: ndarray,
        series: ndarray,
        pformat: str = "-",
        start: Optional[int] = 0,
        end: Optional[int] = -1,
        label: Optional[List[str]] = None,
        figsize: Tuple[int] = (12, 8),
        fontsize: int = 12,
    ) -> None
        Visualizes time series data

    """

    seed = None

    def __init__(
        self,
        seed: int = 42,
    ):
        """
        Constructor for DataSimulator .

        Parameters
        ----------
        seed : int
            Basic seed to be used by packages such as numpy .

        """
        self.seed = seed
        return

    def generate(
        self,
        time_range: int,
        c: Optional[float] = 0.0,
        trend_slope: Optional[float] = None,
        seasonality_period: Optional[int] = None,
        seasonality_amplitude: Optional[float] = None,
        seasonality_phase: Optional[float] = None,
        seasonality_time_threshold: Optional[float] = None,
        seasonality_ncos: Optional[float] = None,
        seasonality_nexp: Optional[float] = None,
        noise_scaling_factor: Optional[float] = None,
        autocorrelation_amplitude: Optional[float] = None,
        autocorrelation_phi: Optional[float] = None,
        disable_trend: Optional[bool] = False,
        disable_seasonality: Optional[bool] = False,
        disable_noise: Optional[bool] = False,
        disable_autocorrelation: Optional[bool] = False,
        return_parameters: Optional[bool] = True,
    ) -> Union[ndarray, Tuple[ndarray, Dict[str, Any]]]:
        """
        Generates adaptable timeseries .

        Parameters
        ----------
        time_range: int
            Range of time to generate the time series .
        c: Optional[float]
            Constant to add to values .
        trend_slope: Optional[float]
            Slope to add .
        seasonality_period : Optional[int]
            Period for the pattern to repeat itselt .
        seasonality_amplitude : Optional[float]
            Amplitude for the pattern .
        seasonality_phase : Optional[float]
            Phase for the sinusoidal pattern .
        seasonality_time_threshold : Optional[float]
            Phase for the sinusoidal pattern .
        seasonality_ncos : Optional[float]
            n for cosine .
        seasonality_nexp : Optional[float]
            n for exp .
        noise_scaling_factor : Optional[float]
            Scaling factor for the signal .
        autocorrelation_amplitude : Optional[float]
            Autocorrelation amplitude for the signal .
        autocorrelation_phi : Optional[float]
            Scaling factor .
        disable_trend: Optional[bool]
            Whether to disable trend or not. Defaults to False.
        disable_seasonality: Optional[bool]
            Whether to disable seasonality or not. Defaults to False .
        disable_noise: Optional[bool]
            Whether to disable noise or not. Defaults to False.
        disable_autocorrelation: Optional[bool]
            Whether to disable autocorrelation or not. Defaults to False .
        return_parameters: Optional[bool]
            Whether to return parameters alongside timeseries. Defaults to True .

        Returns
        -------
        Union[ndarray, Tuple[ndarray, Dict[str, Any]]]
            Time series (or time series and parameters) .

        """

        time_range = time_range
        time = arange(time_range)
        data = time

        # Add trend .
        if not disable_trend:
            if not trend_slope:
                # Float between -10 and 10
                trend_slope = (sample() * 2.0 - 1.0) * 2.0
            data = self.add_trend(time=time, slope=float(trend_slope))

        # Add seasonality .
        if not disable_seasonality:
            if not seasonality_period:
                seasonality_period = int(time_range * sample())
            if not seasonality_amplitude:
                seasonality_amplitude = sample() * 100.0
            if not seasonality_phase:
                seasonality_phase = sample() * 100.0
            if not seasonality_time_threshold:
                seasonality_time_threshold = sample()
            if not seasonality_ncos:
                seasonality_ncos = (sample()) * 10.0
            if not seasonality_nexp:
                seasonality_nexp = (sample()) * 10.0
            data = data + self.add_seasonality(
                time=time,
                period=int(seasonality_period),
                amplitude=float(seasonality_amplitude),
                phase=float(seasonality_phase),
                time_threshold=float(seasonality_time_threshold),
                ncos=float(seasonality_ncos),
                nexp=float(seasonality_nexp),
            )

        # Add white noise .
        if not disable_noise:
            noise_scaling_factor = sample() * 10.0
            data = data + self.add_noise(
                time=time, scaling_factor=float(noise_scaling_factor)
            )

        # Add autocorrelation .
        if not disable_autocorrelation:
            autocorrelation_amplitude = sample() * 50.0
            autocorrelation_phi = sample()
            data = data + self.add_autocorrelation(
                time=time, amplitude=autocorrelation_amplitude, phi=autocorrelation_phi
            )

        # Adds constant (y0)
        data = data + c

        if return_parameters:
            parameters = {
                "trend_slope": trend_slope,
                "seasonality_period": seasonality_period,
                "seasonality_amplitude": seasonality_amplitude,
                "seasonality_phase": seasonality_phase,
                "seasonality_time_threshold": seasonality_time_threshold,
                "seasonality_ncos": seasonality_ncos,
                "seasonality_nexp": seasonality_nexp,
                "noise_scaling_factor": noise_scaling_factor,
                "autocorrelation_amplitude": autocorrelation_amplitude,
                "autocorrelation_phi": autocorrelation_phi,
            }
            return data, parameters
        else:
            return data

    def add_trend(self, time: ndarray, slope: float) -> ndarray:
        """
        Adds trend to data with slope .

        Parameters
        ----------
        time : ndarray
            Time series time.
        slope: float
            Slope to add .

        Returns
        -------
        ndarray
            Time series with slope .

        """
        result = time * slope
        return result

    def add_seasonality(
        self,
        time: ndarray,
        period: int,
        amplitude: float,
        phase: float,
        time_threshold: float,
        ncos: float,
        nexp: float,
    ) -> ndarray:
        """
        Adds seasonality to time series .

        Parameters
        ----------
        time : ndarray
            Time series time.
        period : int
            Period for the pattern to repeat itselt .
        amplitude : float
            Amplitude for the pattern .
        phase : int
            Phase for the sinusoidal pattern .
        time_threshold : float
            Phase for the sinusoidal pattern .
        ncos : float
            n for cosine .
        nexp : float
            n for exp .

        Returns
        -------
        ndarray
            Time series with seasonality .

        """

        def seasonal_pattern(
            index_pattern: ndarray,
            time_threshold: float,
            ncos: float,
            nexp: float,
        ) -> ndarray:
            """
            Seasonal pattern .

            Parameters
            ----------
            index_pattern : ndarray
                Index within pattern .
            time_threshold : float
                Phase for the sinusoidal pattern .
            ncos : float
                n for cosine .
            nexp : float
                n for exp .

            Returns
            -------
            ndarray
                Time series with seasonality .

            """
            # Generate the values using an arbitrary pattern
            data_pattern = where(
                index_pattern < time_threshold,
                cos(index_pattern * ncos * pi),
                1.0 / exp(nexp * index_pattern),
            )
            return data_pattern

        # Define the measured values per period
        index_pattern = ((time + phase) % period) / period

        # Generates the seasonal data scaled by the defined amplitude
        result = amplitude * seasonal_pattern(
            index_pattern=index_pattern,
            time_threshold=time_threshold,
            ncos=ncos,
            nexp=nexp,
        )
        return result

    def add_noise(self, time: ndarray, scaling_factor: float) -> ndarray:
        """
        Adds gaussian noise .

        Parameters
        ----------
        time : ndarray
            Time series time.
        scaling_factor : float
            Scaling factor for the signal .

        Returns
        -------
        ndarray
            Time series with noise .

        """
        # Generate gaussian white noise .
        rnd = RandomState(self.seed)
        noise = rnd.randn(len(time)) * scaling_factor
        return noise

    def add_autocorrelation(
        self,
        time: ndarray,
        amplitude: float,
        phi: float,
    ) -> ndarray:
        """
        Generates autocorrelated data .

        Parameters
        ----------
        time : ndarray
            Time series time.
        amplitude : float
            Autocorrelation amplitude for the signal .
        phi : float
            Scaling factor .

        Returns
        -------
        ndarray
            Time series with noise .

        """
        # Initialize array of random numbers equal to the length
        # of the given time steps plus an additional step
        rnd = RandomState(self.seed)
        ar = rnd.randn(len(time) + 1)

        # Autocorrelate element 11 onwards with the measurement at
        # (t-1), where t is the current time step
        for step in range(1, len(time) + 1):
            ar[step] += phi * ar[step - 1]

        # Get the autocorrelated data and scale with the given amplitude.
        ar = ar[1:] * amplitude

        return ar

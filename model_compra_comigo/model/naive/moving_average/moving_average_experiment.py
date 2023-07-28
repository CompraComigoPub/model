"""
Implementation of a MovingAverage Naive experiment .
"""
from numpy import ndarray, concatenate, apply_along_axis, mean, array
from tensorflow.data import Dataset
from tensorflow.keras.metrics import mean_squared_error, mean_absolute_error
from mlflow import log_metrics, log_params, start_run, log_artifact
from typing import Any, Tuple
from model_compra_comigo.data_handler import DataHandler, Visualizer
from model_compra_comigo.model.naive.moving_average.moving_average_model import (
    MovingAverageModel,
)
from typing import Optional
from pandas import DataFrame


class MovingAverageExperiment:
    """
    Implementation of a Experiment that used Naive Model Moving Average .

    Methods
    -------
    run(
        train_data: Tuple[ndarray, ndarray],
        test_data: Tuple[ndarray, ndarray],
        window_size: int,
        batch_size: int,
        epochs: int,
        nforecast: int,
        validation_data: Optional[Tuple[ndarray, ndarray]] = None,
        generate_full_visualization: bool = True,
        shuffle_buffer_size: int = 1000,
        run_name: Optional[str] = "Moving Average Model",
        experiment_id: Optional[str] = None,
    ) -> None
        MovingAverageExperiment run. Fits nothing, since MovingAverageModel does
        not have metadata and evaluates its performance .
    evaluate(
        model: Any,
        data: Dataset,
        window_size: int,
        nforecast: int = 1,
    ) -> ndarray
        Moving Average evaluation .
    create_gif(
        time_data: ndarray,
        series: ndarray,
        forecast: ndarray,
        window_size: int,
        nforecast: int,
        batch_size: int,
        gif_window: int = -1,
        destination: str = "./tmp/evaluation.gif",
    ) -> None
    forecast_and_visualize(
        model: MovingAverageModel,
        data: Tuple[ndarray, ndarray],
        window_size: int,
        nforecast: int,
        batch_size: int,
        destination: str,
    ) -> None

    """

    def __init__(
        self,
    ):
        """
        Constructor for MovingAverageExperiment .

        Parameters
        ----------
        None

        """
        return

    @staticmethod
    def run(
        test_data: Tuple[ndarray, ndarray],
        window_size: int,
        batch_size: int,
        epochs: int,
        nforecast: int,
        train_data: Tuple[ndarray, ndarray] = None,
        validation_data: Optional[Tuple[ndarray, ndarray]] = None,
        generate_full_visualization: bool = True,
        shuffle_buffer_size: int = 1000,
        run_name: Optional[str] = "Moving Average Model",
        experiment_id: Optional[str] = None,
    ) -> None:
        """
        MovingAverageExperiment run. Fits nothing, since MovingAverageModel does
        not have metadata and evaluates its performance .

        Parameters
        ----------
        test_data: Tuple[ndarray, ndarray]
            Test data time series .
        window_size: int
            Moving average window size .
        batch_size: int
            Batch size .
        epochs: int
            Number of steps to forecast. Defaults to 1 .
        nforecast: int
            Number of steps to look into the future .
        train_data: Tuple[ndarray, ndarray]
            Train data time series. It is ignored, since MovingAverageModel does not require training .
        validation_data: Optional[Tuple[ndarray, ndarray]]
            Validation data time series. Defaults to None .
        generate_full_visualization: bool
            Whether to generate full visualization or just partial as
            artifacts .
        shuffle_buffer_size: int
            Buffer size for the shuffle. Defaults to 1000 .
        run_name: Optional[str]
            Run name. Defaults to "Moving Average Model" .
        experiment_id: Optional[str]
            Id of the experiment. Defaults to None .

        Returns
        -------
        None

        """
        with start_run(run_name=run_name, experiment_id=experiment_id) as run:
            # Creates model
            model = MovingAverageModel()

            # Dataset Manipulation
            if train_data:
                # Train dataset
                windowed_train_data = DataHandler.generate_windowed_dataset(
                    data=train_data[1],
                    window_size=window_size,
                    batch_size=batch_size,
                    shuffle_buffer_size=shuffle_buffer_size,
                    nforecast=nforecast,
                    shuffle=True,
                )
            if validation_data:
                # Validation dataset
                windowed_validation_data = DataHandler.generate_windowed_dataset(
                    data=validation_data[1],
                    window_size=window_size,
                    batch_size=batch_size,
                    shuffle_buffer_size=shuffle_buffer_size,
                    nforecast=nforecast,
                    shuffle=True,
                )
            # Test dataset
            windowed_test_data = DataHandler.generate_windowed_dataset(
                data=test_data[1],
                window_size=window_size,
                batch_size=batch_size,
                shuffle_buffer_size=shuffle_buffer_size,
                nforecast=nforecast,
                shuffle=True,
            )

            # Evaluate
            train_evaluation = {"mse": -1, "mae": -1}
            if train_data:
                # Train dataset
                train_evaluation = MovingAverageExperiment.evaluate(
                    model=model,
                    data=windowed_train_data,
                    window_size=window_size,
                    nforecast=nforecast,
                )
            validation_evaluation = {"mse": -1, "mae": -1}
            if validation_data:
                # Validation dataset
                validation_evaluation = MovingAverageExperiment.evaluate(
                    model=model,
                    data=windowed_validation_data,
                    window_size=window_size,
                    nforecast=nforecast,
                )
            test_evaluation = MovingAverageExperiment.evaluate(
                model=model,
                data=windowed_test_data,
                window_size=window_size,
                nforecast=nforecast,
            )

            # Log params
            params = {
                "window_size": window_size,
                "batch_size": batch_size,
                "nforecast": nforecast,
            }
            log_params(params)

            # Log metrics
            metrics = {
                "train_mse": train_evaluation["mse"],
                "train_mae": train_evaluation["mae"],
                "validation_mse": validation_evaluation["mse"],
                "validation_mae": validation_evaluation["mae"],
                "test_mse": test_evaluation["mse"],
                "test_mae": test_evaluation["mae"],
            }
            log_metrics(metrics)

            # Generate and save prediction
            prediction_path = "./tmp/test_prediction.csv"
            windowed_test_data_f = DataHandler.generate_windowed_data_forecast(
                data = test_data[1],
                window_size = window_size,
                batch_size = batch_size,
            )
            prediction = model.predict_batch(
                data=windowed_test_data_f,
                window_size=window_size,
                nforecast=nforecast
            )
            DataHandler.save_dataset(
                dataset=DataFrame(prediction),
                path=prediction_path
            )
            log_artifact(prediction_path)

            try:
                # Last prediction
                lp_path = "./tmp/prediction.png"
                new_pts = (array(range(test_data[0][-1]+1, test_data[0][-1]+nforecast+1)))
                plot = DataHandler.plot_all(
                    series_lines=[
                        (test_data[0][-window_size:], test_data[1][-window_size:]),
                    ],
                    series_points=[(new_pts, prediction[-1])],
                    labels_lines=["series", "window"],
                    labels_points=["forecast"],
                    xy_label=["Time", "Value"],
                )
                plot.save(
                    lp_path,
                )
                log_artifact(lp_path)
            except:
                return

            if generate_full_visualization:
                # Train
                if train_data:
                    artifact_path = "./tmp/evaluation_train.gif"
                    MovingAverageExperiment.forecast_and_visualize(
                        model=model,
                        data=train_data,
                        window_size=window_size,
                        nforecast=nforecast,
                        batch_size=batch_size,
                        destination=artifact_path,
                    )
                    log_artifact(artifact_path)
                # Validation
                if validation_data:
                    artifact_path = "./tmp/evaluation_validation.gif"
                    MovingAverageExperiment.forecast_and_visualize(
                        model=model,
                        data=validation_data,
                        window_size=window_size,
                        nforecast=nforecast,
                        batch_size=batch_size,
                        destination=artifact_path,
                    )
                    log_artifact(artifact_path)
                # Test
                artifact_path = "./tmp/evaluation_test.gif"
                MovingAverageExperiment.forecast_and_visualize(
                    model=model,
                    data=test_data,
                    window_size=window_size,
                    nforecast=nforecast,
                    batch_size=batch_size,
                    destination=artifact_path,
                )
                log_artifact(artifact_path)
        return

    @staticmethod
    def evaluate(
        model: Any,
        data: Dataset,
        window_size: int,
        nforecast: int = 1,
    ) -> ndarray:
        """
        Moving Average evaluation .

        Parameters
        ----------
        model: Any
            Model to evaluate .
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
        mse_l = []
        mae_l = []
        for example_window, correct_forecast in data:
            forecast = apply_along_axis(
                func1d=model.predict,
                axis=-1,
                arr=example_window.numpy(),
                nforecast=nforecast,
                window_size=window_size,
            )
            # Mean Squared Error
            mse = mean_squared_error(forecast, correct_forecast).numpy()
            mse_l.append(mse)
            # Mean Absolute Error
            mae = mean_absolute_error(forecast, correct_forecast).numpy()
            mae_l.append(mae)
        mse_c = concatenate(mse_l, axis=0)
        mae_c = concatenate(mae_l, axis=0)
        # Metrics
        return {"mse": mean(mse_c), "mae": mean(mae_c)}

    @staticmethod
    def create_gif(
        time_data: ndarray,
        series: ndarray,
        forecast: ndarray,
        window_size: int,
        nforecast: int,
        batch_size: int,
        gif_window: int = -1,
        destination: str = "./tmp/evaluation.gif",
    ) -> None:
        if gif_window == -1:
            gif_window = 2 * (window_size + nforecast)

        plots = Visualizer.create_gif(
            time_data=time_data,
            series=series,
            forecast=forecast,
            batch_size=batch_size,
            window_size=window_size,
            nforecast=nforecast,
            gif_window=gif_window,
        )
        plots[0].save(
            destination,
            save_all=True,
            append_images=plots[1:],
            optimize=False,
            duration=100,
        )

    @staticmethod
    def forecast_and_visualize(
        model: MovingAverageModel,
        data: Tuple[ndarray, ndarray],
        window_size: int,
        nforecast: int,
        batch_size: int,
        destination: str,
    ) -> None:
        # Windowed Data
        windowed_data = DataHandler.generate_windowed_data_forecast(
            data=data[1],
            window_size=window_size,
            batch_size=batch_size,
        )
        # Windowed Data Forecast
        windowed_data_forecast = model.predict_batch(
            data=windowed_data, window_size=window_size, nforecast=nforecast
        )
        MovingAverageExperiment.create_gif(
            time_data=data[0],
            series=data[1],
            forecast=windowed_data_forecast,
            window_size=window_size,
            nforecast=nforecast,
            batch_size=batch_size,
            destination=destination,
        )

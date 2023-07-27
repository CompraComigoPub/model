"""
Implementation of a RNN Example Experiment .
"""
from numpy import ndarray, concatenate, mean
from tensorflow.data import Dataset
from typing import Any, Optional
from tensorflow.keras.metrics import mean_squared_error, mean_absolute_error
from mlflow import log_metrics, start_run, log_artifact
from mlflow.tensorflow import autolog
from model_compra_comigo.model.rnn.rnn_example_model import RNNExampleModel
from model_compra_comigo.data_handler import DataHandler, Visualizer


class RNNExampleExperiment:
    """
    Implementation of a RNN Example Experiment. It makes a run
    with rnn example .

    run(
        train_data: ndarray,
        test_data: ndarray,
        time_data: ndarray,
        window_size: int,
        batch_size: int,
        epochs: int,
        nforecast: int,
        validation_data: Optional[ndarray] = None,
        generate_full_visualization: bool = True,
        shuffle_buffer_size: int = 1000,
        run_name: Optional[str] = "RNN Example",
        experiment_id: Optional[str] = None,
    ) -> None
        RNNExampleExperiment run. Fits RNNExampleModel to data and
        evaluates its performance .
    evaluate(
        model: Any,
        data: Dataset,
        window_size: int,
        nforecast: int = 1,
    ) -> ndarray
        RNN Example evaluation .

    """

    def __init__(self):
        """
        Constructor for RNNExampleExperiment .

        Parameters
        ----------
        None

        """
        return

    @staticmethod
    def run(
        train_data: ndarray,
        test_data: ndarray,
        time_data: ndarray,
        window_size: int,
        batch_size: int,
        epochs: int,
        nforecast: int,
        validation_data: Optional[ndarray] = None,
        generate_full_visualization: bool = True,
        shuffle_buffer_size: int = 400,
        run_name: Optional[str] = "RNN Example",
        experiment_id: Optional[str] = None,
    ) -> None:
        """
        RNNExampleExperiment run. Fits RNNExampleModel to data and
        evaluates its performance .

        Parameters
        ----------
        train_data: ndarray
            Train data time series .
        test_data: ndarray
            Test data time series .
        time_data: ndarray
            Time data for time series .
        window_size: int
            Moving average window size .
        batch_size: int
            Batch size .
        epochs: int
            Number of steps to forecast. Defaults to 1 .
        nforecast: int
            Number of steps to look into the future .
        validation_data: Optional[ndarray]
            Validation data time series. Defaults to None .
        generate_full_visualization: bool
            Whether to generate full visualization or just partial as
            artifacts .        validation_data: Optional[ndarray]
            Validation data time series. Defaults to None .
        shuffle_buffer_size: int
            Buffer size for the shuffle. Defaults to 1000 .
        run_name: Optional[str]
            Run name. Defaults to "RNN Example Model" .
        experiment_id: Optional[str]
            Id of the experiment. Defaults to None .

        Returns
        -------
        None

        """
        autolog()  # Tensorflow autolog
        with start_run(run_name=run_name, experiment_id=experiment_id) as run:
            # Creates model
            model = RNNExampleModel(
                window_size=window_size,
                nforecast=nforecast,
            )

            # Dataset Manipulation
            # Train dataset
            windowed_train_data = DataHandler.generate_windowed_dataset(
                data=train_data,
                window_size=window_size,
                batch_size=batch_size,
                shuffle_buffer_size=shuffle_buffer_size,
                nforecast=nforecast,
                shuffle=True,
            )
            if validation_data:
                # Validation dataset
                windowed_validation_data = DataHandler.generate_windowed_dataset(
                    data=validation_data,
                    window_size=window_size,
                    batch_size=batch_size,
                    shuffle_buffer_size=shuffle_buffer_size,
                )
            # Test dataset
            windowed_test_data = DataHandler.generate_windowed_dataset(
                data=test_data,
                window_size=window_size,
                batch_size=batch_size,
                shuffle_buffer_size=shuffle_buffer_size,
                nforecast=nforecast,
                shuffle=True,
            )

            # Fits model
            history = model.fit(
                data=windowed_train_data,
                epochs=epochs,
                verbose=1,
            )

            # Test dataset
            windowed_test_data = DataHandler.generate_windowed_dataset(
                data=test_data,
                window_size=window_size,
                batch_size=batch_size,
                shuffle_buffer_size=shuffle_buffer_size,
            )
            # Evaluate
            evaluation_train = RNNExampleExperiment.evaluate(
                model=model,
                data=windowed_train_data,
                window_size=window_size,
                nforecast=nforecast,
            )
            evaluation_test = RNNExampleExperiment.evaluate(
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
            log_metrics(params)

            # Log metrics
            metrics = {
                "train_mse": evaluation_train["mse"],
                "train_mae": evaluation_train["mae"],
                "test_mse": evaluation_test["mse"],
                "test_mae": evaluation_test["mae"],
            }
            log_metrics(metrics)
            return model

            if generate_full_visualization:
                train_test = concatenate([train_data, test_data])
                windowed_train_test_data_forecast = (
                    DataHandler.generate_windowed_data_forecast(
                        data=train_test,
                        window_size=window_size,
                        batch_size=batch_size,
                    )
                )
                plots = Visualizer.create_gif(
                    data=windowed_train_test_data_forecast,
                    time_data=time_data,
                    series=train_test,
                    model=model,
                    batch_size=batch_size,
                    window_size=window_size,
                    nforecast=nforecast,
                    gif_window=2 * (window_size + nforecast),
                    method="nn",
                )
                plots[0].save(
                    "./tmp/evaluation_train_test.gif",
                    save_all=True,
                    append_images=plots[1:],
                    optimize=False,
                    duration=100,
                )
                log_artifact("./tmp/evaluation_train_test.gif")
                # tests_len = test_data.shape[0]
                # plots[-tests_len:].save('./tmp/evaluation_train_test.gif',
                #     save_all = True, append_images = plots[tests_len+1:],
                #     optimize = False, duration = 100
                # )

                plots[-1].save(
                    "./tmp/prediction.png", save_all=True, optimize=False, duration=100
                )
                log_artifact("./tmp/prediction.png")
        return

    @staticmethod
    def evaluate(
        model: Any,
        data: Dataset,
        window_size: int,
        nforecast: int = 1,
    ) -> ndarray:
        """
        RNN Example evaluation .

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
            forecast = model.predict(example_window)
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

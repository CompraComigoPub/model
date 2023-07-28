"""
Implementation of a RNN Example Experiment .
"""
from tensorflow.data import Dataset
from typing import Any, Optional, Tuple
from tensorflow.keras.metrics import mean_squared_error, mean_absolute_error
from model_compra_comigo.model.rnn.rnn_example_model import RNNExampleModel
from model_compra_comigo.data_handler import DataHandler, Visualizer
from numpy import ndarray, concatenate, mean, array
from mlflow import log_metrics, log_params, start_run, log_artifact
from pandas import DataFrame
from contextlib import redirect_stdout
from tensorflow import convert_to_tensor


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
        train_data: Tuple[ndarray, ndarray],
        test_data: Tuple[ndarray, ndarray],
        window_size: int,
        batch_size: int,
        epochs: int,
        nforecast: int,
        validation_data: Optional[Tuple[ndarray, ndarray]] = None,
        generate_full_visualization: bool = True,
        shuffle_buffer_size: int = 1000,
        run_name: Optional[str] = "RNN Example Model",
        experiment_id: Optional[str] = None,
    ) -> None:
        """
        RNNExampleExperiment run .

        Parameters
        ----------
        train_data: Tuple[ndarray, ndarray]
            Train data time series .
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
            model = RNNExampleModel(
                window_size=window_size,
                nforecast=nforecast,
            )
            # Dataset Manipulation
            # Train dataset
            windowed_train_data = DataHandler.generate_windowed_dataset(
                data=train_data[1],
                window_size=window_size,
                batch_size=-1,
                shuffle_buffer_size=shuffle_buffer_size,
                nforecast = nforecast,
                shuffle=True,
            )            
            if validation_data:
                # Validation dataset
                windowed_validation_data = DataHandler.generate_windowed_dataset(
                    data=validation_data[1],
                    window_size=window_size,
                    batch_size=-1,
                    shuffle_buffer_size=shuffle_buffer_size,
                    nforecast=nforecast,
                    shuffle=True,
                )
            # Test dataset
            windowed_test_data = DataHandler.generate_windowed_dataset(
                data=test_data[1],
                window_size=window_size,
                batch_size=-1,
                shuffle_buffer_size=shuffle_buffer_size,
                nforecast=nforecast,
                shuffle=True,
            )

            model.fit(
                data=windowed_train_data,
                epochs=epochs,
                batch_size=batch_size,
                patience=100,
                verbose=1,
            )

            # Saves model summary
            summary_path = "./tmp/modelsummary.txt"
            with open(summary_path, 'w') as f:
                with redirect_stdout(f):
                    model.get_summary()    
            log_artifact(summary_path)

            # Evaluate
            train_evaluation = {"mse": -1, "mae": -1}
            # forecast_train = DataHandler.model_forecast(
            #     model=model,
            #     data=train_data[1],
            #     window_size=window_size,
            #     batch_size=batch_size
            # )
            # results = forecast.squeeze()
            # Train dataset
            train_evaluation = RNNExampleExperiment.evaluate(
                model=model,
                data=windowed_train_data,
            )
            validation_evaluation = {"mse": -1, "mae": -1}
            if validation_data:
                # Validation dataset
                validation_evaluation = RNNExampleExperiment.evaluate(
                    model=model,
                    data=windowed_validation_data,
                )
            test_evaluation = RNNExampleExperiment.evaluate(
                model=model,
                data=windowed_test_data,
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

            # Saves model
            model_path = "./tmp/model"
            model.save_model(model_path)
            log_artifact(model_path)

            # # Generate and save prediction
            # prediction_path = "./tmp/test_prediction.csv"
            # windowed_test_data_f = DataHandler.generate_windowed_data_forecast(
            #     data = test_data[1],
            #     window_size = window_size,
            #     batch_size = -1,
            # )
            # prediction = model.predict(
            #     data=windowed_test_data_f,
            # ).squeeze()

            # DataHandler.save_dataset(
            #     dataset=DataFrame(prediction),
            #     path=prediction_path
            # )
            # log_artifact(prediction_path)

            # # Last prediction
            # lp_path = "./tmp/prediction.png"
            # new_pts = (array(range(test_data[0][-1]+1, test_data[0][-1]+nforecast+1)))
            # plot = DataHandler.plot_all(
            #     series_lines=[
            #         (test_data[0][-window_size:], test_data[1][-window_size:]),
            #     ],
            #     series_points=[(new_pts, prediction[-1])],
            #     labels_lines=["series", "window"],
            #     labels_points=["forecast"],
            #     xy_label=["Time", "Value"],
            # )
            # plot.save(
            #     lp_path,
            # )
            # log_artifact(lp_path)

            # if generate_full_visualization:
            #     # Train
            #     if train_data:
            #         artifact_path = "./tmp/evaluation_train.gif"
            #         RNNExampleExperiment.forecast_and_visualize(
            #             model=model,    
            #             data=train_data,
            #             window_size=window_size,
            #             nforecast=nforecast,
            #             batch_size=batch_size,
            #             destination=artifact_path,
            #         )
            #         log_artifact(artifact_path)
            #     # Validation
            #     if validation_data:
            #         artifact_path = "./tmp/evaluation_validation.gif"
            #         RNNExampleExperiment.forecast_and_visualize(
            #             model=model,
            #             data=validation_data,
            #             window_size=window_size,
            #             nforecast=nforecast,
            #             batch_size=batch_size,
            #             destination=artifact_path,
            #         )
            #         log_artifact(artifact_path)
            #     # Test
            #     artifact_path = "./tmp/evaluation_test.gif"
            #     RNNExampleExperiment.forecast_and_visualize(
            #         model=model,
            #         data=test_data,
            #         window_size=window_size,
            #         nforecast=nforecast,
            #         batch_size=batch_size,
            #         destination=artifact_path,
            #     )
            #     log_artifact(artifact_path)
        return

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
        model: RNNExampleModel,
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
        RNNExampleExperiment.create_gif(
            time_data=data[0],
            series=data[1],
            forecast=windowed_data_forecast,
            window_size=window_size,
            nforecast=nforecast,
            batch_size=batch_size,
            destination=destination,
        )

    @staticmethod
    def evaluate(
        model: Any,
        data: Dataset,
    ) -> ndarray:
        """
        RNN Example evaluation .

        Parameters
        ----------
        model: Any
            Model to evaluate .
        data: Dataset
            Time series as a tensorflow dataset .

        Returns
        -------
        ndarray
            Time series predictions .

        """

        inputs = []
        targets = []

        for input_seq, target_seq in data:
            inputs.append(input_seq.numpy())
            targets.append(target_seq.numpy())

        # Inputs reshaped
        inputs_rs = array(inputs)
        inputs_rs = inputs_rs.reshape(inputs_rs.shape[0], inputs_rs.shape[1], 1)
        input = convert_to_tensor(inputs_rs)
        forecast = model.predict(input)

        # Target reshaped
        targets_rs = array(targets)
        target = convert_to_tensor(targets_rs)
        
        mse = mean_squared_error(forecast, target)
        mae = mean_absolute_error(forecast, target)

        # Metrics
        return {"mse": mean(mse), "mae": mean(mae)}

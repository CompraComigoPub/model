"""
Implementation of Experimenter. It implements functions with 
the purpose of automating evaluation and choosing of different models .
"""

from numpy import ndarray
from tensorflow.data import Dataset
from tensorflow_datasets import as_numpy
from typing import List, Optional
from model_compra_comigo.model.rnn import RNNExampleExperiment
from model_compra_comigo.model.naive import (
    MovingAverageExperiment,
    PreviousOneExperiment,
)
from model_compra_comigo.experimenter.utils.dataset import generate_windowed_dataset
from datetime import datetime
from mlflow import create_experiment
from model_compra_comigo.data_handler import DataHandler


class Experimenter:
    """
    Experimenter experiments with different models to generate the
    best model .

    """

    available_experiments = {
        "rnn_example": RNNExampleExperiment,
        "moving_average": MovingAverageExperiment,
        "previous_one": PreviousOneExperiment,
        # "autokeras", AutokerasExperiment
    }

    def __init__(
        self,
    ):
        """
        Constructor for Experimenter .
        """
        return

    @staticmethod
    def run_experiment(
        data_path: str,
        specification_path: str,
        destination_path: str,
        experiment_name: Optional[str] = None,
    ):
        """
        Runs experiments .

        Parameters
        ----------
        data_path: str
            Path to the data file (csv or parquet) .
        specification_path: str
            Path to the specification of the experiment .
        destination_path: str
            Path to the destination of the experiment .
        experiment_name: Optional[str]
            Name of the experiment .

        """

        # Create Experiment .
        if not experiment_name:
            now = datetime.now()
            time = now.strftime("%H_%M_%S")
            experiment_name = f"experiment_{time}"
        experiment_id = create_experiment(
            name=str(experiment_name),
            artifact_location = destination_path,
        )

        # Read data .
        data = DataHandler.read_dataset(path=data_path)

        # Read experiment specification .
        specification = DataHandler.read_yaml(specification_path)
        print(specification)
        # Run experiments .
        # experiment = Experimenter.available_experiments[model_name]
        # run(
        #     train_data=(time_data, train_data),
        #     validation_data=(time_data, validation_data),
        #     =,
        #     window_size,
        #     batch_size,
        #     epochs,
        #     nforecast,
        #     generate_full_visualization: bool = True,
        #     shuffle_buffer_size: int = 1000,
        #     run_name: Optional[str] = "rnn_1",
        #     )

        # # experiment.run()
        # # return experiment
        # print(data)
        # print(specification)

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
        X, y = [], []
        for i, (x_sample, y_sample) in enumerate(as_numpy(dataset)):
            X.append(x_sample)
            y.append(y_sample)
        return X, y

    @staticmethod
    def generate_windowed_dataset(
        data: ndarray,
        window_size: int,
        batch_size: int,
        shuffle_buffer_size: int,
        nforecast: int = 1,
        shuffle: bool = True,
    ):
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
        shuffle_buffer: int
            Buffer size for the shuffle .
        nforecast: int
            Number of steps to look into the future. Defaults to 1 .
        shuffle: bool
            Whether to shuffle data or not .

        Returns
        -------
        ndarray
            Time series predictions .

        """
        return generate_windowed_dataset(
            data=data,
            window_size=window_size,
            batch_size=batch_size,
            shuffle_buffer_size=shuffle_buffer_size,
            nforecast=nforecast,
            shuffle=shuffle,
        )

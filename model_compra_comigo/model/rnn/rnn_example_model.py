"""
Implementation of a RNN Example Model .
"""
from numpy import ndarray
from keras.models import Sequential
from keras.layers import LSTM, Dense, Conv1D
from tensorflow.data import Dataset
from typing import Any
from keras.callbacks import History
from keras.losses import Huber
from tensorflow.keras.optimizers.schedules import ExponentialDecay
from tensorflow.keras.optimizers import SGD


class RNNExampleModel:
    """
    Implementation of a RNN Example Model .

    """

    def __init__(self, window_size: int, nforecast: int):
        """
        Constructor for RNNExampleModel .

        """
        self.model = Sequential(
            [
                Conv1D(
                    filters=64,
                    kernel_size=3,
                    strides=1,
                    activation="relu",
                    padding="causal",
                    input_shape=[window_size, 1],
                ),
                LSTM(64, return_sequences=True),
                LSTM(64),
                Dense(20, activation="relu"),
                Dense(nforecast),
            ]
        )
        return

    def fit(
        self,
        data: Dataset,
        epochs: int,
        verbose: int = 1,
    ) -> History:
        """
        RNNExampleModel fit .

        Parameters
        ----------
        data: Dataset
            Time series as a tensorflow dataset .
        epochs: int
            Number of epochs to run .

        Returns
        -------
        History
            Keras history .

        """
        lr_schedule = ExponentialDecay(
            1.0, decay_steps=500, decay_rate=0.99, staircase=True
        )
        optimizer = SGD(learning_rate=lr_schedule, momentum=0.9)
        self.model.compile(loss=Huber(), optimizer=optimizer, metrics=["mae"])

        history = self.model.fit(data, epochs=epochs, verbose=verbose)
        return history

    def predict(
        self,
        data: Dataset,
        *args: Any,
        **kwargs: Any,
    ) -> ndarray:
        """
        RNNExampleModel prediction .

        Parameters
        ----------
        data: Dataset
            Time series as a tensorflow dataset .

        Returns
        -------
        ndarray
            Time series predictions .

        """
        prediction = self.model.predict(data)
        return prediction

    def get_model(
        self,
    ) -> Sequential:
        """
        RNNExampleModel model .

        Returns
        -------
        Sequential
            Sequential Model .

        """
        return self.model

    def get_summary(
        self,
    ) -> Any:
        """
        RNNExampleModel model summary .

        Returns
        -------
        Any
            Model summary .

        """
        return self.model.summary()

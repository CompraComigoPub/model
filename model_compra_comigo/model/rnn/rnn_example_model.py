"""
Implementation of a RNN Example Model .
"""
from numpy import ndarray, array
from keras.models import Sequential
from keras.layers import LSTM, Dense
from tensorflow.data import Dataset
from typing import Any, List, Optional
from keras.callbacks import History
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.initializers import GlorotUniform
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow import convert_to_tensor


class RNNExampleModel:
    """
    Implementation of a RNN Example Model .

    """

    def __init__(
        self,
        window_size: int,
        nforecast: int,
        loss: Optional[str] = "mse",
        metrics: Optional[List[str]] = ["mae", "mse"],
        adam_learning_rate: Optional[float] = 0.001,
        adam_beta_1: Optional[float] = 0.9,
        adam_beta_2: Optional[float] = 0.999,
        adam_epsilon: Optional[float] = 1e-8,
        adam_clipvalue: Optional[float] = 1.0,
    ):
        """
        Constructor for RNNExampleModel .

        Parameters
        ----------
        window_size: int
            Window size to look back before giving a forecast .
        nforecast: int
            Number of steps to predict .
        loss: Optional[str]
            Loss to be used .
        metrics: Optional[List[str]]
            Metrics to track .
        adam_learning_rate: Optional[float]
            Adam optimizer lr .
        adam_beta_1: Optional[float]
            Adam beta 1 .
        adam_beta_2: Optional[float]
            Adam beta 2 .
        adam_epsilon: Optional[float]
            Adam epsilon .
        adam_clipvalue: Optional[float]
            Adam optimizer clipvalue (may help with exploding weights)
        """
        initializer = GlorotUniform()
        self.model = Sequential(
            [
                LSTM(32, return_sequences=True, input_shape=(window_size, 1), kernel_initializer=initializer),
                LSTM(32, kernel_initializer=initializer),
                Dense(nforecast),
            ]
        )
        # Initialize the optimizer
        optimizer = Adam(
            learning_rate=adam_learning_rate,
            beta_1=adam_beta_1,
            beta_2=adam_beta_2,
            epsilon=adam_epsilon,
            clipvalue=adam_clipvalue,
        )
        # Set the training parameters
        self.model.compile(loss=loss, optimizer=optimizer, metrics=metrics)
        return

    def fit(
        self,
        data: Dataset,
        epochs: int,
        batch_size: int,
        patience: int = 100,
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
        batch_size: int
            Batch size .
        patience: int
            Number of runs before stop .
        verbose: int
            Verbosity .

        Returns
        -------
        History
            Keras history .

        """
        es = EarlyStopping(monitor='loss', patience=patience)

        inputs = []
        targets = []

        for input_seq, target_seq in data:
            inputs.append(input_seq.numpy())
            targets.append(target_seq.numpy())

        # Inputs reshaped
        inputs_rs = array(inputs)
        inputs_rs = inputs_rs.reshape(inputs_rs.shape[0], inputs_rs.shape[1], 1)
        input = convert_to_tensor(inputs_rs)

        # Target reshaped
        targets_rs = array(targets)
        targets_rs = targets_rs.reshape(targets_rs.shape[0], targets_rs.shape[1], 1)
        target = convert_to_tensor(targets_rs)

        # Train the model
        history = self.model.fit(
            input,
            target,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[es],
            verbose=verbose,
        )
        return history

    def predict(
        self,
        data: Dataset,
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

    def save_model(
        self,
        path: str
    ):
        self.model.save(path)
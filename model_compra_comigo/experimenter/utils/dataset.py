"""
Dataset utilities .
"""
from numpy import ndarray
from tensorflow.data import Dataset


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
    dataset = (
        Dataset.from_tensor_slices(data)
        .window(window_size + nforecast, shift=1, drop_remainder=True)
        .flat_map(lambda window: window.batch(window_size + nforecast))
        .map(lambda window: (window[:-nforecast], window[:-nforecast]))
    )
    if shuffle:
        dataset = dataset.shuffle(shuffle_buffer_size)
    if batch_size == -1:
        return dataset
    else:
        return dataset.batch(batch_size)


def evaluate_nn(
    data_predicted: ndarray,
    data_correct: ndarray,
) -> float:
    """
    Evaluates a neural network .

    Parameters
    ----------
    data_predicted: ndarray
        Predicted series .
    data_correct: ndarray
        Corect series .

    Returns
    -------
    float
        Evaluation .

    """
    return ((data_predicted - data_correct) ** 2).mean()

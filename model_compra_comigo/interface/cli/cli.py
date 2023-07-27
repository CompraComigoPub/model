"""
Implementation of application cli commands. Contains cli instructions that wrap application commands .
"""

from typing import Optional

from pyfiglet import Figlet
from typer import Option, Typer

from model_compra_comigo.experimenter import Experimenter
from model_compra_comigo.interface.cli.callback import version
from model_compra_comigo.logger import logger

application = Typer(help="Experiments with models .")


@application.callback()
def main(
    version: Optional[bool] = Option(
        None,
        "--version",
        callback=version,
        is_eager=True,
        help=("Shows Model CompraComigo application version ."),
    ),
) -> None:
    """
    Initializes cli and its functionalities .

    Parameters
    ----------
    version : Optional[bool]
        Callback that return application version .

    Returns
    -------
    None

    """
    print(Figlet().renderText("Model CompraComigo"))
    return


@application.command("run-experiment")
def run_experiment(
    data_path: str = Option(
        ..., "--data-path", "-dp", help=("Path to the data file (csv or parquet) .")
    ),
    specification_path: str = Option(
        ...,
        "--specification-path",
        "-sp",
        help=("Path to the specification of the experiment ."),
    ),
    destination_path: str = Option(
        ...,
        "--destination-path",
        "-dp",
        help=("Path to the destination of the experiment ."),
    ),
    experiment_name: Optional[str] = Option(
        None, "--experiment-name", "-e", help=("Name of the experiment .")
    ),
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

    Returns
    -------
    None

    """
    logger.info(
        f"Running run_experiment with data '{data_path}', specification '{specification_path}' and destination '{destination_path}'."
    )
    Experimenter.run_experiment(
        data_path=data_path,
        specification_path=specification_path,
        destination_path=destination_path,
        experiment_name=experiment_name,
    )


if __name__ == "__main__":
    application()

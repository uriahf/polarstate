"""Public API for the :mod:`polarstate` package."""

from .aj import prepare_event_table
from .predict import predict_aj_estimates

__all__ = ["aalen_johansen", "prepare_event_table", "predict_aj_estimates"]


def main() -> None:
    print("Hello from polarstate!")

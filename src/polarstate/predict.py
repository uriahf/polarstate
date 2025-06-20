import polars as pl
from .aj import prepare_event_table


def predict_aj_estimates(times: pl.Series, reals: pl.Series) -> pl.DataFrame:
    """Predict state occupancy probabilities using the Aalen-Johansen estimator.

    Parameters
    ----------
    times : pl.Series
        Event or censoring times for each observation.
    reals : pl.Series
        Event type for each observation.

    Returns
    -------
    pl.DataFrame
        The event table with intermediate calculations.
    """
    times_and_reals = pl.DataFrame({"times": times, "reals": reals})
    return prepare_event_table(times_and_reals)

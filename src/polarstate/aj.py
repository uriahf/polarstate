import polars as pl

def aalen_johansen(
    times: pl.Series,
    reals: pl.Series,
    event_of_interest: int = 1,
    competing_events: list[int] = None
) -> pl.DataFrame:
    """
    Compute the Aalen-Johansen estimator for cumulative incidence in the presence of competing risks.

    Parameters
    ----------
    times : pl.Series
        Event or censoring times for each observation.
    reals : pl.Series
        Event type for each observation (0 for censored, event codes otherwise).
    event_of_interest : int, optional
        The event code for the event of interest (default is 1).
    competing_events : list of int, optional
        List of event codes considered as competing events. If None, defaults to [2].

    Returns
    -------
    pl.DataFrame
        DataFrame with columns:
        - 'time': unique event times
        - 'cuminc': cumulative incidence estimate at each time

    Notes
    -----
    This implementation assumes that event codes are integers, with 0 indicating censoring.
    """
    if competing_events is None:
        competing_events = [2]
    df = pl.DataFrame({"times": times, "reals": reals})
    df = df.sort("times")

    event_times = df.filter(pl.col("reals") != 0)["times"].unique().sort()
    result = []
    cum_hazard = 0.0

    for t in event_times:
        n_at_risk = df.filter(pl.col("times") >= t).height
        n_event = df.filter((pl.col("times") == t) & (pl.col("reals") == event_of_interest)).height

        if n_at_risk > 0:
            hazard = n_event / n_at_risk
            cum_hazard += hazard
            result.append({"time": t, "cuminc": cum_hazard})

    return pl.DataFrame(result)

def create_sorted_times_and_reals_data(
        times: pl.Series,
        reals: pl.Series
):
    return pl.DataFrame({ 
        "times": times,
        "reals": reals
    }).sort("times")
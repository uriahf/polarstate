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

def add_events_at_times_column(
        sorted_times_and_reals: pl.DataFrame
):
    return sorted_times_and_reals.with_columns(
        (pl.col("count_0") + pl.col("count_1") + pl.col("count_2")).alias("events_at_times")
    )
     

def group_reals_by_times(df: pl.DataFrame) -> pl.DataFrame:
    """
    Count occurrences of each event type (0, 1, 2) per unique observed time.

    Parameters
    ----------
    df : pl.DataFrame
        A Polars DataFrame with at least two columns:
        - 'times' (int): The observed time for each record.
        - 'reals' (int): The event type for each record, where:
            - 0 indicates censoring,
            - 1 indicates the primary event,
            - 2 indicates a competing event.

    Returns
    -------
    pl.DataFrame
        A DataFrame with one row per unique time and three additional columns:
        - 'count_0': Number of censored observations at that time.
        - 'count_1': Number of primary events at that time.
        - 'count_2': Number of competing events at that time.

    Notes
    -----
    - Input is assumed to be clean (i.e., `times` and `reals` are properly typed).
    - Times are sorted in ascending order in the output.
    - If a particular event type does not occur at a time, its count will be 0.
    """
    return (
        df.group_by("times")
          .agg([
              (pl.col("reals") == 0).sum().cast(pl.Int64).alias("count_0"),
              (pl.col("reals") == 1).sum().cast(pl.Int64).alias("count_1"),
              (pl.col("reals") == 2).sum().cast(pl.Int64).alias("count_2"),
          ])
          .sort("times")
    )

def add_at_risk_column(
        events_data: pl.DataFrame
) -> pl.DataFrame:
    """
    Add a column to the DataFrame that counts the number of individuals at risk at each time point.

    Parameters
    ----------
    events_data : pl.DataFrame
        A DataFrame with columns 'times', 'count_0', 'count_1', and 'count_2'.

    Returns
    -------
    pl.DataFrame
        The input DataFrame with an additional column 'at_risk' that contains the number of individuals at risk at each time point.
    """
    return events_data.with_columns(
        pl.col("events_at_times").cum_sum(reverse=True).alias("at_risk")
    )
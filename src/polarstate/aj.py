import polars as pl


def create_sorted_times_and_reals_data(times: pl.Series, reals: pl.Series):
    return pl.DataFrame({"times": times, "reals": reals}).sort("times")


def add_events_at_times_column(sorted_times_and_reals: pl.DataFrame):
    return sorted_times_and_reals.with_columns(
        (pl.col("count_0") + pl.col("count_1") + pl.col("count_2")).alias(
            "events_at_times"
        )
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
        .agg(
            [
                (pl.col("reals") == 0).sum().cast(pl.Int64).alias("count_0"),
                (pl.col("reals") == 1).sum().cast(pl.Int64).alias("count_1"),
                (pl.col("reals") == 2).sum().cast(pl.Int64).alias("count_2"),
            ]
        )
        .sort("times")
    )


def add_at_risk_column(events_data: pl.DataFrame) -> pl.DataFrame:
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


def add_cause_specific_hazards_columns(events_data: pl.DataFrame) -> pl.DataFrame:
    """
    Add columns for cause-specific hazards and conditional survival at each time point.

    Parameters
    ----------
    events_data : pl.DataFrame
        A DataFrame with columns:
        - 'count_0': number of censored individuals at each time point,
        - 'count_1': number of primary events at each time point,
        - 'count_2': number of competing events at each time point,
        - 'at_risk': number of individuals at risk at each time point.

    Returns
    -------
    pl.DataFrame
        The input DataFrame with three additional columns:
        - 'csh_1': cause-specific hazard for event type 1 (count_1 / at_risk)
        - 'csh_2': cause-specific hazard for event type 2 (count_2 / at_risk)
        - 'conditional_survival': probability of not having any event at that time (count_0 / at_risk)
    """
    return events_data.with_columns(
        [
            (pl.col("count_1") / pl.col("at_risk")).alias("csh_1"),
            (pl.col("count_2") / pl.col("at_risk")).alias("csh_2"),
        ]
    ).with_columns(
        [
            (1 - pl.col("csh_1") - pl.col("csh_2")).alias("conditional_survival"),
        ]
    )


def add_overall_survival_column(events_data: pl.DataFrame) -> pl.DataFrame:
    """
    Add a column for overall survival, defined as the cumulative product of
    the conditional survival probabilities up to and including each time point.

    Parameters
    ----------
    events_data : pl.DataFrame
        A Polars DataFrame with a column 'conditional_survival' representing
        the probability of surviving past each time point.

    Returns
    -------
    pl.DataFrame
        The input DataFrame with an additional column 'overall_survival',
        which contains the Kaplan-Meier-style survival probability at each time.
    """
    return events_data.with_columns(
        pl.col("conditional_survival").cum_prod().alias("overall_survival")
    )


def add_previous_overal_survival_column(events_data: pl.DataFrame) -> pl.DataFrame:
    """
    Add a column for previous overall survival, defined as the overall survival probability just before the current time point.
    Parameters
    ----------
    events_data : pl.DataFrame
        A Polars DataFrame with a column 'overall_survival' representing the overall survival probability at each time point.
    Returns
    -------
    pl.DataFrame
        The input DataFrame with an additional column 'previous_overall_survival',
        which contains the overall survival probability at the previous time point.
    """
    return events_data.with_columns(
        pl.col("overall_survival")
        .shift(1, fill_value=1)
        .alias("previous_overall_survival")
    )


def add_transition_probabilities_at_times_columns(
    events_data: pl.DataFrame,
) -> pl.DataFrame:
    """
    Add columns for transition probabilities at each time point based on cause-specific hazards and previous overall survival.
    Parameters
    ----------
    events_data : pl.DataFrame
        A Polars DataFrame with columns:
        - 'csh_1': cause-specific hazard for event type 1,
        - 'csh_2': cause-specific hazard for event type 2,
        - 'previous_overall_survival': overall survival probability at the previous time point.
    Returns
    -------
    pl.DataFrame
        The input DataFrame with additional columns:
        - 'trainsition_probabilities_to_1_at_times': transition probability to event type 1 at each time point,
        - 'trainsition_probabilities_to_2_at_times': transition probability to event type 2 at each time point.
    """
    return events_data.with_columns(
        [
            (pl.col("csh_1") * pl.col("previous_overall_survival")).alias(
                "trainsition_probabilities_to_1_at_times"
            ),
            (pl.col("csh_2") * pl.col("previous_overall_survival")).alias(
                "trainsition_probabilities_to_2_at_times"
            ),
        ]
    )


def add_state_occupancy_probabilities_at_times_columns(
    events_data: pl.DataFrame,
) -> pl.DataFrame:
    """
    Add columns for state occupancy probabilities at each time point based on trainsition_probabilities_to_1_at_times and trainsition_probabilities_to_2_at_times columns.
    Parameters
    ----------
    events_data : pl.DataFrame
        A Polars DataFrame with columns 'trainsition_probabilities_to_1_at_times' and 'trainsition_probabilities_to_2_at_times'.
    Returns
    -------
    pl.DataFrame
        The input DataFrame with additional columns 'state_occupancy_probability_1_at_times' and 'state_occupancy_probability_2_at_times',
        which contain the state occupancy probabilities at each time point. This function should sum all the previous values from state_occupancy_probability_1_at_times and state_occupancy_probability_2_at_times accordingly and assign the sum of the previous values to the new columns
        state_occupancy_probability_1 and state_occupancy_probability_2.
    """
    return events_data.with_columns(
        [
            pl.col("trainsition_probabilities_to_1_at_times")
            .cum_sum()
            .alias("state_occupancy_probability_1_at_times"),
            pl.col("trainsition_probabilities_to_2_at_times")
            .cum_sum()
            .alias("state_occupancy_probability_2_at_times"),
        ]
    )


def prepare_event_table(times_and_reals: pl.DataFrame) -> pl.DataFrame:
    """Generate the full event table from raw ``times`` and ``reals`` data.

    Parameters
    ----------
    times_and_reals : pl.DataFrame
        A Polars DataFrame containing at least ``times`` and ``reals`` columns.

    Returns
    -------
    pl.DataFrame
        The event table with all intermediate columns computed.
    """

    return (
        times_and_reals.pipe(group_reals_by_times)
        .pipe(add_events_at_times_column)
        .pipe(add_at_risk_column)
        .pipe(add_cause_specific_hazards_columns)
        .pipe(add_overall_survival_column)
        .pipe(add_previous_overal_survival_column)
        .pipe(add_transition_probabilities_at_times_columns)
        .pipe(add_state_occupancy_probabilities_at_times_columns)
    )

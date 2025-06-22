import polars as pl


def predict_aj_estimates(
    event_table: pl.DataFrame, fixed_time_horizons: pl.Series
) -> pl.DataFrame:
    """Predict state-occupancy probabilities at ``fixed_time_horizons``.

    Parameters
    ----------
    event_table : pl.DataFrame
        The event table created by :func:`prepare_event_table`.
    fixed_time_horizons : pl.Series
        Times at which to obtain the state-occupancy probabilities.

    Returns
    -------
    pl.DataFrame
        DataFrame with ``fixed_time_horizons`` and the estimated probabilities
        for states 0, 1 and 2.
    """

    event_table = event_table.sort("times")

    horizons_df = pl.DataFrame({"fixed_time_horizons": fixed_time_horizons}).sort(
        "fixed_time_horizons"
    )

    joined = horizons_df.join_asof(
        event_table, left_on="fixed_time_horizons", right_on="times"
    )

    joined = joined.with_columns(
        [
            pl.col("state_occupancy_probability_1_at_times")
            .fill_null(0.0)
            .alias("state_occupancy_probability_1"),
            pl.col("state_occupancy_probability_2_at_times")
            .fill_null(0.0)
            .alias("state_occupancy_probability_2"),
        ]
    ).with_columns(
        (
            1
            - pl.col("state_occupancy_probability_1")
            - pl.col("state_occupancy_probability_2")
        ).alias("state_occupancy_probability_0")
    )

    return joined.select(
        [
            "fixed_time_horizons",
            "state_occupancy_probability_0",
            "state_occupancy_probability_1",
            "state_occupancy_probability_2",
        ]
    )

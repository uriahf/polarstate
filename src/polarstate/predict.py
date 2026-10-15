import polars as pl

from .transition_steps import (
    TransitionTableSteps,
    _predict_transition_steps,
)


def _validate_prediction_inputs(
    event_table: pl.DataFrame,
    fixed_time_horizons: pl.Series,
    full_event_table: bool,
) -> None:
    if not isinstance(event_table, pl.DataFrame):
        raise TypeError("event_table must be a Polars DataFrame.")
    if not isinstance(fixed_time_horizons, pl.Series):
        raise TypeError("fixed_time_horizons must be a Polars Series.")
    if not isinstance(full_event_table, bool):
        raise TypeError("full_event_table must be a boolean.")

    required = {
        "times",
        "state_occupancy_probability_1_at_times",
        "state_occupancy_probability_2_at_times",
    }
    missing = required - set(event_table.columns)
    if missing:
        raise ValueError(
            "event_table is missing required column(s): " + ", ".join(sorted(missing))
        )
    if event_table.is_empty():
        raise ValueError("event_table must contain at least one row.")
    if fixed_time_horizons.is_empty():
        raise ValueError("fixed_time_horizons must contain at least one value.")
    if not fixed_time_horizons.dtype.is_numeric():
        raise TypeError("fixed_time_horizons must have a numeric dtype.")
    if fixed_time_horizons.null_count():
        raise ValueError("fixed_time_horizons must not contain null values.")
    if fixed_time_horizons.is_finite().not_().any():
        raise ValueError("fixed_time_horizons must contain only finite values.")
    if (fixed_time_horizons < 0).any():
        raise ValueError("fixed_time_horizons must contain non-negative values.")


def predict_aj_estimates(
    event_table: pl.DataFrame | TransitionTableSteps,
    fixed_time_horizons: pl.Series | None = None,
    full_event_table: bool = False,
) -> pl.DataFrame:
    """Evaluate state-occupancy probabilities at fixed time horizons or scheduled steps.

    Parameters
    ----------
    event_table : pl.DataFrame or TransitionTableSteps
        Event table returned by ``prepare_event_table`` or transition table returned
        by ``prepare_transition_table_steps``.
    fixed_time_horizons : pl.Series, optional
        Numeric horizons to evaluate for time-based event tables, or integer step horizons
        for step transition tables. If omitted for step transition tables, occupancy for
        all observed steps is returned.
    full_event_table : bool, default False
        If ``True``, append estimates at every observed event time. Use the
        ``estimate_origin`` column to distinguish those rows from requested
        horizons. Only supported for continuous event tables.

    Returns
    -------
    pl.DataFrame
        For competing risk event tables: columns are ``times``, state-occupancy probabilities for
        states 0, 1, and 2, and ``estimate_origin``.
        For discrete step transition tables: columns are ``step``, ``state``, and ``state_occupancy_probability``.

    Raises
    ------
    TypeError
        If inputs are not the documented Polars/polarstate types, horizons are not of
        expected numeric/integer dtypes, or full_event_table is not boolean.
    ValueError
        If required event-table columns are missing, either input is empty,
        or horizons contain null, non-finite, invalid, or unobserved step values.
    """
    if isinstance(event_table, TransitionTableSteps):
        if full_event_table:
            raise ValueError("full_event_table is not supported for step panels")
        return _predict_transition_steps(event_table, fixed_time_horizons)

    _validate_prediction_inputs(
        event_table,
        fixed_time_horizons,
        full_event_table,  # type: ignore[arg-type]
    )

    estimate_origin_enum = pl.Enum(["fixed_time_horizons", "event_table"])

    event_table = event_table.sort("times")

    horizons_df = pl.DataFrame({"times": fixed_time_horizons}).sort("times")

    joined = horizons_df.join_asof(
        event_table, left_on="times", right_on="times"
    ).with_columns(
        pl.lit("fixed_time_horizons")
        .cast(estimate_origin_enum)
        .alias("estimate_origin")
    )

    if full_event_table:
        joined = pl.concat(
            [
                joined,
                event_table.with_columns(
                    pl.lit("event_table")
                    .cast(estimate_origin_enum)
                    .alias("estimate_origin")
                ),
            ],
            how="vertical",
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
            "times",
            "state_occupancy_probability_0",
            "state_occupancy_probability_1",
            "state_occupancy_probability_2",
            "estimate_origin",
        ]
    )

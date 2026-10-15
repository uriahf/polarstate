import pandas as pd
import polars as pl

from polarstate import prepare_event_table, predict_aj_estimates


def test_documented_recipes_execute() -> None:
    single = pl.DataFrame({
        "times": [2.0, 4.0, 5.0, 8.0, 10.0],
        "reals": [1, 0, 1, 0, 1],
    })
    single_table = prepare_event_table(single)
    single_result = predict_aj_estimates(
        single_table, pl.Series([3.0, 6.0, 12.0])
    )
    assert single_result["state_occupancy_probability_2"].sum() == 0

    competing = pl.DataFrame({
        "times": [2.0, 4.0, 5.0, 8.0, 10.0, 12.0],
        "reals": [1, 2, 0, 1, 2, 1],
    })
    competing_table = prepare_event_table(competing)
    combined = predict_aj_estimates(
        competing_table,
        pl.Series([3.0, 6.0, 9.0, 12.0]),
        full_event_table=True,
    )
    assert set(combined["estimate_origin"].cast(pl.String)) == {
        "fixed_time_horizons", "event_table"
    }

    grouped = competing.with_columns(
        pl.Series("group", ["A", "A", "A", "B", "B", "B"])
    )
    estimates = {
        group[0]: predict_aj_estimates(
            data.select("times", "reals").pipe(prepare_event_table),
            pl.Series([6.0, 12.0]),
        )
        for group, data in grouped.group_by("group")
    }
    assert set(estimates) == {"A", "B"}

    pandas_data = pd.DataFrame({
        "time": [2, 4, 6, 8], "event": [1, 0, 2, 1]
    })
    converted = pl.from_pandas(pandas_data).select(
        pl.col("time").cast(pl.Float64).alias("times"),
        pl.col("event").cast(pl.Int64).alias("reals"),
    )
    assert prepare_event_table(converted).height == 4

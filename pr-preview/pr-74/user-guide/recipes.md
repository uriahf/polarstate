# Recipes

Every example on this page is executed when the documentation builds. If an API change breaks a recipe, the documentation check fails with it.


``` python
import pandas as pd
import polars as pl
from polarstate import prepare_event_table, predict_aj_estimates
```


# Choose the event structure

The workflow stays the same; only the outcome coding changes.


- <a href="" id="tabset-1-1-tab" class="nav-link active" data-bs-toggle="tab" data-bs-target="#tabset-1-1" role="tab" aria-controls="tabset-1-1" aria-selected="true">Single event</a>
- <a href="" id="tabset-1-2-tab" class="nav-link" data-bs-toggle="tab" data-bs-target="#tabset-1-2" role="tab" aria-controls="tabset-1-2" aria-selected="false">Competing risks</a>


Use `0` for right-censoring and `1` for the event.


``` python
single_event_data = pl.DataFrame({
    "times": [2.0, 4.0, 5.0, 8.0, 10.0],
    "reals": [1, 0, 1, 0, 1],
})
single_table = prepare_event_table(single_event_data)
single_result = predict_aj_estimates(
    single_table, pl.Series([3.0, 6.0, 12.0])
)
single_result
```


shape: (3, 5)

| times | state_occupancy_probability_0 | state_occupancy_probability_1 | state_occupancy_probability_2 | estimate_origin |
|----|----|----|----|----|
| f64 | f64 | f64 | f64 | enum |
| 3.0 | 0.8 | 0.2 | 0.0 | "fixed_time_horizons" |
| 6.0 | 0.533333 | 0.466667 | 0.0 | "fixed_time_horizons" |
| 12.0 | 0.0 | 1.0 | 0.0 | "fixed_time_horizons" |


State 2 remains zero, while state 0 is the event-free probability.


Encode the event of interest as `1` and the competing event as `2`.


``` python
competing_data = pl.DataFrame({
    "times": [2.0, 4.0, 5.0, 8.0, 10.0, 12.0],
    "reals": [1, 2, 0, 1, 2, 1],
})
competing_table = prepare_event_table(competing_data)
competing_result = predict_aj_estimates(
    competing_table, pl.Series([3.0, 6.0, 9.0, 12.0])
)
competing_result
```


shape: (4, 5)

| times | state_occupancy_probability_0 | state_occupancy_probability_1 | state_occupancy_probability_2 | estimate_origin |
|----|----|----|----|----|
| f64 | f64 | f64 | f64 | enum |
| 3.0 | 0.833333 | 0.166667 | 0.0 | "fixed_time_horizons" |
| 6.0 | 0.666667 | 0.166667 | 0.166667 | "fixed_time_horizons" |
| 9.0 | 0.444444 | 0.388889 | 0.166667 | "fixed_time_horizons" |
| 12.0 | -1.1102e-16 | 0.611111 | 0.388889 | "fixed_time_horizons" |


Do not censor competing events: coding them as `0` changes the estimand.


# Return horizons and event times together


``` python
combined_result = predict_aj_estimates(
    competing_table,
    pl.Series([3.0, 6.0, 9.0, 12.0]),
    full_event_table=True,
)
combined_result.select("times", "estimate_origin")
```


shape: (10, 2)

| times | estimate_origin       |
|-------|-----------------------|
| f64   | enum                  |
| 3.0   | "fixed_time_horizons" |
| 6.0   | "fixed_time_horizons" |
| 9.0   | "fixed_time_horizons" |
| 12.0  | "fixed_time_horizons" |
| 2.0   | "event_table"         |
| 4.0   | "event_table"         |
| 5.0   | "event_table"         |
| 8.0   | "event_table"         |
| 10.0  | "event_table"         |
| 12.0  | "event_table"         |


Use `estimate_origin` to distinguish requested horizons from observed times.


# Estimate within groups

The public functions operate on one cohort at a time.


``` python
data_with_groups = competing_data.with_columns(
    pl.Series("group", ["A", "A", "A", "B", "B", "B"])
)
estimates_by_group = {
    group[0]: predict_aj_estimates(
        prepare_event_table(group_data.select("times", "reals")),
        pl.Series([6.0, 12.0]),
    )
    for group, group_data in data_with_groups.group_by("group")
}
{k: value.shape for k, value in estimates_by_group.items()}
```


    {'A': (2, 5), 'B': (2, 5)}


These are descriptive within-group estimates, not an adjusted regression model.


# Convert pandas input


``` python
pandas_data = pd.DataFrame({
    "time": [2, 4, 6, 8],
    "event": [1, 0, 2, 1],
})
polars_data = pl.from_pandas(pandas_data).select(
    pl.col("time").cast(pl.Float64).alias("times"),
    pl.col("event").cast(pl.Int64).alias("reals"),
)
prepare_event_table(polars_data).select("times", "at_risk")
```


shape: (4, 2)

| times | at_risk |
|-------|---------|
| f64   | i64     |
| 2.0   | 4       |
| 4.0   | 3       |
| 6.0   | 2       |
| 8.0   | 1       |


Recode outcomes before estimation and confirm that only `0`, `1`, and optional `2` are present.


# Next steps

- Read [How the estimator works](how-it-works.md).
- Review [Assumptions and limitations](assumptions-limitations.md).
- See [Package comparisons](package-comparisons.md).
- Consult the [API Reference](../reference/index.md).

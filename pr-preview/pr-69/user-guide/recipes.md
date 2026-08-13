# Recipes


# Choose the event structure

The estimation workflow is unchanged; only the outcome coding differs.


- <a href="" id="tabset-1-1-tab" class="nav-link active" data-bs-toggle="tab" data-bs-target="#tabset-1-1" role="tab" aria-controls="tabset-1-1" aria-selected="true">Single event</a>
- <a href="" id="tabset-1-2-tab" class="nav-link" data-bs-toggle="tab" data-bs-target="#tabset-1-2" role="tab" aria-controls="tabset-1-2" aria-selected="false">Competing risks</a>


Use `0` when follow-up ended without an observed event (right-censoring) and `1` for the event. No competing-event rows are required.

``` python
import polars as pl
from polarstate import prepare_event_table, predict_aj_estimates

data = pl.DataFrame({
    "times": [2.0, 4.0, 5.0, 8.0, 10.0],
    "reals": [1, 0, 1, 0, 1],
})

table = prepare_event_table(data)
result = predict_aj_estimates(table, pl.Series([3.0, 6.0, 12.0]))
```

Here, `state_occupancy_probability_2` is zero and state 0 is the event-free probability.


Encode the event of interest as `1` and the competing event as `2`.

``` python
import polars as pl
from polarstate import prepare_event_table, predict_aj_estimates

data = pl.DataFrame({
    "times": [2.0, 4.0, 5.0, 8.0, 10.0, 12.0],
    "reals": [1, 2, 0, 1, 2, 1],
})

table = prepare_event_table(data)
result = predict_aj_estimates(table, pl.Series([3.0, 6.0, 9.0, 12.0]))
```

Do not censor competing events. Encoding them as `0` changes the estimand and generally overstates the cumulative incidence of the event of interest.


# Return horizons and event times together

``` python
result = predict_aj_estimates(
    table,
    pl.Series([3.0, 6.0, 9.0, 12.0]),
    full_event_table=True,
)
```

Use `estimate_origin` to distinguish requested horizons from rows copied from the event table.


# Estimate within groups

The public functions operate on one cohort at a time. Split explicitly when you need stratum-specific estimates.

``` python
estimates_by_group = {
    group[0]: predict_aj_estimates(
        prepare_event_table(group_data.select("times", "reals")),
        pl.Series([6.0, 12.0]),
    )
    for group, group_data in data_with_groups.group_by("group")
}
```

This produces descriptive estimates within each group; it is not an adjusted regression model.


# Convert pandas input

``` python
polars_data = pl.from_pandas(pandas_data).select(
    pl.col("time").cast(pl.Float64).alias("times"),
    pl.col("event").cast(pl.Int64).alias("reals"),
)
```

Recode outcomes before estimation and confirm that only `0`, `1`, and optional `2` are present.

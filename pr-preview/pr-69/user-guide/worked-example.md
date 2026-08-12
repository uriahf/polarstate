# Worked Example

This tutorial follows a small cohort from subject-level observations to state-occupancy estimates. The code is executed when the documentation builds, so it also acts as a continuously checked example.


# Create the observations

Each row contains an observed time and an outcome code: `0` for censoring, `1` for the event of interest, and optional `2` for a competing event.


``` python
import polars as pl
from polarstate import prepare_event_table, predict_aj_estimates

observations = pl.DataFrame(
    {
        "times": [4.3, 9.7, 14.2, 18.6, 24.1, 31.5, 34.8, 39.2, 46.0, 49.9],
        "reals": [1, 1, 2, 1, 1, 0, 0, 1, 2, 1],
    }
)
observations
```


shape: (10, 2)

| times | reals |
|-------|-------|
| f64   | i64   |
| 4.3   | 1     |
| 9.7   | 1     |
| 14.2  | 2     |
| 18.6  | 1     |
| 24.1  | 1     |
| 31.5  | 0     |
| 34.8  | 0     |
| 39.2  | 1     |
| 46.0  | 2     |
| 49.9  | 1     |


# Prepare the event table


``` python
event_table = prepare_event_table(observations)
event_table.select(
    "times",
    "at_risk",
    "count_0",
    "count_1",
    "count_2",
    "overall_survival",
    "state_occupancy_probability_1_at_times",
    "state_occupancy_probability_2_at_times",
)
```


shape: (10, 8)

| times | at_risk | count_0 | count_1 | count_2 | overall_survival | state_occupancy_probability_1_at_times | state_occupancy_probability_2_at_times |
|----|----|----|----|----|----|----|----|
| f64 | i64 | i64 | i64 | i64 | f64 | f64 | f64 |
| 4.3 | 10 | 0 | 1 | 0 | 0.9 | 0.1 | 0.0 |
| 9.7 | 9 | 0 | 1 | 0 | 0.8 | 0.2 | 0.0 |
| 14.2 | 8 | 0 | 0 | 1 | 0.7 | 0.2 | 0.1 |
| 18.6 | 7 | 0 | 1 | 0 | 0.6 | 0.3 | 0.1 |
| 24.1 | 6 | 0 | 1 | 0 | 0.5 | 0.4 | 0.1 |
| 31.5 | 5 | 1 | 0 | 0 | 0.5 | 0.4 | 0.1 |
| 34.8 | 4 | 1 | 0 | 0 | 0.5 | 0.4 | 0.1 |
| 39.2 | 3 | 0 | 1 | 0 | 0.333333 | 0.566667 | 0.1 |
| 46.0 | 2 | 0 | 0 | 1 | 0.166667 | 0.566667 | 0.266667 |
| 49.9 | 1 | 0 | 1 | 0 | 0.0 | 0.733333 | 0.266667 |


The risk set decreases as observations experience an event or are censored. The two cumulative state-occupancy columns record the estimated probability of having entered each absorbing state by each observed time.


# Predict at useful horizons


``` python
horizons = pl.Series("times", [0.0, 10.0, 20.0, 30.0, 40.0, 50.0])
estimates = predict_aj_estimates(event_table, horizons)
estimates
```


shape: (6, 5)

| times | state_occupancy_probability_0 | state_occupancy_probability_1 | state_occupancy_probability_2 | estimate_origin |
|----|----|----|----|----|
| f64 | f64 | f64 | f64 | enum |
| 0.0 | 1.0 | 0.0 | 0.0 | "fixed_time_horizons" |
| 10.0 | 0.8 | 0.2 | 0.0 | "fixed_time_horizons" |
| 20.0 | 0.6 | 0.3 | 0.1 | "fixed_time_horizons" |
| 30.0 | 0.5 | 0.4 | 0.1 | "fixed_time_horizons" |
| 40.0 | 0.333333 | 0.566667 | 0.1 | "fixed_time_horizons" |
| 50.0 | -5.5511e-17 | 0.733333 | 0.266667 | "fixed_time_horizons" |


``` python
estimates.with_columns(
    pl.sum_horizontal(
        "state_occupancy_probability_0",
        "state_occupancy_probability_1",
        "state_occupancy_probability_2",
    ).alias("probability_sum")
)
```


shape: (6, 6)

| times | state_occupancy_probability_0 | state_occupancy_probability_1 | state_occupancy_probability_2 | estimate_origin | probability_sum |
|----|----|----|----|----|----|
| f64 | f64 | f64 | f64 | enum | f64 |
| 0.0 | 1.0 | 0.0 | 0.0 | "fixed_time_horizons" | 1.0 |
| 10.0 | 0.8 | 0.2 | 0.0 | "fixed_time_horizons" | 1.0 |
| 20.0 | 0.6 | 0.3 | 0.1 | "fixed_time_horizons" | 1.0 |
| 30.0 | 0.5 | 0.4 | 0.1 | "fixed_time_horizons" | 1.0 |
| 40.0 | 0.333333 | 0.566667 | 0.1 | "fixed_time_horizons" | 1.0 |
| 50.0 | -5.5511e-17 | 0.733333 | 0.266667 | "fixed_time_horizons" | 1.0 |


Every row sums to one, up to floating-point precision. State 0 means no absorbing event has occurred by the horizon; states 1 and 2 are the cumulative probabilities of the event of interest and competing event.

> **Tip: Tip**
>
> To model a simple single-event analysis, use the same workflow with only codes `0` and `1`. The state-2 probability remains zero.

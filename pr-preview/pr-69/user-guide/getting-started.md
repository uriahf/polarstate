# Getting Started

`polarstate` implements a compact Aalen-Johansen workflow with Polars DataFrames. It works for ordinary single-event analyses and binary event/censoring data, while also supporting an optional competing event.

| Code | Meaning                                                   |
|-----:|-----------------------------------------------------------|
|  `0` | Right-censored: follow-up ended without an observed event |
|  `1` | Event of interest                                         |
|  `2` | Competing event (optional)                                |

Code `0` may represent administrative censoring at the end of observation or another form of right-censoring. It does not assert that the person would have remained event-free after follow-up ended. For a single-event analysis, use only codes `0` and `1`.


# Install

With [uv](https://docs.astral.sh/uv/):

``` bash
uv add polarstate
```

Alternatively, with pip:

``` bash
pip install polarstate
```


# Prepare an event table

Start with one row per observation. The `times` column contains the observed time and `reals` contains the outcome code.


``` python
import polars as pl
from polarstate import prepare_event_table

observations = pl.DataFrame(
    {
        "times": [24.1, 9.7, 49.9, 18.6, 34.8, 14.2, 39.2, 46.0, 31.5, 4.3],
        "reals": [1, 1, 1, 1, 0, 2, 1, 2, 0, 1],
    }
)

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


The rendered table above is the actual output from `polarstate`, generated when the documentation builds. It exposes the risk set, outcome counts, overall survival, and cumulative state-occupancy probabilities.


# Predict at fixed horizons


``` python
from polarstate import predict_aj_estimates

estimates = predict_aj_estimates(
    event_table,
    pl.Series([10.0, 20.0, 30.0, 40.0, 50.0]),
)
estimates
```


shape: (5, 5)

| times | state_occupancy_probability_0 | state_occupancy_probability_1 | state_occupancy_probability_2 | estimate_origin |
|----|----|----|----|----|
| f64 | f64 | f64 | f64 | enum |
| 10.0 | 0.8 | 0.2 | 0.0 | "fixed_time_horizons" |
| 20.0 | 0.6 | 0.3 | 0.1 | "fixed_time_horizons" |
| 30.0 | 0.5 | 0.4 | 0.1 | "fixed_time_horizons" |
| 40.0 | 0.333333 | 0.566667 | 0.1 | "fixed_time_horizons" |
| 50.0 | -5.5511e-17 | 0.733333 | 0.266667 | "fixed_time_horizons" |


The rendered result contains the requested time and one probability for each state. Each row sums to one (up to floating-point precision).

> **Tip: Need the observed event times too?**
>
> Pass `full_event_table=True` to the prediction function to include estimates at every observed event time alongside your fixed horizons.


# Where to go next

- Continue with the [worked example](worked-example.md) for a complete, executable analysis.
- Read [How the estimator works](how-it-works.md) for the statistical flow.
- Open the API Reference for signatures, parameter details, and source links.

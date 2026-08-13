# polarstate

Fast, Polars-native Aalen-Johansen estimates for time-to-event data.

`polarstate` turns event times and outcomes into state-occupancy probabilities
at the time horizons you care about. It supports ordinary single-event analyses
and binary event/censoring data, with optional competing events.

The workflow has two explicit steps:

1. `prepare_event_table()` turns subject-level observations into an inspectable
   Aalen-Johansen event table at each observed time.
2. `predict_aj_estimates()` evaluates those estimates at the time horizons you
   request.

## Install

With [uv](https://docs.astral.sh/uv/):

```bash
uv add polarstate
```

Alternatively, with pip:

```bash
pip install polarstate
```

## Quick start

Start with one row per observation:

```python
import polars as pl
from polarstate import prepare_event_table, predict_aj_estimates

observations = pl.DataFrame(
    {
        "times": [24.1, 9.7, 49.9, 18.6, 34.8, 14.2, 39.2, 46.0, 31.5, 4.3],
        "reals": [1, 1, 1, 1, 0, 2, 1, 2, 0, 1],
    }
)
```

### 1. Prepare the event table

```python
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

Selected columns from the output:

| times | at_risk | count_0 | count_1 | count_2 | overall_survival | state 1 probability | state 2 probability |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 4.3 | 10 | 0 | 1 | 0 | 0.900000 | 0.100000 | 0.000000 |
| 9.7 | 9 | 0 | 1 | 0 | 0.800000 | 0.200000 | 0.000000 |
| 14.2 | 8 | 0 | 0 | 1 | 0.700000 | 0.200000 | 0.100000 |
| 18.6 | 7 | 0 | 1 | 0 | 0.600000 | 0.300000 | 0.100000 |
| 24.1 | 6 | 0 | 1 | 0 | 0.500000 | 0.400000 | 0.100000 |
| 31.5 | 5 | 1 | 0 | 0 | 0.500000 | 0.400000 | 0.100000 |
| 34.8 | 4 | 1 | 0 | 0 | 0.500000 | 0.400000 | 0.100000 |
| 39.2 | 3 | 0 | 1 | 0 | 0.333333 | 0.566667 | 0.100000 |
| 46.0 | 2 | 0 | 0 | 1 | 0.166667 | 0.566667 | 0.266667 |
| 49.9 | 1 | 0 | 1 | 0 | 0.000000 | 0.733333 | 0.266667 |

The complete event table also retains the intermediate hazards, survival
increments, and transition probabilities used in the calculation.

### 2. Query fixed time horizons

```python
estimates = predict_aj_estimates(
    event_table,
    pl.Series([10.0, 20.0, 30.0, 40.0]),
)

estimates
```

Output:

| times | state 0 probability | state 1 probability | state 2 probability | estimate_origin |
|---:|---:|---:|---:|---|
| 10.0 | 0.800000 | 0.200000 | 0.000000 | fixed_time_horizons |
| 20.0 | 0.600000 | 0.300000 | 0.100000 | fixed_time_horizons |
| 30.0 | 0.500000 | 0.400000 | 0.100000 | fixed_time_horizons |
| 40.0 | 0.333333 | 0.566667 | 0.100000 | fixed_time_horizons |

State 0 is remaining event-free, state 1 is the event of interest, and state 2
is the competing event.

Code `0` in the input indicates right-censoring: follow-up ended without an
observed event. This includes administrative censoring at the end of
observation. Code `1` indicates the event of interest, and optional code `2`
a competing event.

## Why polarstate?

- Polars-native inputs and outputs
- A compact, explicit event table you can inspect
- Predictions at arbitrary time horizons
- Support for single-event, binary event/censoring, and competing-risks data


## Learn more

- [How the estimator works](user_guide/02-how-it-works.qmd) explains the
  Aalen-Johansen calculation and every returned column.
- [Recipes](user_guide/05-recipes.qmd) covers single-event, competing-risks,
  grouped, and pandas-to-Polars workflows.
- The [API Reference](https://uriahf.github.io/polarstate/reference/) provides
  complete signatures and parameter details.

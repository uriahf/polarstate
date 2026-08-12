# polarstate

Fast, Polars-native Aalen-Johansen estimates for time-to-event data.

`polarstate` turns event times and outcomes into an event table, then predicts
state-occupancy probabilities at the time horizons you care about. It supports
ordinary single-event analyses and binary event/censoring data, with optional
competing events. The public API is deliberately small: one function prepares
the estimates and one function queries them.

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

```python
import polars as pl
from polarstate import prepare_event_table, predict_aj_estimates

observations = pl.DataFrame(
    {
        "times": [24.1, 9.7, 49.9, 18.6, 34.8, 14.2, 39.2, 46.0, 31.5, 4.3],
        "reals": [1, 1, 1, 1, 0, 2, 1, 2, 0, 1],
    }
)

event_table = prepare_event_table(observations)
estimates = predict_aj_estimates(
    event_table,
    pl.Series([10.0, 20.0, 30.0, 40.0, 50.0]),
)

print(estimates)
```

Outcomes use `0` for censoring and `1` for the event of interest. Use `2`
only when the data include a competing event. See the
[documentation](https://uriahf.github.io/polarstate/) for a complete
walkthrough and API reference.

## Why polarstate?

- Polars-native inputs and outputs
- A compact, explicit event table you can inspect
- Predictions at arbitrary time horizons
- Support for single-event, binary event/censoring, and competing-risks data

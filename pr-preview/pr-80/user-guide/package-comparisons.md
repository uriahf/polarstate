# Package Comparisons

This page uses the same competing-risks data in three implementations. As in the original polarstate documentation, each tab shows the package's **native output**, not only a normalized summary.


# Shared example

Each row is one observation. Code `0` means right-censored, `1` is the event of interest, and `2` is the competing event.

| `times` | 24.1 | 9.7 | 49.9 | 18.6 | 34.8 | 14.2 | 39.2 | 46.0 | 31.5 | 4.3 |
|--------:|-----:|----:|-----:|-----:|-----:|-----:|-----:|-----:|-----:|----:|
| `reals` |    1 |   1 |    1 |    1 |    0 |    2 |    1 |    2 |    0 |   1 |


- <a href="" id="tabset-1-1-tab" class="nav-link active" data-bs-toggle="tab" data-bs-target="#tabset-1-1" role="tab" aria-controls="tabset-1-1" aria-selected="true">polarstate (Py 🐍)</a>
- <a href="" id="tabset-1-2-tab" class="nav-link" data-bs-toggle="tab" data-bs-target="#tabset-1-2" role="tab" aria-controls="tabset-1-2" aria-selected="false">lifelines (Py 🐍)</a>
- <a href="" id="tabset-1-3-tab" class="nav-link" data-bs-toggle="tab" data-bs-target="#tabset-1-3" role="tab" aria-controls="tabset-1-3" aria-selected="false">tidycmprsk (R 🔵)</a>


`polarstate` first returns the complete Polars event table, exposing the risk set, cause-specific hazards, survival, transition increments, and state occupancy probabilities.

``` python
import polars as pl
from polarstate import prepare_event_table, predict_aj_estimates

data = pl.DataFrame({
    "times": [24.1, 9.7, 49.9, 18.6, 34.8, 14.2, 39.2, 46.0, 31.5, 4.3],
    "reals": [1, 1, 1, 1, 0, 2, 1, 2, 0, 1],
})

event_table = prepare_event_table(data)
print(event_table)
```

Native Polars output:

``` text
shape: (10, 16)
┌───────┬─────────┬─────────┬─────────┬───┬────────────────┬────────────────┬────────────────┬────────────────┐
│ times ┆ count_0 ┆ count_1 ┆ count_2 ┆ … ┆ trainsition_pr ┆ state_occupanc ┆ state_occupanc ┆ state_occupanc │
│ ---   ┆ ---     ┆ ---     ┆ ---     ┆   ┆ obabilities_to ┆ y_probability_ ┆ y_probability_ ┆ y_probability_ │
│ f64   ┆ u32     ┆ u32     ┆ u32     ┆   ┆ _2_at_times    ┆ 0_at_times     ┆ 1_at_times     ┆ 2_at_times     │
│       ┆         ┆         ┆         ┆   ┆ ---            ┆ ---            ┆ ---            ┆ ---            │
│       ┆         ┆         ┆         ┆   ┆ f64            ┆ f64            ┆ f64            ┆ f64            │
╞═══════╪═════════╪═════════╪═════════╪═══╪════════════════╪════════════════╪════════════════╪════════════════╡
│ 4.3   ┆ 0       ┆ 1       ┆ 0       ┆ … ┆ 0.0            ┆ 0.9            ┆ 0.1            ┆ 0.0            │
│ 9.7   ┆ 0       ┆ 1       ┆ 0       ┆ … ┆ 0.0            ┆ 0.8            ┆ 0.2            ┆ 0.0            │
│ 14.2  ┆ 0       ┆ 0       ┆ 1       ┆ … ┆ 0.1            ┆ 0.7            ┆ 0.2            ┆ 0.1            │
│ 18.6  ┆ 0       ┆ 1       ┆ 0       ┆ … ┆ 0.0            ┆ 0.6            ┆ 0.3            ┆ 0.1            │
│ 24.1  ┆ 0       ┆ 1       ┆ 0       ┆ … ┆ 0.0            ┆ 0.5            ┆ 0.4            ┆ 0.1            │
│ 31.5  ┆ 1       ┆ 0       ┆ 0       ┆ … ┆ 0.0            ┆ 0.5            ┆ 0.4            ┆ 0.1            │
│ 34.8  ┆ 1       ┆ 0       ┆ 0       ┆ … ┆ 0.0            ┆ 0.5            ┆ 0.4            ┆ 0.1            │
│ 39.2  ┆ 0       ┆ 1       ┆ 0       ┆ … ┆ 0.0            ┆ 0.333333       ┆ 0.566667       ┆ 0.1            │
│ 46.0  ┆ 0       ┆ 0       ┆ 1       ┆ … ┆ 0.166667       ┆ 0.166667       ┆ 0.566667       ┆ 0.266667       │
│ 49.9  ┆ 0       ┆ 1       ┆ 0       ┆ … ┆ 0.0            ┆ 0.0            ┆ 0.733333       ┆ 0.266667       │
└───────┴─────────┴─────────┴─────────┴───┴────────────────┴────────────────┴────────────────┴────────────────┘
```


## Prediction for specific time horizons

``` python
estimates = predict_aj_estimates(
    event_table,
    pl.Series([10.0, 20.0, 30.0, 40.0, 50.0]),
)
print(estimates)
```

Native Polars output:

``` text
shape: (5, 5)
┌───────┬────────────────────────┬────────────────────────┬────────────────────────┬─────────────────────┐
│ times ┆ state_occupancy_prob…  ┆ state_occupancy_prob…  ┆ state_occupancy_prob…  ┆ estimate_origin     │
│ ---   ┆ ---                    ┆ ---                    ┆ ---                    ┆ ---                 │
│ f64   ┆ f64                    ┆ f64                    ┆ f64                    ┆ enum                │
╞═══════╪════════════════════════╪════════════════════════╪════════════════════════╪═════════════════════╡
│ 10.0  ┆ 0.8                    ┆ 0.2                    ┆ 0.0                    ┆ fixed_time_horizons │
│ 20.0  ┆ 0.6                    ┆ 0.3                    ┆ 0.1                    ┆ fixed_time_horizons │
│ 30.0  ┆ 0.5                    ┆ 0.4                    ┆ 0.1                    ┆ fixed_time_horizons │
│ 40.0  ┆ 0.333333               ┆ 0.566667               ┆ 0.1                    ┆ fixed_time_horizons │
│ 50.0  ┆ 0.0                    ┆ 0.733333               ┆ 0.266667               ┆ fixed_time_horizons │
└───────┴────────────────────────┴────────────────────────┴────────────────────────┴─────────────────────┘
```


`lifelines` fits and returns one cumulative-incidence table for each event type.

``` python
import pandas as pd
from lifelines import AalenJohansenFitter

data = pd.DataFrame({
    "duration": [24.1, 9.7, 49.9, 18.6, 34.8, 14.2, 39.2, 46.0, 31.5, 4.3],
    "event": [1, 1, 1, 1, 0, 2, 1, 2, 0, 1],
})

primary = AalenJohansenFitter(calculate_variance=False)
primary.fit(data["duration"], data["event"], event_of_interest=1)
print("Event table for event type 1")
print(primary.cumulative_density_)

competing = AalenJohansenFitter(calculate_variance=False)
competing.fit(data["duration"], data["event"], event_of_interest=2)
print("Event table for event type 2")
print(competing.cumulative_density_)
```

Native pandas output:

``` text
Event table for event type 1
          CIF_1
event_at
0.0    0.000000
4.3    0.100000
9.7    0.200000
14.2   0.200000
18.6   0.300000
24.1   0.400000
31.5   0.400000
34.8   0.400000
39.2   0.566667
46.0   0.566667
49.9   0.733333

Event table for event type 2
          CIF_2
event_at
0.0    0.000000
4.3    0.000000
9.7    0.000000
14.2   0.100000
18.6   0.100000
24.1   0.100000
31.5   0.100000
34.8   0.100000
39.2   0.100000
46.0   0.266667
49.9   0.266667
```


## Prediction for specific time horizons

``` python
print(primary.predict([10.0, 20.0, 30.0, 40.0, 50.0]))
print(competing.predict([10.0, 20.0, 30.0, 40.0, 50.0]))
```

Native pandas output:

``` text
10.0    0.200000
20.0    0.300000
30.0    0.400000
40.0    0.566667
50.0    0.733333
Name: CIF_1, dtype: float64

10.0    0.000000
20.0    0.100000
30.0    0.100000
40.0    0.100000
50.0    0.266667
Name: CIF_2, dtype: float64
```


`tidycmprsk` returns outcome-specific rows in a tibble.

``` r
library(dplyr)
library(tidycmprsk)

data <- tibble(
  times = c(24.1, 9.7, 49.9, 18.6, 34.8, 14.2, 39.2, 46.0, 31.5, 4.3),
  reals = factor(
    c(1, 1, 1, 1, 0, 2, 1, 2, 0, 1),
    levels = c(0, 1, 2),
    labels = c("censored", "primary_event", "competing_event")
  )
)

fit <- cuminc(Surv(times, reals) ~ 1, data)
tidy(fit) |>
  select(outcome, time, estimate) |>
  print(n = 20)
```

Native tibble output:

``` text
# A tibble: 20 × 3
   outcome           time estimate
   <chr>             <dbl>    <dbl>
 1 primary_event       4.3    0.100
 2 primary_event       9.7    0.200
 3 primary_event      14.2    0.200
 4 primary_event      18.6    0.300
 5 primary_event      24.1    0.400
 6 primary_event      31.5    0.400
 7 primary_event      34.8    0.400
 8 primary_event      39.2    0.567
 9 primary_event      46.0    0.567
10 primary_event      49.9    0.733
11 competing_event     4.3    0
12 competing_event     9.7    0
13 competing_event    14.2    0.100
14 competing_event    18.6    0.100
15 competing_event    24.1    0.100
16 competing_event    31.5    0.100
17 competing_event    34.8    0.100
18 competing_event    39.2    0.100
19 competing_event    46.0    0.267
20 competing_event    49.9    0.267
```

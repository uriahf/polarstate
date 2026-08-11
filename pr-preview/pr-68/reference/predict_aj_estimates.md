## predict_aj_estimates()


Predict state-occupancy probabilities at `fixed_time_horizons`.


Usage

``` python
predict_aj_estimates(
    event_table,
    fixed_time_horizons,
    full_event_table=False,
)
```


## Parameters


`event_table: pl.DataFrame`  
The event table created by [prepare_event_table()](prepare_event_table.md#polarstate.prepare_event_table).

`fixed_time_horizons: pl.Series`  
Times at which to obtain the state-occupancy probabilities.


## Returns


`pl.DataFrame`  
DataFrame with `fixed_time_horizons` and the estimated probabilities for states 0, 1 and 2.

## predict_aj_estimates()


Evaluate state-occupancy probabilities at fixed time horizons.


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
Event table returned by [prepare_event_table](prepare_event_table.md#polarstate.prepare_event_table).

`fixed_time_horizons: pl.Series`  
Numeric horizons to evaluate. Horizons may be unsorted and need not coincide with observed event times.

`full_event_table: bool = ``False`  
If `True`, append estimates at every observed event time. Use the `estimate_origin` column to distinguish those rows from requested horizons.


## Returns


`pl.DataFrame`  
Columns are `times`, state-occupancy probabilities for states 0, 1, and 2, and `estimate_origin`. State 0 means no absorbing event by the horizon. States 1 and 2 are the event of interest and optional competing event. The probabilities sum to one up to floating-point precision.


## Raises


`TypeError`  
If inputs are not the documented Polars types, horizons are not numeric, or full_event_table is not boolean.

`ValueError`  
If required event-table columns are missing, either input is empty, or horizons contain null, non-finite, or negative values.


## Notes

Estimates are step functions. At a horizon between observed times, the latest estimate at or before that horizon is returned. Horizons before the first observed time receive probabilities `(1, 0, 0)`.


## Examples

``` python
>>> import polars as pl
>>> from polarstate import prepare_event_table, predict_aj_estimates
>>> observations = pl.DataFrame(
...     {"times": [2.0, 4.0, 5.0], "reals": [1, 0, 1]}
... )
>>> table = prepare_event_table(observations)
>>> predict_aj_estimates(table, pl.Series([1.0, 3.0])).shape
```

(2, 5)


## See Also

[prepare_event_table()](prepare_event_table.md#polarstate.prepare_event_table)  
Build the event table from subject-level data.

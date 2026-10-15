## predict_aj_estimates()


Evaluate state-occupancy probabilities at fixed time horizons or scheduled steps.


Usage

``` python
predict_aj_estimates(
    event_table, fixed_time_horizons=None, full_event_table=False
)
```


## Parameters


`event_table: pl.DataFrame or TransitionTableSteps`  
Event table returned by [prepare_event_table](prepare_event_table.md#polarstate.prepare_event_table) or transition table returned by `prepare_transition_table_steps`.

`fixed_time_horizons: pl.Series = None`  
Numeric horizons to evaluate for time-based event tables, or integer step horizons for step transition tables. If omitted for step transition tables, occupancy for all observed steps is returned.

`full_event_table: bool = ``False`  
If `True`, append estimates at every observed event time. Use the `estimate_origin` column to distinguish those rows from requested horizons. Only supported for continuous event tables.


## Returns


`pl.DataFrame`  
For competing risk event tables: columns are `times`, state-occupancy probabilities for states 0, 1, and 2, and `estimate_origin`. For discrete step transition tables: columns are `step`, `state`, and `state_occupancy_probability`.


## Raises


`TypeError`  
If inputs are not the documented Polars/polarstate types, horizons are not of expected numeric/integer dtypes, or full_event_table is not boolean.

`ValueError`  
If required event-table columns are missing, either input is empty, or horizons contain null, non-finite, invalid, or unobserved step values.

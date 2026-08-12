## prepare_event_table()


Build an inspectable Aalen-Johansen event table.


Usage

``` python
prepare_event_table(times_and_reals)
```


## Parameters


`times_and_reals: pl.DataFrame`  
Subject-level observations containing a numeric `times` column and an integer `reals` column. Outcome code `0` means censored, `1` means the event of interest, and optional `2` means a competing event. A simple event/censoring analysis uses only `0` and `1`.


## Returns


`pl.DataFrame`  
One row per unique observed time. The result includes outcome counts, the risk set, cause-specific event increments, conditional and overall survival, transition increments, and cumulative state-occupancy probabilities for states 1 and 2.


## Notes

Input rows need not be sorted. The function assumes the required columns are present, non-null, correctly typed, and use only supported outcome codes. Validate or recode data upstream.


## Examples

``` python
>>> import polars as pl
>>> from polarstate import prepare_event_table
>>> observations = pl.DataFrame(
...     {"times": [2.0, 4.0, 5.0], "reals": [1, 0, 1]}
... )
>>> prepare_event_table(observations).select(
...     "times", "at_risk", "count_1", "overall_survival"
... ).shape
```

(3, 4)


## See Also

[predict_aj_estimates()](predict_aj_estimates.md#polarstate.predict_aj_estimates)  
Evaluate the event table at fixed horizons.

## prepare_event_table()


Compute an inspectable Aalen-Johansen event table.


Usage

``` python
prepare_event_table(times_and_reals)
```


## Parameters


`times_and_reals: pl.DataFrame`  
Subject-level observations containing a numeric `times` column and a numeric `reals` column containing discrete event codes. Outcome code `0` means follow-up ended without an observed event (right-censoring), `1` means the event of interest, and optional `2` means a competing event. A simple event/censoring analysis uses only `0` and `1`.


## Returns


`pl.DataFrame`  
One row per unique observed time. The result includes outcome counts, the risk set, cause-specific event increments, conditional and overall survival, transition increments, and cumulative state-occupancy probabilities for states 1 and 2.


## Raises


`TypeError`  
If the input is not a Polars DataFrame, times is not numeric, or reals is not numeric.

`ValueError`  
If required columns are missing, values are null, times are non-finite or negative, or outcome codes are outside {0, 1, 2}.


## Notes

Empty input is supported for backward compatibility. Integer-valued floating codes (0.0, 1.0, 2.0) are also accepted; other codes are rejected.

Input rows need not be sorted. Extra columns are ignored. Duplicate times are allowed and are aggregated into a single event-table row.


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

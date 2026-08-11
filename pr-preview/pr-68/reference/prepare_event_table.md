## prepare_event_table()


Generate the full event table from raw `times` and `reals` data.


Usage

``` python
prepare_event_table(times_and_reals)
```


## Parameters


`times_and_reals: pl.DataFrame`  
A Polars DataFrame containing at least `times` and `reals` columns.


## Returns


`pl.DataFrame`  
The event table with all intermediate columns computed.

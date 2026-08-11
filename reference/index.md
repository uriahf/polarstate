# API Reference


The complete public API for preparing time-to-event tables and predicting state-occupancy probabilities, with or without competing events.


## Estimation


Build an inspectable event table from time-to-event observations.


[prepare_event_table()](prepare_event_table.md#polarstate.prepare_event_table)  
Generate the full event table from raw `times` and `reals` data.


## Prediction


Evaluate state-occupancy probabilities at fixed time horizons.


[predict_aj_estimates()](predict_aj_estimates.md#polarstate.predict_aj_estimates)  
Predict state-occupancy probabilities at `fixed_time_horizons`.

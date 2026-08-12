# How the Estimator Works

The Aalen-Johansen estimator generalizes Kaplan-Meier reasoning to one or more absorbing outcomes. With only outcome codes `0` and `1`, `polarstate` handles an ordinary single-event or binary event/censoring analysis. Adding outcome code `2` introduces a competing event. Each calculation stage remains available as a column in the event table.


*\[Rich HTML output -- view on the documentation site\]*


Codes `0` and `1` are sufficient for the single-event case. Code `2` adds a second absorbing state; it is not required by the API.


# Calculation flow

1.  Observations are grouped by time and counted by outcome.
2.  The risk set is computed at each observed time.
3.  Cause-specific hazards are calculated for the event of interest and the optional competing event.
4.  Conditional survival is accumulated into overall survival.
5.  The previous survival probability weights each cause-specific hazard.
6.  Weighted transitions are cumulatively summed into state-occupancy probabilities.

For a requested time horizon, the prediction function performs a backward as-of join: it returns the latest estimate available at or before that horizon. Horizons before the first observed event receive probability 1 for state 0 and 0 for the absorbing states.

> **Note: Input contract**
>
> The estimator expects outcomes encoded as `0` or `1`, with `2` used only when a competing event is present. Validate and recode upstream data before preparing the event table.

Continue with [Understanding the output](understanding-output.md) for a column-by-column interpretation.

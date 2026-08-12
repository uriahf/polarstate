# How the Estimator Works

The Aalen-Johansen estimator generalizes Kaplan-Meier reasoning to one or more absorbing outcomes. With only outcome codes `0` and `1`, `polarstate` handles an ordinary single-event or binary event/censoring analysis. Adding outcome code `2` introduces a competing event. Each calculation stage remains available as a column in the event table.


*\[Rich HTML output -- view on the documentation site\]*


Codes `0` and `1` are sufficient for the single-event case. Code `2` adds a second absorbing state; it is not required by the API.


# The estimator

For event type (j), the estimated state-occupancy probability is

<span id="eq-aalen-johansen"> \widehat F_j(t) = \sum\_{u \le t} \widehat S(u-) \frac{dN_j(u)}{Y(u)}. \tag{1}</span>

Here, (Y(u)) is the risk set immediately before time (u), (dN_j(u)) is the number of type-(j) events at that time, and (widehat S(u-)) is the probability of remaining in state 0 immediately beforehand.


# Calculation flow

The columns returned by [prepare_event_table()](../reference/prepare_event_table.md#polarstate.prepare_event_table) expose each component of [Equation 1](#eq-aalen-johansen):

1.  Observations are grouped by time and counted by outcome.
2.  `at_risk` supplies (Y(u)).
3.  `csh_1` and `csh_2` supply (dN_j(u)/Y(u)).
4.  `previous_overall_survival` supplies (widehat S(u-)).
5.  `trainsition_probabilities_to_*_at_times` contains each weighted increment.
6.  `state_occupancy_probability_*_at_times` cumulatively sums those increments.

For a requested time horizon, the prediction function performs a backward as-of join: it returns the latest estimate available at or before that horizon. Horizons before the first observed event receive probability 1 for state 0 and 0 for the absorbing states.

> **Note: Input contract**
>
> The estimator expects outcomes encoded as `0` or `1`, with `2` used only when a competing event is present. Validate and recode upstream data before preparing the event table.

Continue with [Understanding the output](understanding-output.md) for a column-by-column interpretation, or open the [worked example](worked-example.md) to inspect Equation [Equation 1](#eq-aalen-johansen) numerically.

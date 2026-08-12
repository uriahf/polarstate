# Understanding the Output

`polarstate` deliberately returns inspectable Polars DataFrames. This page explains what the columns represent and which ones usually matter downstream.


# Event-table columns

| Column | Interpretation |
|----|----|
| `times` | Unique observed time |
| `count_0` | Censored observations at that time |
| `count_1` | Events of interest at that time |
| `count_2` | Competing events at that time; zero in a single-event analysis |
| `events_at_times` | Total observations ending follow-up at that time |
| `at_risk` | Observations still under observation immediately before that time |
| `csh_1`, `csh_2` | Cause-specific event increments: event count divided by risk set |
| `conditional_survival` | Probability of avoiding either event at that time, conditional on being at risk |
| `overall_survival` | Product of conditional-survival increments through that time |
| `previous_overall_survival` | Survival immediately before the current time |
| `trainsition_probabilities_to_*_at_times` | Probability mass entering each absorbing state at that time |
| `state_occupancy_probability_*_at_times` | Cumulative probability of occupying each absorbing state |

The `trainsition` spelling is retained for compatibility with the current public output schema.


# Prediction columns

| Column | Interpretation |
|----|----|
| `times` | Requested horizon or observed event time |
| `state_occupancy_probability_0` | Probability of remaining event-free |
| `state_occupancy_probability_1` | Probability of the event of interest |
| `state_occupancy_probability_2` | Probability of the competing event |
| `estimate_origin` | Whether the row came from a requested horizon or the full event table |

At each horizon,

 P_0(t) + P_1(t) + P_2(t) = 1. 

For a simple event/censoring analysis, `P_2(t)=0`, so `P_0(t)` behaves like the Kaplan-Meier survival estimate and `P_1(t)=1-P_0(t)`.


# Why estimates differ from raw proportions

A raw event proportion ignores how long each observation was followed. Aalen-Johansen estimates use the risk set at each event time, so censoring removes an observation from later risk sets without treating it as an event.


# Horizons between observed times

Predictions are step functions. For a requested horizon between two observed times, [predict_aj_estimates](../reference/predict_aj_estimates.md#polarstate.predict_aj_estimates) returns the latest estimate at or before that horizon. Before the first event, state 0 is one and the absorbing-state probabilities are zero.

> **Important: Important**
>
> These estimates describe the observed cohort under the usual independent censoring assumptions. They do not by themselves adjust for confounding or establish causal effects.

"""Expose the main public API for :mod:`polarstate`."""

from .aj import (
    aalen_johansen,
    add_at_risk_column,
    add_cause_specific_hazards_columns,
    add_events_at_times_column,
    add_overall_survival_column,
    add_previous_overal_survival_column,
    add_state_occupancy_probabilities_at_times_columns,
    add_transition_probabilities_at_times_columns,
    create_sorted_times_and_reals_data,
    group_reals_by_times,
    prepare_event_table,
)

# ``predict_aj_estimates`` is provided as a convenience alias so users can simply
# ``from polarstate import predict_aj_estimates``.
predict_aj_estimates = aalen_johansen

__all__ = [
    "predict_aj_estimates",
    "aalen_johansen",
    "create_sorted_times_and_reals_data",
    "group_reals_by_times",
    "add_events_at_times_column",
    "add_at_risk_column",
    "add_cause_specific_hazards_columns",
    "add_overall_survival_column",
    "add_previous_overal_survival_column",
    "add_transition_probabilities_at_times_columns",
    "add_state_occupancy_probabilities_at_times_columns",
    "prepare_event_table",
]


def main() -> None:
    print("Hello from polarstate!")

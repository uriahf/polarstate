import polars as pl
from polarstate.aj import (
    create_sorted_times_and_reals_data,
    group_reals_by_times,
    add_events_at_times_column,
    add_at_risk_column,
    add_cause_specific_hazards_columns,
    add_overall_survival_column,
    add_previous_overal_survival_column,
    add_transition_probabilities_at_times_columns,
    add_state_occupancy_probabilities_at_times_columns,
    prepare_event_table,
)
from polarstate.predict import predict_aj_estimates
from polars.testing import assert_frame_equal


def test_create_sorted_times_and_reals_data() -> None:
    # Test data
    times = pl.Series([5, 3, 1, 4, 2])
    reals = pl.Series([0, 1, 2, 1, 0])

    # Expected output
    expected_output = pl.DataFrame({"times": [1, 2, 3, 4, 5], "reals": [2, 0, 1, 1, 0]})

    result = create_sorted_times_and_reals_data(times, reals)

    assert_frame_equal(result, expected_output)


def test_add_events_at_times_column() -> None:
    times_and_counts = pl.DataFrame(
        {
            "times": [1, 2, 3],
            "count_0": [1, 1, 0],
            "count_1": [1, 1, 0],
            "count_2": [0, 1, 2],
        }
    )

    result = add_events_at_times_column(times_and_counts)

    expected_output = pl.DataFrame(
        {
            "times": [1, 2, 3],
            "count_0": [1, 1, 0],
            "count_1": [1, 1, 0],
            "count_2": [0, 1, 2],
            "events_at_times": [2, 3, 2],
        }
    )

    assert_frame_equal(result, expected_output)


def test_group_reals_by_times():
    sorted_times_and_reals = pl.DataFrame(
        {"times": [1, 1, 2, 2, 2, 3, 3], "reals": [0, 1, 0, 1, 2, 2, 2]}
    )

    result = group_reals_by_times(sorted_times_and_reals)

    expected_output = pl.DataFrame(
        {
            "times": [1, 2, 3],
            "count_0": [1, 1, 0],
            "count_1": [1, 1, 0],
            "count_2": [0, 1, 2],
        }
    )

    assert_frame_equal(result, expected_output)


def test_add_at_risk_column():
    events_data = pl.DataFrame(
        {
            "times": [1, 2, 3],
            "count_0": [1, 1, 0],
            "count_1": [1, 1, 0],
            "count_2": [0, 1, 2],
            "events_at_times": [2, 3, 2],
        }
    )

    result = add_at_risk_column(events_data)

    expected_output = pl.DataFrame(
        {
            "times": [1, 2, 3],
            "count_0": [1, 1, 0],
            "count_1": [1, 1, 0],
            "count_2": [0, 1, 2],
            "events_at_times": [2, 3, 2],
            "at_risk": [7, 5, 2],
        }
    )

    assert_frame_equal(result, expected_output)


def test_add_cause_specific_hazards_columns():
    events_data = pl.DataFrame(
        {
            "times": [1, 2, 3],
            "count_0": [1, 1, 0],
            "count_1": [1, 1, 0],
            "count_2": [0, 1, 2],
            "events_at_times": [2, 3, 2],
            "at_risk": [7, 5, 2],
        }
    )

    result = add_cause_specific_hazards_columns(events_data)

    expected_output = pl.DataFrame(
        {
            "times": [1, 2, 3],
            "count_0": [1, 1, 0],
            "count_1": [1, 1, 0],
            "count_2": [0, 1, 2],
            "events_at_times": [2, 3, 2],
            "at_risk": [7, 5, 2],
            "csh_1": [1 / 7, 1 / 5, 0.0],
            "csh_2": [0.0, 1 / 5, 2 / 2],
            "conditional_survival": [6 / 7, 3 / 5, 0.0],
        }
    )

    assert_frame_equal(result, expected_output)


def test_add_overall_survival_column():
    events_data = pl.DataFrame(
        {
            "times": [1, 2, 3],
            "count_0": [1, 1, 0],
            "count_1": [1, 1, 0],
            "count_2": [0, 1, 2],
            "events_at_times": [2, 3, 2],
            "at_risk": [7, 5, 2],
            "csh_1": [1 / 7, 1 / 5, 0.0],
            "csh_2": [0.0, 1 / 5, 1.0],
            "conditional_survival": [6 / 7, 3 / 5, 0.0],
        }
    )

    result = add_overall_survival_column(events_data)

    expected_output = events_data.with_columns(
        [
            pl.Series(
                "overall_survival",
                [(6 / 7), (6 / 7) * (3 / 5), (6 / 7) * (3 / 5) * 0.0],
            )
        ]
    )

    assert_frame_equal(result, expected_output)


def test_add_previous_overal_survival_column() -> None:
    events_data = pl.DataFrame(
        {
            "times": [1, 2, 3],
            "count_0": [1, 1, 0],
            "count_1": [1, 1, 0],
            "count_2": [0, 1, 2],
            "events_at_times": [2, 3, 2],
            "at_risk": [7, 5, 2],
            "csh_1": [1 / 7, 1 / 5, 0.0],
            "csh_2": [0.0, 1 / 5, 1.0],
            "conditional_survival": [6 / 7, 3 / 5, 0.0],
            "overall_survival": [
                (6 / 7),
                (6 / 7) * (3 / 5),
                (6 / 7) * (3 / 5) * 0.0,
            ],
        }
    )

    result = add_previous_overal_survival_column(events_data)

    expected_output = pl.DataFrame(
        {
            "times": [1, 2, 3],
            "count_0": [1, 1, 0],
            "count_1": [1, 1, 0],
            "count_2": [0, 1, 2],
            "events_at_times": [2, 3, 2],
            "at_risk": [7, 5, 2],
            "csh_1": [1 / 7, 1 / 5, 0.0],
            "csh_2": [0.0, 1 / 5, 1.0],
            "conditional_survival": [6 / 7, 3 / 5, 0.0],
            "overall_survival": [
                (6 / 7),
                (6 / 7) * (3 / 5),
                (6 / 7) * (3 / 5) * 0.0,
            ],
            "previous_overall_survival": [1.0, (6 / 7), (6 / 7) * (3 / 5)],
        }
    )

    assert_frame_equal(result, expected_output)


def test_add_transition_probabilities_at_times_columns() -> None:
    events_data = pl.DataFrame(
        {
            "times": [1, 2, 3],
            "count_0": [1, 1, 0],
            "count_1": [1, 1, 0],
            "count_2": [0, 1, 2],
            "events_at_times": [2, 3, 2],
            "at_risk": [7, 5, 2],
            "csh_1": [1 / 7, 1 / 5, 0.0],
            "csh_2": [0.0, 1 / 5, 1.0],
            "conditional_survival": [6 / 7, 3 / 5, 0.0],
            "overall_survival": [
                (6 / 7),
                (6 / 7) * (3 / 5),
                (6 / 7) * (3 / 5) * 0.0,
            ],
            "previous_overall_survival": [1.0, (6 / 7), (6 / 7) * (3 / 5)],
        }
    )

    result = add_transition_probabilities_at_times_columns(events_data)

    expected_output = pl.DataFrame(
        {
            "times": [1, 2, 3],
            "count_0": [1, 1, 0],
            "count_1": [1, 1, 0],
            "count_2": [0, 1, 2],
            "events_at_times": [2, 3, 2],
            "at_risk": [7, 5, 2],
            "csh_1": [1 / 7, 1 / 5, 0.0],
            "csh_2": [0.0, 1 / 5, 1.0],
            "conditional_survival": [6 / 7, 3 / 5, 0.0],
            "overall_survival": [
                (6 / 7),
                (6 / 7) * (3 / 5),
                (6 / 7) * (3 / 5) * 0.0,
            ],
            "previous_overall_survival": [1.0, (6 / 7), (6 / 7) * (3 / 5)],
            "trainsition_probabilities_to_1_at_times": [
                1.0 * (1 / 7),
                (6 / 7) * (1 / 5),
                (6 / 7) * (3 / 5) * 0.0,
            ],
            "trainsition_probabilities_to_2_at_times": [
                1.0 * 0.0,
                (6 / 7) * (1 / 5),
                (6 / 7) * (3 / 5) * 1.0,
            ],
        }
    )

    assert_frame_equal(result, expected_output)


def test_add_state_occupancy_probabilities_at_times_columns() -> None:
    events_data = pl.DataFrame(
        {
            "times": [1, 2, 3],
            "count_0": [1, 1, 0],
            "count_1": [1, 1, 0],
            "count_2": [0, 1, 2],
            "events_at_times": [2, 3, 2],
            "at_risk": [7, 5, 2],
            "csh_1": [1 / 7, 1 / 5, 0.0],
            "csh_2": [0.0, 1 / 5, 1.0],
            "conditional_survival": [6 / 7, 3 / 5, 0.0],
            "overall_survival": [
                (6 / 7),
                (6 / 7) * (3 / 5),
                (6 / 7) * (3 / 5) * 0.0,
            ],
            "previous_overall_survival": [1.0, (6 / 7), (6 / 7) * (3 / 5)],
            "trainsition_probabilities_to_1_at_times": [
                (1.0 * (1 / 7)),
                ((6 / 7) * (1 / 5)),
                ((6 / 7) * (3 / 5) * 0.0),
            ],
            "trainsition_probabilities_to_2_at_times": [
                (1.0 * 0.0),
                ((6 / 7) * (1 / 5)),
                ((6 / 7) * (3 / 5) * 1.0),
            ],
        }
    )

    result = add_state_occupancy_probabilities_at_times_columns(events_data)

    expected_output = pl.DataFrame(
        {
            "times": [1, 2, 3],
            "count_0": [1, 1, 0],
            "count_1": [1, 1, 0],
            "count_2": [0, 1, 2],
            "events_at_times": [2, 3, 2],
            "at_risk": [7, 5, 2],
            "csh_1": [1 / 7, 1 / 5, 0.0],
            "csh_2": [0.0, 1 / 5, 1.0],
            "conditional_survival": [6 / 7, 3 / 5, 0.0],
            "overall_survival": [
                (6 / 7),
                (6 / 7) * (3 / 5),
                (6 / 7) * (3 / 5) * 0.0,
            ],
            "previous_overall_survival": [1.0, (6 / 7), (6 / 7) * (3 / 5)],
            "trainsition_probabilities_to_1_at_times": [
                (1.0 * (1 / 7)),
                ((6 / 7) * (1 / 5)),
                ((6 / 7) * (3 / 5) * 0.0),
            ],
            "trainsition_probabilities_to_2_at_times": [
                (1.0 * 0.0),
                ((6 / 7) * (1 / 5)),
                ((6 / 7) * (3 / 5) * 1.0),
            ],
            "state_occupancy_probability_1_at_times": [
                (1.0 * (1 / 7)),
                (1.0 * (1 / 7)) + ((6 / 7) * (1 / 5)),
                (1.0 * (1 / 7)) + ((6 / 7) * (1 / 5)) + ((6 / 7) * (3 / 5) * 0.0),
            ],
            "state_occupancy_probability_2_at_times": [
                (1.0 * 0.0),
                (1.0 * 0.0) + ((6 / 7) * (1 / 5)),
                (1.0 * 0.0) + ((6 / 7) * (1 / 5)) + ((6 / 7) * (3 / 5) * 1.0),
            ],
        }
    )

    assert_frame_equal(result, expected_output)


def test_prepare_event_table() -> None:
    times_and_reals = pl.DataFrame(
        {"times": [1, 1, 2, 2, 2, 3, 3], "reals": [0, 1, 0, 1, 2, 2, 2]}
    )

    result = prepare_event_table(times_and_reals)

    expected_output = pl.DataFrame(
        {
            "times": [1, 2, 3],
            "count_0": [1, 1, 0],
            "count_1": [1, 1, 0],
            "count_2": [0, 1, 2],
            "events_at_times": [2, 3, 2],
            "at_risk": [7, 5, 2],
            "csh_1": [1 / 7, 1 / 5, 0.0],
            "csh_2": [0.0, 1 / 5, 1.0],
            "conditional_survival": [6 / 7, 3 / 5, 0.0],
            "overall_survival": [
                (6 / 7),
                (6 / 7) * (3 / 5),
                (6 / 7) * (3 / 5) * 0.0,
            ],
            "previous_overall_survival": [1.0, (6 / 7), (6 / 7) * (3 / 5)],
            "trainsition_probabilities_to_1_at_times": [
                (1.0 * (1 / 7)),
                ((6 / 7) * (1 / 5)),
                ((6 / 7) * (3 / 5) * 0.0),
            ],
            "trainsition_probabilities_to_2_at_times": [
                (1.0 * 0.0),
                ((6 / 7) * (1 / 5)),
                ((6 / 7) * (3 / 5) * 1.0),
            ],
            "state_occupancy_probability_1_at_times": [
                (1.0 * (1 / 7)),
                (1.0 * (1 / 7)) + ((6 / 7) * (1 / 5)),
                (1.0 * (1 / 7)) + ((6 / 7) * (1 / 5)) + ((6 / 7) * (3 / 5) * 0.0),
            ],
            "state_occupancy_probability_2_at_times": [
                (1.0 * 0.0),
                (1.0 * 0.0) + ((6 / 7) * (1 / 5)),
                (1.0 * 0.0) + ((6 / 7) * (1 / 5)) + ((6 / 7) * (3 / 5) * 1.0),
            ],
        }
    )

    assert_frame_equal(result, expected_output)


def test_predict_aj_estimates() -> None:
    times_and_reals = pl.DataFrame(
        {"times": [1, 1, 2, 2, 2, 3, 3], "reals": [0, 1, 0, 1, 2, 2, 2]}
    )

    event_table = prepare_event_table(times_and_reals)

    fixed_time_horizons = pl.Series([1, 3, 5])

    result = predict_aj_estimates(event_table, fixed_time_horizons)

    expected_output = pl.DataFrame(
        {
            "times": [1, 3, 5],
            "state_occupancy_probability_0": [
                (6 / 7),
                0.0,
                0.0,
            ],
            "state_occupancy_probability_1": [
                (1 / 7),
                (11 / 35),
                (11 / 35),
            ],
            "state_occupancy_probability_2": [
                0.0,
                (24 / 35),
                (24 / 35),
            ],
            "estimate_origin": [
                "fixed_time_horizons",
                "fixed_time_horizons",
                "fixed_time_horizons",
            ],
        }
    ).with_columns(
        pl.col("estimate_origin").cast(pl.Enum(["fixed_time_horizons", "event_table"]))
    )

    assert_frame_equal(result, expected_output)

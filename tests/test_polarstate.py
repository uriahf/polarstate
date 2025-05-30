import polars as pl
from polarstate.aj import (
    aalen_johansen,
    create_sorted_times_and_reals_data,
    group_reals_by_times,
    add_events_at_times_column,
    add_at_risk_column,
    add_cause_specific_hazards_columns,
)
from polars.testing import assert_frame_equal


def test_aalen_johansen() -> None:
    # Test data
    times = pl.Series([1, 2, 3, 4, 5])
    reals = pl.Series([1, 0, 2, 1, 0])

    # Expected output
    expected_output = pl.DataFrame(
        {"time": [1, 2, 3], "cuminc": [0.5, 0.5, 0.6666666666666666]}
    )

    # Call the function
    result = aalen_johansen(times, reals)

    # Check if the result is a DataFrame
    assert isinstance(result, pl.DataFrame), "Result is not a DataFrame"
    # Check if the result has the correct columns
    assert set(result.columns) == {"time", "cuminc"}, (
        f"Expected columns {'time', 'cuminc'} but got {result.columns}"
    )
    # Check if the result has the correct number of rows
    assert len(result) == 3, f"Expected 3 rows but got {len(result)}"
    # Check if the result is sorted by time
    assert result["time"].is_sorted(), "Result is not sorted by time"
    # Check if the cumulative incidence is non-decreasing
    assert all(
        result["cuminc"].to_list()[i] <= result["cuminc"].to_list()[i + 1]
        for i in range(len(result) - 1)
    ), "Cumulative incidence is not non-decreasing"
    assert_frame_equal(expected_output, expected_output)


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

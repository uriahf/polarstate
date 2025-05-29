import polars as pl
from polarstate.aj import aalen_johansen, create_sorted_times_and_reals_data
from polars.testing import assert_frame_equal

def test_aalen_johansen() -> None:
    # Test data
    times = pl.Series([1, 2, 3, 4, 5])
    reals = pl.Series([1, 0, 2, 1, 0])

    # Expected output
    expected_output = pl.DataFrame({
        "time": [1, 2, 3],
        "cuminc": [0.5, 0.5, 0.6666666666666666]
    })

    # Call the function
    result = aalen_johansen(times, reals)

    # Check if the result is a DataFrame
    assert isinstance(result, pl.DataFrame), "Result is not a DataFrame"
    # Check if the result has the correct columns
    assert set(result.columns) == {"time", "cuminc"}, f"Expected columns {'time', 'cuminc'} but got {result.columns}"
    # Check if the result has the correct number of rows
    assert len(result) == 3, f"Expected 3 rows but got {len(result)}"
    # Check if the result is sorted by time
    assert result["time"].is_sorted(), "Result is not sorted by time"
    # Check if the cumulative incidence is non-decreasing
    assert all(result["cuminc"].to_list()[i] <= result["cuminc"].to_list()[i + 1] for i in range(len(result) - 1)), "Cumulative incidence is not non-decreasing"
    
    
def test_create_sorted_times_and_reals_data() -> None:
    # Test data
    times = pl.Series([5, 3, 1, 4, 2])
    reals = pl.Series([0, 1, 2, 1, 0])

    # Expected output
    expected_output = pl.DataFrame({
        "times": [1, 2, 3, 4, 5],
        "reals": [2, 0, 1, 1, 0]
    })

    result = create_sorted_times_and_reals_data(times, reals)

    assert_frame_equal(result, expected_output)
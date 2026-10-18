"""Regression tests for valid legacy event-table inputs."""

import polars as pl
import pytest

from polarstate import prepare_event_table


def test_empty_event_table_is_supported():
    empty = pl.DataFrame(schema={"times": pl.Float64, "reals": pl.Int64})
    result = prepare_event_table(empty)
    assert result.is_empty()


def test_integer_valued_float_event_codes_match_integers():
    times = [1.0, 2.0, 3.0, 4.0]
    integer_data = pl.DataFrame({"times": times, "reals": [0, 1, 2, 1]})
    float_data = pl.DataFrame({"times": times, "reals": [0.0, 1.0, 2.0, 1.0]})
    integer_result = prepare_event_table(integer_data)
    float_result = prepare_event_table(float_data)
    for col in (
        "times",
        "count_0",
        "count_1",
        "count_2",
        "at_risk",
        "state_occupancy_probability_1_at_times",
        "state_occupancy_probability_2_at_times",
    ):
        assert float_result[col].to_list() == pytest.approx(
            integer_result[col].to_list()
        )


@pytest.mark.parametrize("invalid", [0.5, -1.0, 3.0, float("nan"), float("inf")])
def test_invalid_float_event_codes_are_rejected(invalid):
    data = pl.DataFrame({"times": [1.0], "reals": [invalid]})
    with pytest.raises(ValueError, match="only 0, 1, and 2"):
        prepare_event_table(data)

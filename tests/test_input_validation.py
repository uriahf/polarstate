import polars as pl
import pytest

from polarstate import prepare_event_table, predict_aj_estimates


@pytest.mark.parametrize(
    ("data", "error", "message"),
    [
        (pl.DataFrame(), ValueError, "missing required"),
        (pl.DataFrame({"times": [-1.0], "reals": [1]}), ValueError, "non-negative"),
        (pl.DataFrame({"times": [1.0], "reals": [3]}), ValueError, "only 0, 1, and 2"),
    ],
)
def test_prepare_event_table_rejects_invalid_input(data, error, message) -> None:
    with pytest.raises(error, match=message):
        prepare_event_table(data)


def test_transition_columns_have_backward_compatible_aliases() -> None:
    result = prepare_event_table(
        pl.DataFrame({"times": [1.0, 2.0], "reals": [1, 0]})
    )
    for state in (1, 2):
        correct = f"transition_probabilities_to_{state}_at_times"
        legacy = f"trainsition_probabilities_to_{state}_at_times"
        assert result[correct].equals(result[legacy])


def test_predict_rejects_invalid_horizons() -> None:
    table = prepare_event_table(
        pl.DataFrame({"times": [1.0, 2.0], "reals": [1, 0]})
    )
    with pytest.raises(ValueError, match="non-negative"):
        predict_aj_estimates(table, pl.Series([-1.0]))

def test_empty_event_table_remains_usable() -> None:
    table = prepare_event_table(
        pl.DataFrame(schema={"times": pl.Float64, "reals": pl.Int64})
    )
    assert table.is_empty()
    result = predict_aj_estimates(table, pl.Series([1.0, 2.0]))
    assert result["state_occupancy_probability_0"].to_list() == [1.0, 1.0]
    assert result["state_occupancy_probability_1"].to_list() == [0.0, 0.0]
    assert result["state_occupancy_probability_2"].to_list() == [0.0, 0.0]


def test_float_integer_codes_have_identical_aj_results() -> None:
    ints = prepare_event_table(
        pl.DataFrame({"times": [1.0, 2.0, 3.0], "reals": [0, 1, 2]})
    )
    floats = prepare_event_table(
        pl.DataFrame({"times": [1.0, 2.0, 3.0], "reals": [0.0, 1.0, 2.0]})
    )
    assert ints.select(["count_0", "count_1", "count_2", "overall_survival"]).equals(
        floats.select(["count_0", "count_1", "count_2", "overall_survival"])
    )


def test_fractional_code_is_rejected() -> None:
    with pytest.raises(ValueError, match="integer-valued"):
        prepare_event_table(
            pl.DataFrame({"times": [1.0, 2.0], "reals": [0.0, 1.5]})
        )

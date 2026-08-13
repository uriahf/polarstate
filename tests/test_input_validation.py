import polars as pl
import pytest

from polarstate import prepare_event_table, predict_aj_estimates


@pytest.mark.parametrize(
    ("data", "error", "message"),
    [
        (pl.DataFrame(), ValueError, "missing required"),
        (pl.DataFrame({"times": [], "reals": []}, schema={"times": pl.Float64, "reals": pl.Int64}), ValueError, "at least one"),
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

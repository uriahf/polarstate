import polars as pl
import pytest
from polars.testing import assert_frame_equal

from polarstate import (
    TransitionTableSteps,
    predict_aj_estimates,
    prepare_transition_table_steps,
)


def test_exact_complete_panel_fixture() -> None:
    """Test exact complete panel fixture with 4 subjects across steps 0, 1, 2.

    Fixture Data:
    ID 1: 1 -> 2 -> 3
    ID 2: 1 -> 1 -> 2
    ID 3: 2 -> 3 -> 2
    ID 4: 3 -> 2 -> 1

    Initial occupancy pi(0) = (1/2, 1/4, 1/4)
    Q1 = [[1/2, 1/2, 0], [0, 0, 1], [0, 1, 0]]
    Q2 = [[0, 1, 0], [1/2, 0, 1/2], [0, 1, 0]]
    pi(1) = (1/4, 1/2, 1/4)
    pi(2) = (1/4, 1/2, 1/4)
    """
    observations = pl.DataFrame(
        {
            "id": [1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4],
            "step": [0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2],
            "reals": [1, 2, 3, 1, 1, 2, 2, 3, 2, 3, 2, 1],
        }
    )

    table = prepare_transition_table_steps(observations, state_order=[1, 2, 3])
    assert table.steps == (0, 1, 2)
    assert table.states == (1, 2, 3)

    # Initial occupancy assertion
    expected_pi0 = (1 / 2, 1 / 4, 1 / 4)
    for p_est, p_exp in zip(table.initial, expected_pi0):
        assert abs(p_est - p_exp) < 1e-12

    # Assert transition counts
    trans_s1 = table.transitions.filter(pl.col("step") == 1)
    c_11 = trans_s1.filter((pl.col("from_state") == 1) & (pl.col("to_state") == 1))[
        "count"
    ].item()
    c_12 = trans_s1.filter((pl.col("from_state") == 1) & (pl.col("to_state") == 2))[
        "count"
    ].item()
    c_23 = trans_s1.filter((pl.col("from_state") == 2) & (pl.col("to_state") == 3))[
        "count"
    ].item()
    c_32 = trans_s1.filter((pl.col("from_state") == 3) & (pl.col("to_state") == 2))[
        "count"
    ].item()
    assert (c_11, c_12, c_23, c_32) == (1, 1, 1, 1)

    # Predictions
    estimates = predict_aj_estimates(table)

    def get_pi(step: int) -> tuple[float, float, float]:
        df_s = estimates.filter(pl.col("step") == step)
        return tuple(df_s["state_occupancy_probability"].to_list())

    pi0 = get_pi(0)
    pi1 = get_pi(1)
    pi2 = get_pi(2)

    assert all(abs(a - b) < 1e-12 for a, b in zip(pi0, (1 / 2, 1 / 4, 1 / 4)))
    assert all(abs(a - b) < 1e-12 for a, b in zip(pi1, (1 / 4, 1 / 2, 1 / 4)))
    assert all(abs(a - b) < 1e-12 for a, b in zip(pi2, (1 / 4, 1 / 2, 1 / 4)))

    # Empirical state proportion equality on complete uncensored panel
    N = 4
    for step_val, expected_vector in [
        (0, (1 / 2, 1 / 4, 1 / 4)),
        (1, (1 / 4, 1 / 2, 1 / 4)),
        (2, (1 / 4, 1 / 2, 1 / 4)),
    ]:
        emp = observations.filter(pl.col("step") == step_val).group_by("reals").len()
        for st, exp_p in zip([1, 2, 3], expected_vector):
            row = emp.filter(pl.col("reals") == st)
            emp_p = row["len"].item() / N if row.height > 0 else 0.0
            assert abs(emp_p - exp_p) < 1e-12


def test_exact_estimable_terminal_dropout_fixture() -> None:
    """Test exact estimable terminal dropout fixture with 6 subjects A-F across steps 0, 1, 2.

    Fixture Data:
    ID A: 1 -> 1 -> 2
    ID B: 1 -> 2 -> 2
    ID C: 1 -> 1 -> (dropped out)
    ID D: 2 -> 2 -> 1
    ID E: 2 -> 1 -> (dropped out)
    ID F: 2 -> (dropped out)

    Available-pair transition matrices:
    pi0 = (1/2, 1/2)
    Q1 = [[2/3, 1/3], [1/2, 1/2]]
    pi1 = (7/12, 5/12)
    Q2 = [[0, 1], [1/2, 1/2]]
    pi2 = (5/24, 19/24)
    """
    observations = pl.DataFrame(
        {
            "id": [
                "A",
                "A",
                "A",
                "B",
                "B",
                "B",
                "C",
                "C",
                "D",
                "D",
                "D",
                "E",
                "E",
                "F",
            ],
            "step": [0, 1, 2, 0, 1, 2, 0, 1, 0, 1, 2, 0, 1, 0],
            "reals": [1, 1, 2, 1, 2, 2, 1, 1, 2, 2, 1, 2, 1, 2],
        }
    )

    table = prepare_transition_table_steps(observations, state_order=[1, 2])
    assert table.steps == (0, 1, 2)
    assert table.states == (1, 2)

    for p_est, p_exp in zip(table.initial, (1 / 2, 1 / 2)):
        assert abs(p_est - p_exp) < 1e-12

    estimates = predict_aj_estimates(table)

    def get_pi(step: int) -> tuple[float, float]:
        df_s = estimates.filter(pl.col("step") == step)
        return tuple(df_s["state_occupancy_probability"].to_list())

    pi0 = get_pi(0)
    pi1 = get_pi(1)
    pi2 = get_pi(2)

    assert all(abs(a - b) < 1e-12 for a, b in zip(pi0, (1 / 2, 1 / 2)))
    assert all(abs(a - b) < 1e-12 for a, b in zip(pi1, (7 / 12, 5 / 12)))
    assert all(abs(a - b) < 1e-12 for a, b in zip(pi2, (5 / 24, 19 / 24)))

    # Compare with raw unadjusted observed-case proportions at step 2
    raw_step2_state1 = 1 / 3
    assert abs(pi2[0] - raw_step2_state1) > 0.05


def test_out_of_order_horizons_and_output_sorting() -> None:
    """Test passing unsorted horizons (e.g. [2, 0]) returns step-ascending sorted output."""
    observations = pl.DataFrame(
        {
            "id": [1, 1, 1, 2, 2, 2],
            "step": [0, 1, 2, 0, 1, 2],
            "reals": [1, 2, 2, 1, 1, 2],
        }
    )

    table = prepare_transition_table_steps(observations, state_order=[1, 2])

    estimates = predict_aj_estimates(table, pl.Series([2, 0]))

    # Step column must be sorted ascending [0, 0, 2, 2]
    assert estimates["step"].to_list() == [0, 0, 2, 2]
    assert estimates.filter(pl.col("step") == 0)["state"].to_list() == [1, 2]
    assert estimates.filter(pl.col("step") == 2)["state"].to_list() == [1, 2]


def test_preserved_dtypes_int32_and_categorical() -> None:
    """Test that Int32 step dtypes and Categorical state dtypes are strictly preserved."""
    observations = pl.DataFrame(
        {
            "id": [1, 1, 2, 2],
            "step": pl.Series([0, 1, 0, 1], dtype=pl.Int32),
            "reals": pl.Series(["A", "B", "A", "A"], dtype=pl.Categorical),
        }
    )

    table = prepare_transition_table_steps(observations)
    estimates = predict_aj_estimates(table)

    assert estimates.schema["step"] == pl.Int32
    assert estimates.schema["state"] == pl.Categorical


def test_nonlexical_state_ordering() -> None:
    """Test that custom non-lexical state_order is strictly preserved in row output order."""
    observations = pl.DataFrame(
        {
            "id": [1, 1, 2, 2],
            "step": [1, 2, 1, 2],
            "reals": ["high", "low", "medium", "low"],
        }
    )

    custom_order = ["medium", "high", "low"]
    table = prepare_transition_table_steps(observations, state_order=custom_order)
    estimates = predict_aj_estimates(table)

    step1_states = estimates.filter(pl.col("step") == 1)["state"].to_list()
    step2_states = estimates.filter(pl.col("step") == 2)["state"].to_list()

    assert step1_states == custom_order
    assert step2_states == custom_order


def test_complete_panel_and_empirical_occupancy() -> None:
    """Test uncensored complete panel against direct empirical proportions."""
    observations = pl.DataFrame(
        {
            "id": [1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4],
            "step": [1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3],
            "reals": ["A", "B", "C", "A", "A", "B", "B", "B", "B", "A", "C", "C"],
        }
    )

    table = prepare_transition_table_steps(observations)
    assert isinstance(table, TransitionTableSteps)
    assert table.steps == (1, 2, 3)
    assert table.states == ("A", "B", "C")
    assert table.initial == (0.75, 0.25, 0.0)

    estimates = predict_aj_estimates(table)

    N = observations["id"].n_unique()
    empirical = (
        observations.group_by("step", "reals")
        .len(name="count")
        .with_columns((pl.col("count") / N).alias("expected_prob"))
    )

    for row in estimates.iter_rows(named=True):
        step = row["step"]
        state = row["state"]
        prob = row["state_occupancy_probability"]

        emp_match = empirical.filter(
            (pl.col("step") == step) & (pl.col("reals") == state)
        )
        expected = emp_match["expected_prob"].item() if emp_match.height > 0 else 0.0
        assert pytest.approx(prob) == expected


def test_multiple_baseline_states() -> None:
    """Test panel where subjects start in different baseline states."""
    observations = pl.DataFrame(
        {
            "id": [1, 1, 2, 2, 3, 3],
            "step": [1, 2, 1, 2, 1, 2],
            "reals": [1, 2, 2, 3, 3, 1],
        }
    )

    table = prepare_transition_table_steps(observations)
    assert table.initial == (1 / 3, 1 / 3, 1 / 3)

    estimates = predict_aj_estimates(table)
    assert estimates.columns == ["step", "state", "state_occupancy_probability"]
    assert estimates.height == 6  # 2 steps * 3 states


def test_stable_and_reversible_transitions() -> None:
    """Test transitions that stay in state or transition back and forth."""
    observations = pl.DataFrame(
        {
            "id": [1, 1, 1, 2, 2, 2],
            "step": [1, 2, 3, 1, 2, 3],
            "reals": ["health", "sick", "health", "health", "health", "sick"],
        }
    )

    table = prepare_transition_table_steps(observations)
    estimates = predict_aj_estimates(table)

    s1_health = estimates.filter((pl.col("step") == 1) & (pl.col("state") == "health"))[
        "state_occupancy_probability"
    ].item()
    assert pytest.approx(s1_health) == 1.0

    s2_health = estimates.filter((pl.col("step") == 2) & (pl.col("state") == "health"))[
        "state_occupancy_probability"
    ].item()
    assert pytest.approx(s2_health) == 0.5

    s3_health = estimates.filter((pl.col("step") == 3) & (pl.col("state") == "health"))[
        "state_occupancy_probability"
    ].item()
    assert pytest.approx(s3_health) == 0.5


def test_nonadjacent_jumps_and_simultaneous_transitions() -> None:
    """Test direct jumps between non-adjacent ordinal states and multiple simultaneous transitions."""
    observations = pl.DataFrame(
        {
            "id": [1, 1, 2, 2, 3, 3],
            "step": [1, 2, 1, 2, 1, 2],
            "reals": [1, 3, 1, 3, 1, 2],
        }
    )

    table = prepare_transition_table_steps(observations)
    estimates = predict_aj_estimates(table)

    s2_state2 = estimates.filter((pl.col("step") == 2) & (pl.col("state") == 2))[
        "state_occupancy_probability"
    ].item()
    s2_state3 = estimates.filter((pl.col("step") == 2) & (pl.col("state") == 3))[
        "state_occupancy_probability"
    ].item()

    assert pytest.approx(s2_state2) == 1 / 3
    assert pytest.approx(s2_state3) == 2 / 3


def test_terminal_right_censoring_error() -> None:
    """Test panel with subject dropouts (right censoring) causing positive occupancy but zero transitions."""
    observations = pl.DataFrame(
        {
            "id": [1, 1, 1, 2, 2, 3],
            "step": [1, 2, 3, 1, 2, 1],
            "reals": [1, 1, 2, 1, 2, 2],
        }
    )

    table = prepare_transition_table_steps(observations)
    assert table.initial == (2 / 3, 1 / 3)

    with pytest.raises(ValueError, match="No observed transitions from state 2"):
        predict_aj_estimates(table)


def test_zero_risk_set_with_zero_occupancy() -> None:
    """Test state with 0 occupancy and 0 at-risk transitions handled gracefully via identity row."""
    observations = pl.DataFrame(
        {
            "id": [1, 1, 1, 2, 2, 2, 3, 3, 3],
            "step": [1, 2, 3, 1, 2, 3, 1, 2, 3],
            "reals": [1, 1, 1, 1, 1, 1, 3, 2, 2],
        }
    )

    table = prepare_transition_table_steps(observations, state_order=[1, 2, 3])
    estimates = predict_aj_estimates(table)

    s3_step3 = estimates.filter((pl.col("step") == 3) & (pl.col("state") == 3))[
        "state_occupancy_probability"
    ].item()
    assert pytest.approx(s3_step3) == 0.0


def test_intermittent_gaps_error() -> None:
    """Test that missing internal steps for a subject raise ValueError."""
    observations = pl.DataFrame(
        {
            "id": [1, 1, 2, 2],
            "step": [1, 3, 1, 2],
            "reals": [1, 2, 1, 2],
        }
    )

    with pytest.raises(
        ValueError, match="Intermittent missing steps are not supported"
    ):
        prepare_transition_table_steps(observations)


def test_duplicate_subject_step_error() -> None:
    """Test that duplicate (id, step) rows raise ValueError."""
    observations = pl.DataFrame(
        {
            "id": [1, 1, 1, 2],
            "step": [1, 1, 2, 1],
            "reals": [1, 2, 1, 2],
        }
    )

    with pytest.raises(ValueError, match="Duplicate subject-step observations"):
        prepare_transition_table_steps(observations)


def test_invalid_state_order_error() -> None:
    """Test error when state_order does not match observed states."""
    observations = pl.DataFrame(
        {
            "id": [1, 1],
            "step": [1, 2],
            "reals": ["A", "B"],
        }
    )

    with pytest.raises(
        ValueError, match="state_order must contain each observed state"
    ):
        prepare_transition_table_steps(observations, state_order=["A", "C"])


def test_all_step_and_selected_horizons() -> None:
    """Test default all-step predictions vs specific requested horizons."""
    observations = pl.DataFrame(
        {
            "id": [1, 1, 1, 2, 2, 2],
            "step": [10, 11, 12, 10, 11, 12],
            "reals": [1, 2, 2, 1, 1, 2],
        }
    )

    table = prepare_transition_table_steps(observations)
    assert table.steps == (10, 11, 12)

    all_est = predict_aj_estimates(table)
    assert sorted(all_est["step"].unique().to_list()) == [10, 11, 12]

    selected = predict_aj_estimates(table, pl.Series([10, 12]))
    assert sorted(selected["step"].unique().to_list()) == [10, 12]


def test_invalid_horizons() -> None:
    """Test errors for invalid horizons."""
    observations = pl.DataFrame(
        {
            "id": [1, 1, 2, 2],
            "step": [1, 2, 1, 2],
            "reals": [1, 2, 1, 2],
        }
    )
    table = prepare_transition_table_steps(observations)

    with pytest.raises(
        ValueError, match="fixed_time_horizons must contain observed integer steps"
    ):
        predict_aj_estimates(table, pl.Series([1, 99]))

    with pytest.raises(TypeError, match="integer dtype"):
        predict_aj_estimates(table, pl.Series([1.0, 2.0]))


def test_row_order_invariance() -> None:
    """Test that input row ordering does not affect transition table or predictions."""
    obs1 = pl.DataFrame(
        {
            "id": [1, 1, 2, 2, 3, 3],
            "step": [1, 2, 1, 2, 1, 2],
            "reals": ["A", "B", "A", "A", "B", "B"],
        }
    )

    obs2 = obs1.sample(fraction=1.0, shuffle=True, seed=42)

    t1 = prepare_transition_table_steps(obs1)
    t2 = prepare_transition_table_steps(obs2)

    assert t1.steps == t2.steps
    assert t1.states == t2.states
    assert t1.initial == t2.initial
    assert_frame_equal(t1.transitions, t2.transitions)

    p1 = predict_aj_estimates(t1)
    p2 = predict_aj_estimates(t2)
    assert_frame_equal(p1, p2)


def test_empty_and_invalid_inputs() -> None:
    """Test error handling for empty DataFrame, missing columns, non-integer steps, non-consecutive grid."""
    with pytest.raises(ValueError, match="observations DataFrame is empty"):
        prepare_transition_table_steps(pl.DataFrame())

    with pytest.raises(ValueError, match="Missing columns"):
        prepare_transition_table_steps(pl.DataFrame({"id": [1], "step": [1]}))

    with pytest.raises(TypeError, match="step_col must have an integer dtype"):
        prepare_transition_table_steps(
            pl.DataFrame({"id": [1], "step": [1.5], "reals": [1]})
        )

    with pytest.raises(ValueError, match="observed step grid must be consecutive"):
        prepare_transition_table_steps(
            pl.DataFrame(
                {
                    "id": [1, 1, 2, 2],
                    "step": [1, 3, 1, 3],
                    "reals": [1, 2, 1, 2],
                }
            )
        )

    with pytest.raises(
        ValueError, match="Every subject must have a baseline observation"
    ):
        prepare_transition_table_steps(
            pl.DataFrame(
                {
                    "id": [1, 1, 2],
                    "step": [1, 2, 2],
                    "reals": [1, 2, 2],
                }
            )
        )


def test_competing_risk_backward_compatibility() -> None:
    """Ensure existing competing risk preparation and prediction workflows remain unchanged."""
    obs = pl.DataFrame(
        {
            "times": [1.0, 2.0, 3.0],
            "reals": [1, 0, 2],
        }
    )

    from polarstate import prepare_event_table

    evt = prepare_event_table(obs)
    res = predict_aj_estimates(evt, pl.Series([1.0, 2.5]))
    assert "times" in res.columns
    assert "state_occupancy_probability_0" in res.columns
    assert res.height == 2

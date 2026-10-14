import polars as pl
import pytest
from polars.testing import assert_frame_equal

from polarstate import (
    TransitionTableSteps,
    predict_aj_estimates,
    prepare_transition_table_steps,
)


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

    # Direct Polars empirical group_by calculation
    N = observations["id"].n_unique()
    empirical = (
        observations.group_by("step", "reals")
        .len(name="count")
        .with_columns((pl.col("count") / N).alias("expected_prob"))
    )

    # Check each step and state in estimates
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

    # Step 1: id 1=health, id 2=health -> P(health)=1.0, P(sick)=0.0
    s1_health = estimates.filter((pl.col("step") == 1) & (pl.col("state") == "health"))[
        "state_occupancy_probability"
    ].item()
    assert pytest.approx(s1_health) == 1.0

    # Step 2: id 1->sick, id 2->health -> P(health)=0.5, P(sick)=0.5
    s2_health = estimates.filter((pl.col("step") == 2) & (pl.col("state") == "health"))[
        "state_occupancy_probability"
    ].item()
    assert pytest.approx(s2_health) == 0.5

    # Step 3: id 1 sick->health, id 2 health->sick -> P(health)=0.5, P(sick)=0.5
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

    # Step 2: from state 1 at step 1 (3 subjects), 2 jumped to state 3, 1 moved to state 2
    s2_state2 = estimates.filter((pl.col("step") == 2) & (pl.col("state") == 2))[
        "state_occupancy_probability"
    ].item()
    s2_state3 = estimates.filter((pl.col("step") == 2) & (pl.col("state") == 3))[
        "state_occupancy_probability"
    ].item()

    assert pytest.approx(s2_state2) == 1 / 3
    assert pytest.approx(s2_state3) == 2 / 3


def test_terminal_right_censoring() -> None:
    """Test panel with subject dropouts (right censoring) after step 1 and step 2."""
    observations = pl.DataFrame(
        {
            "id": [1, 1, 1, 2, 2, 3],
            "step": [1, 2, 3, 1, 2, 1],
            "reals": [1, 1, 2, 1, 2, 2],
        }
    )

    table = prepare_transition_table_steps(observations)
    # At step 1: id 1=1, id 2=1, id 3=2. Baseline: P(1)=2/3, P(2)=1/3.
    assert table.initial == (2 / 3, 1 / 3)

    # Transitions step 1 -> step 2:
    # id 1 (state 1) -> state 1
    # id 2 (state 1) -> state 2
    # id 3 (state 2) -> dropped out (censored)
    # At risk at step 2 from state 1: Y_1(2) = 2. Transitions: 1->1 (1), 1->2 (1).
    # At risk at step 2 from state 2: Y_2(2) = 0. State 2 occupancy at step 1 is 1/3 > 0.
    # Therefore, no observed transitions from state 2 at step 2 despite positive occupancy!
    with pytest.raises(ValueError, match="No observed transitions from state 2"):
        predict_aj_estimates(table)


def test_zero_risk_set_with_zero_occupancy() -> None:
    """Test state with 0 occupancy and 0 at-risk transitions handled gracefully via identity row."""
    # State 3 is observed at step 1 (id 3 = state 3), but moves to state 2 at step 2.
    # At step 2, no subjects remain in state 3.
    # At step 3, no subjects are in state 3 at step 2, so Y_3(3) = 0.
    # Since occupancy of state 3 at step 2 is 0.0, zero risk set is handled via identity row.
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
            "step": [1, 3, 1, 2],  # Subject 1 skips step 2
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
            "step": [1, 1, 2, 1],  # Subject 1 has duplicate step 1
            "reals": [1, 2, 1, 2],
        }
    )

    with pytest.raises(ValueError, match="Duplicate subject-step observations"):
        prepare_transition_table_steps(observations)


def test_explicit_nonlexical_state_ordering() -> None:
    """Test custom state ordering."""
    observations = pl.DataFrame(
        {
            "id": [1, 1, 2, 2],
            "step": [1, 2, 1, 2],
            "reals": ["high", "low", "medium", "low"],
        }
    )

    custom_order = ["low", "medium", "high"]
    table = prepare_transition_table_steps(observations, state_order=custom_order)
    assert table.states == ("low", "medium", "high")

    estimates = predict_aj_estimates(table)
    # Check that rows for each step are output in custom_order
    step1_states = estimates.filter(pl.col("step") == 1)["state"].to_list()
    assert step1_states == custom_order


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

    # Default horizons = all steps
    all_est = predict_aj_estimates(table)
    assert sorted(all_est["step"].unique().to_list()) == [10, 11, 12]

    # Selected horizons
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

    # Horizon not in observed step grid
    with pytest.raises(
        ValueError, match="fixed_time_horizons must contain observed integer steps"
    ):
        predict_aj_estimates(table, pl.Series([1, 99]))

    # Non-integer horizons
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
    # Empty DataFrame
    with pytest.raises(ValueError, match="observations DataFrame is empty"):
        prepare_transition_table_steps(pl.DataFrame())

    # Missing column
    with pytest.raises(ValueError, match="Missing columns"):
        prepare_transition_table_steps(pl.DataFrame({"id": [1], "step": [1]}))

    # Float step
    with pytest.raises(TypeError, match="step_col must have an integer dtype"):
        prepare_transition_table_steps(
            pl.DataFrame({"id": [1], "step": [1.5], "reals": [1]})
        )

    # Non-consecutive grid
    with pytest.raises(ValueError, match="observed step grid must be consecutive"):
        prepare_transition_table_steps(
            pl.DataFrame(
                {
                    "id": [1, 1, 2, 2],
                    "step": [1, 3, 1, 3],  # Overall grid is [1, 3] missing step 2
                    "reals": [1, 2, 1, 2],
                }
            )
        )

    # Missing baseline observation for a subject
    with pytest.raises(
        ValueError, match="Every subject must have a baseline observation"
    ):
        prepare_transition_table_steps(
            pl.DataFrame(
                {
                    "id": [1, 1, 2],
                    "step": [1, 2, 2],  # Subject 2 missing step 1
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

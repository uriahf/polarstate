"""Permanent R survival parity regression tests for discrete-step Aalen-Johansen estimation.

Provenance & Metadata:
- R Version: R version 4.3.3 (2024-02-29)
- Package Version: survival 3.5-8
- Conversion Convention: Discrete-step counting process start-stop (tstart, tstop, event)
  where events occur at interval ends (tstop). Unchanged states are assigned (censor) status.
  Initial state distribution p0 reflects the full original baseline cohort at min(step).
- Audit Commit SHA: 8c2cacd05f7d02b27958ee16c95efaeae7956096
- Target Absolute Tolerance: 1e-10

Reference outputs were independently derived from R survival::survfit() via
tests/generate_r_references.R and not generated using polarstate.
"""

from typing import Any

import polars as pl
import pytest

from polarstate import predict_aj_estimates, prepare_transition_table_steps

# ------------------------------------------------------------------------------
# Fixtures & Verified R Reference Results
# ------------------------------------------------------------------------------

PARITY_FIXTURES: list[dict[str, Any]] = [
    {
        "name": "1. Homogeneous Baseline Complete Panel",
        "observations": pl.DataFrame(
            {
                "id": [1, 1, 1, 2, 2, 2, 3, 3, 3],
                "step": [0, 1, 2, 0, 1, 2, 0, 1, 2],
                "reals": [1, 1, 2, 1, 2, 2, 1, 1, 1],
            }
        ),
        "states": [1, 2],
        "is_complete_panel": True,
        # Reference results from R survival::survfit()
        "ref_initial_p0": [1.0, 0.0],
        "ref_steps": [1, 2],
        "ref_n_risk": [
            [3, 0],
            [2, 1],
        ],
        "ref_n_event": [
            [0, 1],
            [0, 1],
        ],
        "ref_pstate": [
            [2 / 3, 1 / 3],
            [1 / 3, 2 / 3],
        ],
    },
    {
        "name": "2. Heterogeneous Baseline Complete Panel",
        "observations": pl.DataFrame(
            {
                "id": [1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4],
                "step": [0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2],
                "reals": [1, 2, 3, 1, 1, 2, 2, 3, 2, 3, 2, 1],
            }
        ),
        "states": [1, 2, 3],
        "is_complete_panel": True,
        "ref_initial_p0": [0.5, 0.25, 0.25],
        "ref_steps": [1, 2],
        "ref_n_risk": [
            [2, 1, 1],
            [1, 2, 1],
        ],
        "ref_n_event": [
            [0, 2, 1],
            [1, 2, 1],
        ],
        "ref_pstate": [
            [0.25, 0.5, 0.25],
            [0.25, 0.5, 0.25],
        ],
    },
    {
        "name": "3. Complex Transitions & Initial States",
        "observations": pl.DataFrame(
            {
                "id": [
                    1,
                    1,
                    1,
                    2,
                    2,
                    2,
                    3,
                    3,
                    3,
                    4,
                    4,
                    4,
                    5,
                    5,
                    5,
                ],
                "step": [0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2],
                "reals": [
                    "A",
                    "B",
                    "C",
                    "B",
                    "B",
                    "A",
                    "C",
                    "A",
                    "B",
                    "A",
                    "A",
                    "C",
                    "B",
                    "C",
                    "C",
                ],
            }
        ),
        "states": ["A", "B", "C"],
        "is_complete_panel": True,
        "ref_initial_p0": [0.4, 0.4, 0.2],
        "ref_steps": [1, 2],
        "ref_n_risk": [
            [2, 2, 1],
            [2, 2, 1],
        ],
        "ref_n_event": [
            [1, 1, 1],
            [1, 1, 2],
        ],
        "ref_pstate": [
            [0.4, 0.4, 0.2],
            [0.2, 0.2, 0.6],
        ],
    },
    {
        "name": "4. Reversible Transitions & Same-state Persistence",
        "observations": pl.DataFrame(
            {
                "id": [1, 1, 1, 2, 2, 2],
                "step": [1, 2, 3, 1, 2, 3],
                "reals": ["H", "S", "H", "H", "H", "S"],
            }
        ),
        "states": ["H", "S"],
        "is_complete_panel": True,
        "ref_initial_p0": [1.0, 0.0],
        "ref_steps": [2, 3],
        "ref_n_risk": [
            [2, 0],
            [1, 1],
        ],
        "ref_n_event": [
            [0, 1],
            [1, 1],
        ],
        "ref_pstate": [
            [0.5, 0.5],
            [0.5, 0.5],
        ],
    },
    {
        "name": "5. Nonadjacent Jumps & Simultaneous Transitions",
        "observations": pl.DataFrame(
            {
                "id": [1, 1, 2, 2, 3, 3],
                "step": [1, 2, 1, 2, 1, 2],
                "reals": [1, 3, 1, 3, 1, 2],
            }
        ),
        "states": [1, 2, 3],
        "is_complete_panel": True,
        "ref_initial_p0": [1.0, 0.0, 0.0],
        "ref_steps": [2],
        "ref_n_risk": [
            [3, 0, 0],
        ],
        "ref_n_event": [
            [0, 1, 2],
        ],
        "ref_pstate": [
            [0.0, 1 / 3, 2 / 3],
        ],
    },
    {
        "name": "7. Independent Terminal Dropout",
        "observations": pl.DataFrame(
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
        ),
        "states": [1, 2],
        "is_complete_panel": False,
        "ref_initial_p0": [0.5, 0.5],
        "ref_steps": [1, 2],
        "ref_n_risk": [
            [3, 2],
            [1, 2],
        ],
        "ref_n_event": [
            [1, 1],
            [1, 1],
        ],
        "ref_pstate": [
            [7 / 12, 5 / 12],
            [5 / 24, 19 / 24],
        ],
    },
]


@pytest.mark.parametrize("fixture", PARITY_FIXTURES, ids=lambda f: f["name"])
def test_r_survival_parity_and_invariants(fixture: dict[str, Any]) -> None:
    """Verify exact parity with R survival::survfit() reference outputs and check invariants."""
    obs: pl.DataFrame = fixture["observations"]
    states: list[Any] = fixture["states"]
    ref_initial_p0: list[float] = fixture["ref_initial_p0"]
    ref_steps: list[int] = fixture["ref_steps"]
    ref_n_risk: list[list[int]] = fixture["ref_n_risk"]
    ref_n_event: list[list[int]] = fixture["ref_n_event"]
    ref_pstate: list[list[float]] = fixture["ref_pstate"]

    table = prepare_transition_table_steps(obs, state_order=states)

    # 1. Initial distribution parity
    for p_est, p_ref in zip(table.initial, ref_initial_p0):
        assert abs(p_est - p_ref) < 1e-10

    # 2. Origin-state denominators parity
    # In TransitionTableSteps, transitions contains (step, from_state, to_state, count)
    # Summing counts over to_state for each step and from_state gives state-specific risk denominators.
    trans_df = table.transitions
    for s_idx, step_val in enumerate(ref_steps):
        for st_idx, st in enumerate(states):
            ref_risk = ref_n_risk[s_idx][st_idx]
            sub = trans_df.filter(
                (pl.col("step") == step_val) & (pl.col("from_state") == st)
            )
            calc_risk = sub["count"].sum() if sub.height > 0 else 0
            assert calc_risk == ref_risk, (
                f"Denominator mismatch at step {step_val}, state {st}: "
                f"got {calc_risk}, expected {ref_risk}"
            )

    # 3. Off-diagonal transition counts parity against R survfitms$n.event
    # R's n.event matrix reports counts of transitions into destination state (excluding censor events)
    for s_idx, step_val in enumerate(ref_steps):
        for to_idx, to_st in enumerate(states):
            ref_events = ref_n_event[s_idx][to_idx]
            # Off-diagonal transitions to `to_st` at `step_val`
            sub = trans_df.filter(
                (pl.col("step") == step_val)
                & (pl.col("from_state") != to_st)
                & (pl.col("to_state") == to_st)
            )
            calc_events = sub["count"].sum() if sub.height > 0 else 0
            assert calc_events == ref_events, (
                f"Off-diagonal transition count mismatch at step {step_val}, destination {to_st}: "
                f"got {calc_events}, expected {ref_events}"
            )

    # 4. AJ State Occupancy parity against R survfit pstate (at 1e-10 tolerance)
    estimates = predict_aj_estimates(table)

    for s_idx, step_val in enumerate(ref_steps):
        for st_idx, st in enumerate(states):
            ref_p = ref_pstate[s_idx][st_idx]
            est_row = estimates.filter(
                (pl.col("step") == step_val) & (pl.col("state") == st)
            )
            assert est_row.height == 1, (
                f"Missing prediction row for step {step_val}, state {st}"
            )
            est_p = est_row["state_occupancy_probability"].item()

            assert abs(est_p - ref_p) < 1e-10, (
                f"AJ occupancy discrepancy at step {step_val}, state {st}: "
                f"got {est_p:.17g}, expected {ref_p:.17g}, diff {abs(est_p - ref_p):.2e}"
            )

    # 5. Selected horizons predictions match all-step predictions exactly
    horizon_step = ref_steps[-1]
    horizon_est = predict_aj_estimates(table, pl.Series([horizon_step]))
    all_step_subset = estimates.filter(pl.col("step") == horizon_step)

    for st in states:
        p_all = all_step_subset.filter(pl.col("state") == st)[
            "state_occupancy_probability"
        ].item()
        p_hor = horizon_est.filter(pl.col("state") == st)[
            "state_occupancy_probability"
        ].item()
        assert p_all == p_hor

    # 6. Empirical state proportion check for complete uncensored panels
    if fixture["is_complete_panel"]:
        n_unique_ids = obs["id"].n_unique()
        for step_val in table.steps:
            obs_step = obs.filter(pl.col("step") == step_val)
            for st in states:
                emp_count = obs_step.filter(pl.col("reals") == st).height
                emp_prop = emp_count / n_unique_ids

                est_p = estimates.filter(
                    (pl.col("step") == step_val) & (pl.col("state") == st)
                )["state_occupancy_probability"].item()

                assert abs(est_p - emp_prop) < 1e-10, (
                    f"Empirical proportion discrepancy for complete panel at step {step_val}, state {st}: "
                    f"estimated {est_p}, empirical {emp_prop}"
                )

    # 7. Numerical invariants
    for step_val in table.steps:
        df_step = estimates.filter(pl.col("step") == step_val)

        # Invariant A: All probabilities are finite and in [0, 1]
        probs = df_step["state_occupancy_probability"].to_list()
        for p in probs:
            assert abs(p) == abs(p)  # Not NaN / finite check
            assert -1e-12 <= p <= 1.0 + 1e-12

        # Invariant B: Probabilities sum to 1 at each step
        step_sum = sum(probs)
        assert abs(step_sum - 1.0) < 1e-10

    # Invariant C: Result is invariant to input-row ordering
    shuffled_obs = obs.sample(fraction=1.0, shuffle=True, seed=12345)
    table_shuffled = prepare_transition_table_steps(shuffled_obs, state_order=states)
    estimates_shuffled = predict_aj_estimates(table_shuffled)

    for row in estimates.iter_rows(named=True):
        st = row["state"]
        sp = row["step"]
        prob1 = row["state_occupancy_probability"]

        prob2 = estimates_shuffled.filter(
            (pl.col("step") == sp) & (pl.col("state") == st)
        )["state_occupancy_probability"].item()

        assert prob1 == prob2


@pytest.mark.parametrize(
    ("fixture_name", "obs"),
    [
        (
            "9. Minimal Baseline-only Subject Diagnostic",
            pl.DataFrame(
                {
                    "id": [1, 1, 2],
                    "step": [0, 1, 0],
                    "reals": [1, 2, 2],
                }
            ),
        ),
        (
            "Terminal Dropout With Positive Occupancy and Zero Transitions",
            pl.DataFrame(
                {
                    "id": [1, 1, 1, 2, 2, 3],
                    "step": [1, 2, 3, 1, 2, 1],
                    "reals": [1, 1, 2, 1, 2, 2],
                }
            ),
        ),
    ],
)
def test_zero_available_next_step_observations_error(
    fixture_name: str, obs: pl.DataFrame
) -> None:
    """Verify zero-available-transition scenarios raise expected ValueError.

    Note on intentional difference from R survival::survfit():
    In polarstate, if a state has positive occupancy at step k but zero observations
    at step k+1 (e.g., all subjects in that state censor/drop out), predict_aj_estimates
    raises ValueError('No observed transitions from state ...') rather than carrying forward
    or synthesizing transition probabilities.
    """
    table = prepare_transition_table_steps(obs)
    with pytest.raises(ValueError, match="No observed transitions from state 2"):
        predict_aj_estimates(table)

"""Discrete-step multistate Aalen–Johansen estimation."""

from typing import Any, NamedTuple, Sequence
import polars as pl


class TransitionTableSteps(NamedTuple):
    """Container for discrete-step transition table components.

    Attributes
    ----------
    steps : tuple
        Observed discrete integer steps.
    states : tuple
        Ordered states represented in the panel.
    initial : tuple
        Initial state occupancy probabilities at baseline.
    transitions : pl.DataFrame
        Aggregated transition counts between consecutive steps.
    step_dtype : pl.DataType, optional
        Data type of the original step column.
    state_dtype : pl.DataType, optional
        Data type of the original state column.
    """

    steps: tuple
    states: tuple
    initial: tuple
    transitions: pl.DataFrame
    step_dtype: Any = pl.Int64
    state_dtype: Any = pl.String


def prepare_transition_table_steps(
    observations: pl.DataFrame,
    *,
    id_col: str = "id",
    step_col: str = "step",
    state_col: str = "reals",
    state_order: Sequence | None = None,
) -> TransitionTableSteps:
    """Prepare discrete-step transition table and initial state occupancy from a long-format panel.

    Parameters
    ----------
    observations : pl.DataFrame
        Long-format panel data with one row per observed subject-step-state.
    id_col : str, default "id"
        Column name for subject identifier.
    step_col : str, default "step"
        Column name for discrete integer scheduled steps.
    state_col : str, default "reals"
        Column name for subject state at each step.
    state_order : sequence, optional
        Explicit ordering for categorical/ordinal states. Must contain every observed state exactly once.

    Returns
    -------
    TransitionTableSteps
        Named tuple containing observed steps grid, ordered states, baseline occupancy probabilities,
        aggregated transition counts, and original column data types.

    Raises
    ------
    TypeError
        If input is not a Polars DataFrame or step_col is not integer-valued.
    ValueError
        If required columns are missing, DataFrame is empty, contains nulls, duplicate subject-steps,
        non-consecutive step grid, missing baseline for any subject, invalid state_order, or intermittent gaps.

    Notes
    -----
    Statistical Estimator:
    For transitions occurring between step k-1 and step k, the at-risk denominator Y_i(k) is defined
    as the number of subjects observed in state i at step k-1 who remain under observation at step k.
    Subjects dropping out (right-censored) after step k-1 are excluded from interval k's denominator.
    This defines an available-pair discrete-step transition estimator. It assumes that censoring/dropout
    between steps is non-informative with respect to state transition probabilities conditional on the
    observed state at step k-1.
    """
    if not isinstance(observations, pl.DataFrame):
        raise TypeError("observations must be a Polars DataFrame")

    if observations.is_empty():
        raise ValueError("observations DataFrame is empty")

    required = {id_col, step_col, state_col}
    missing = required - set(observations.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")

    step_dt = observations.schema[step_col]
    state_dt = observations.schema[state_col]

    df = observations.select(id_col, step_col, state_col)

    if df.null_count().sum_horizontal().item() > 0:
        raise ValueError("Input columns contain null values")

    if not df.schema[step_col].is_integer():
        raise TypeError("step_col must have an integer dtype")

    if df.group_by(id_col, step_col).len().filter(pl.col("len") > 1).height > 0:
        raise ValueError("Duplicate subject-step observations detected")

    df = df.sort(id_col, step_col)

    steps = sorted(df[step_col].unique().to_list())

    if steps != list(range(steps[0], steps[-1] + 1)):
        raise ValueError("The observed step grid must be consecutive")

    baseline_step = steps[0]
    baseline = df.filter(pl.col(step_col) == baseline_step)

    n_subjects = df[id_col].n_unique()
    if baseline.height != n_subjects:
        raise ValueError(
            f"Every subject must have a baseline observation at step {baseline_step}"
        )

    observed_states = df[state_col].unique().to_list()

    if state_order is None:
        try:
            states = tuple(sorted(observed_states))
        except TypeError:
            states = tuple(
                sorted(
                    observed_states,
                    key=lambda x: (type(x).__name__, str(x)),
                )
            )
    else:
        states = tuple(state_order)
        if len(set(states)) != len(states) or set(states) != set(observed_states):
            raise ValueError(
                "state_order must contain each observed state exactly once"
            )

    baseline_counts = dict(baseline.group_by(state_col).len().iter_rows())

    initial = tuple(baseline_counts.get(state, 0) / baseline.height for state in states)

    df_shifted = df.with_columns(
        pl.col(step_col).shift().over(id_col).alias("_previous_step"),
        pl.col(state_col).shift().over(id_col).alias("_previous_state"),
    )

    pairs = df_shifted.filter(pl.col("_previous_step").is_not_null())

    if pairs.filter(pl.col(step_col) != pl.col("_previous_step") + 1).height > 0:
        raise ValueError("Intermittent missing steps are not supported")

    transitions = (
        pairs.group_by(
            step_col,
            "_previous_state",
            state_col,
        )
        .len(name="count")
        .rename(
            {
                step_col: "step",
                "_previous_state": "from_state",
                state_col: "to_state",
            }
        )
        .sort("step", "from_state", "to_state")
    )

    return TransitionTableSteps(
        steps=tuple(steps),
        states=states,
        initial=initial,
        transitions=transitions,
        step_dtype=step_dt,
        state_dtype=state_dt,
    )


def _predict_transition_steps(
    table: TransitionTableSteps,
    horizons: pl.Series | None = None,
) -> pl.DataFrame:
    """Return occupancy probabilities from discrete AJ transition matrix products."""
    steps = table.steps
    states = table.states
    n = len(states)

    if horizons is None:
        requested = sorted(steps)
    else:
        if not isinstance(horizons, pl.Series):
            raise TypeError("fixed_time_horizons must be a Polars Series")
        if not horizons.dtype.is_integer():
            raise TypeError("fixed_time_horizons must have integer dtype")
        if horizons.is_empty():
            raise ValueError("fixed_time_horizons must not be empty")
        if horizons.null_count() > 0:
            raise ValueError("fixed_time_horizons must not contain null values")

        requested_list = horizons.to_list()
        step_set = set(steps)
        if any(s not in step_set for s in requested_list):
            raise ValueError("fixed_time_horizons must contain observed integer steps")
        requested = sorted(set(requested_list))

    index = {state: i for i, state in enumerate(states)}

    counts_by_step: dict[Any, list[tuple[int, int, int]]] = {}
    for step, source, target, count in table.transitions.iter_rows():
        counts_by_step.setdefault(step, []).append(
            (index[source], index[target], count)
        )

    occupancy = list(table.initial)
    estimates = {steps[0]: occupancy.copy()}

    for step in steps[1:]:
        matrix = [[0.0] * n for _ in range(n)]
        risk = [0] * n

        for i, j, count in counts_by_step.get(step, []):
            matrix[i][j] += count
            risk[i] += count

        for i in range(n):
            if risk[i] > 0:
                matrix[i] = [val / risk[i] for val in matrix[i]]
            elif occupancy[i] > 1e-12:
                raise ValueError(
                    f"No observed transitions from state {states[i]!r} at step {step} "
                    f"despite positive occupancy ({occupancy[i]:.6e})"
                )
            else:
                matrix[i][i] = 1.0

        occupancy = [
            sum(occupancy[i] * matrix[i][j] for i in range(n)) for j in range(n)
        ]

        estimates[step] = occupancy.copy()

    records = [
        {
            "step": step,
            "state": state,
            "state_occupancy_probability": estimates[step][i],
            "_state_rank": i,
        }
        for step in requested
        for i, state in enumerate(states)
    ]

    res_df = pl.DataFrame(records)
    if not res_df.is_empty():
        step_dt = getattr(table, "step_dtype", pl.Int64)
        state_dt = getattr(table, "state_dtype", None)

        res_df = res_df.sort("step", "_state_rank").drop("_state_rank")
        res_df = res_df.with_columns(pl.col("step").cast(step_dt))
        if state_dt is not None:
            res_df = res_df.with_columns(pl.col("state").cast(state_dt))

    return res_df.select("step", "state", "state_occupancy_probability")

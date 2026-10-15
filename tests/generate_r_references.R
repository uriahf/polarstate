# ==============================================================================
# R Reference Generation Script for polarstate Discrete-Step AJ Parity Tests
# ==============================================================================
#
# Provenance & Audit Info:
#   R Version: R version 4.3.3 (2024-02-29)
#   survival Package Version: 3.5-8
#   Conversion Convention: Discrete step start-stop counting process where
#     events occur at interval ends (tstop). Unchanged states and baseline-only
#     subjects (censored at tstop = step + 0.5) are assigned (censor) status.
#     Initial distribution p0 is derived from the full baseline cohort (step_min).
#   Audit Commit SHA: 8c2cacd05f7d02b27958ee16c95efaeae7956096
#   Target Numerical Tolerance: 1e-10
#
# Note: This script reproduces reference values from R survival::survfit().
# It is an optional, standalone script used to independently generate reference
# values and does not depend on polarstate.
# ==============================================================================

suppressPackageStartupMessages({
  library(survival)
})

build_start_stop <- function(df, states) {
  df <- df[order(df$id, df$step), ]

  # Identify baseline-only subjects (subjects with only 1 observation)
  counts <- ave(df$step, df$id, FUN = length)
  base_only <- df[counts == 1, ]

  # Pairs for subjects with >= 2 observations
  df_multi <- df[counts > 1, ]
  if (nrow(df_multi) > 0) {
    df_multi$prev_step <- ave(df_multi$step, df_multi$id, FUN = function(x) c(NA, x[-length(x)]))
    df_multi$prev_state <- ave(as.character(df_multi$state), df_multi$id, FUN = function(x) c(NA, x[-length(x)]))
    pairs <- df_multi[!is.na(df_multi$prev_step), ]
    pairs$tstart <- pairs$prev_step
    pairs$tstop <- pairs$step
  } else {
    pairs <- data.frame()
  }

  state_levels <- as.character(states)
  all_levels <- c("(censor)", state_levels)

  if (nrow(pairs) > 0) {
    event_status <- ifelse(as.character(pairs$state) != pairs$prev_state,
                           as.character(pairs$state),
                           "(censor)")
    pairs$event <- factor(event_status, levels = all_levels)
    pairs$from <- factor(pairs$prev_state, levels = state_levels)
  }

  if (nrow(base_only) > 0) {
    base_pairs <- data.frame(
      id = base_only$id,
      step = base_only$step,
      state = base_only$state,
      prev_step = base_only$step,
      prev_state = as.character(base_only$state),
      tstart = base_only$step,
      tstop = base_only$step + 0.5, # Censored before next step
      event = factor("(censor)", levels = all_levels),
      from = factor(as.character(base_only$state), levels = state_levels)
    )
    if (nrow(pairs) > 0) {
      pairs <- rbind(pairs, base_pairs)
    } else {
      pairs <- base_pairs
    }
  }

  pairs <- pairs[order(pairs$id, pairs$tstart), ]
  return(pairs)
}

fmt_vector <- function(vec) {
  sprintf("[%s]", paste(sprintf("%.17g", vec), collapse = ", "))
}

fmt_matrix <- function(mat) {
  rows <- apply(mat, 1, function(r) sprintf("[%s]", paste(sprintf("%.17g", r), collapse = ", ")))
  sprintf("[%s]", paste(rows, collapse = ",\n    "))
}

run_survfit_fixture <- function(name, df, states) {
  df$state <- factor(df$state, levels = states)
  states_char <- as.character(states)

  min_step <- min(df$step)
  base_df <- df[df$step == min_step, ]
  p0_table <- table(factor(base_df$state, levels = states))
  p0_vec <- as.numeric(p0_table / sum(p0_table))

  pairs <- build_start_stop(df, states)
  fit <- survfit(Surv(tstart, tstop, event) ~ 1, data = pairs, id = id, istate = from, p0 = p0_vec)
  sfit <- summary(fit)

  # Filter sfit to integer steps
  int_mask <- (sfit$time %% 1 == 0)
  steps_int <- sfit$time[int_mask]
  pstate_int <- sfit$pstate[int_mask, , drop = FALSE]
  n_risk_int <- sfit$n.risk[int_mask, , drop = FALSE]

  cat(sprintf("\n# Fixture: %s\n", name))
  cat("states = ", sprintf("%s\n", paste(sprintf("'%s'", states_char), collapse = ", ")))
  cat("initial_p0 = ", fmt_vector(p0_vec), "\n")
  cat("steps = ", fmt_vector(steps_int), "\n")
  cat("n_risk = ", fmt_matrix(n_risk_int), "\n")
  cat("pstate = ", fmt_matrix(pstate_int), "\n")

  cat("directed_transitions = [\n")
  for (s in steps_int) {
    sub_p <- pairs[pairs$tstop == s & pairs$event != "(censor)", ]
    mat <- table(factor(sub_p$from, levels = states), factor(sub_p$event, levels = states))
    for (i in seq_along(states_char)) {
      for (j in seq_along(states_char)) {
        cnt <- mat[i, j]
        if (i != j) { # off-diagonal
          cat(sprintf("    {'step': %d, 'from': '%s', 'to': '%s', 'count': %d},\n", s, states_char[i], states_char[j], cnt))
        }
      }
    }
  }
  cat("]\n")
}

f1_df <- data.frame(
  id = c(1, 1, 1, 2, 2, 2, 3, 3, 3),
  step = c(0, 1, 2, 0, 1, 2, 0, 1, 2),
  state = c(1, 1, 2, 1, 2, 2, 1, 1, 1)
)
run_survfit_fixture("1. Homogeneous Baseline Complete Panel", f1_df, c(1, 2))

f2_df <- data.frame(
  id = c(1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4),
  step = c(0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2),
  state = c(1, 2, 3, 1, 1, 2, 2, 3, 2, 3, 2, 1)
)
run_survfit_fixture("2. Heterogeneous Baseline Complete Panel", f2_df, c(1, 2, 3))

f3_df <- data.frame(
  id = c(1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 5, 5, 5),
  step = c(0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2),
  state = c("A", "B", "C", "B", "B", "A", "C", "A", "B", "A", "A", "C", "B", "C", "C")
)
run_survfit_fixture("3. Complex Transitions & Initial States", f3_df, c("A", "B", "C"))

f4_df <- data.frame(
  id = c(1, 1, 1, 2, 2, 2),
  step = c(1, 2, 3, 1, 2, 3),
  state = c("H", "S", "H", "H", "H", "S")
)
run_survfit_fixture("4. Reversible Transitions", f4_df, c("H", "S"))

f5_df <- data.frame(
  id = c(1, 1, 2, 2, 3, 3),
  step = c(1, 2, 1, 2, 1, 2),
  state = c(1, 3, 1, 3, 1, 2)
)
run_survfit_fixture("5. Nonadjacent Jumps & Simultaneous", f5_df, c(1, 2, 3))

f7_df <- data.frame(
  id = c("A", "A", "A", "B", "B", "B", "C", "C", "D", "D", "D", "E", "E", "F"),
  step = c(0, 1, 2, 0, 1, 2, 0, 1, 0, 1, 2, 0, 1, 0),
  state = c(1, 1, 2, 1, 2, 2, 1, 1, 2, 2, 1, 2, 1, 2)
)
run_survfit_fixture("7. Independent Terminal Dropout", f7_df, c(1, 2))

f9_df <- data.frame(
  id = c("A", "B", "B", "C", "C"),
  step = c(0, 0, 1, 0, 1),
  state = c(1, 1, 2, 2, 2)
)
run_survfit_fixture("9. Minimal Estimable Baseline-Only Subject Diagnostic", f9_df, c(1, 2))

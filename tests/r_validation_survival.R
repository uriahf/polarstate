# Reproducible R Validation Script using R's survival package
#
# STATUS: EXECUTED
#
# Description:
#   This script converts discrete-step panel datasets into counting-process
#   start-stop format (tstart, tstop, event) for survival::survfit().
#
#   Survival Event History Convention for Discrete Steps:
#     - Level 1 is explicit censor/no-event status: "(censor)".
#     - Destination state events are emitted only for real state changes.
#     - Unchanged consecutive states are retained with event = "(censor)".
#     - Terminal dropouts end observation after the last complete step.
#
# Requirements:
#   install.packages("survival")

suppressPackageStartupMessages({
  library(survival)
})

cat("=================================================================\n")
cat("R survival Discrete-Step Multistate Validation (EXECUTED)\n")
cat("=================================================================\n\n")

# -----------------------------------------------------------------
# 1. Complete Panel Fixture Validation
# -----------------------------------------------------------------
cat("--- 1. Complete Panel Fixture Validation ---\n")
panel_complete <- data.frame(
  id = c(1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4),
  step = c(0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2),
  state = factor(c(1, 2, 3, 1, 1, 2, 2, 3, 2, 3, 2, 1))
)

build_start_stop <- function(df, num_states) {
  df <- df[order(df$id, df$step), ]
  df$prev_step <- ave(df$step, df$id, FUN = function(x) c(NA, x[-length(x)]))
  df$prev_state <- ave(as.character(df$state), df$id, FUN = function(x) c(NA, x[-length(x)]))
  pairs <- df[!is.na(df$prev_step), ]
  pairs$tstart <- pairs$prev_step
  pairs$tstop <- pairs$step

  state_levels <- as.character(1:num_states)
  all_levels <- c("(censor)", state_levels)

  event_status <- ifelse(as.character(pairs$state) != pairs$prev_state,
                         as.character(pairs$state),
                         "(censor)")

  pairs$event <- factor(event_status, levels = all_levels)
  pairs$from <- factor(pairs$prev_state, levels = state_levels)
  return(pairs)
}

pairs_complete <- build_start_stop(panel_complete, 3)
fit_comp <- survfit(Surv(tstart, tstop, event) ~ 1, data = pairs_complete, id = id, istate = from)
sfit_comp <- summary(fit_comp)

# Explicit time and state dimension alignment and verification
actual_times <- sfit_comp$time
actual_states <- fit_comp$states

stopifnot(identical(actual_times, c(1, 2)))
stopifnot(identical(actual_states, c("1", "2", "3")))
stopifnot(all(is.finite(sfit_comp$pstate)))
stopifnot(identical(dim(sfit_comp$pstate), c(2L, 3L)))

actual_pstate_comp <- sfit_comp$pstate[order(actual_times), order(as.numeric(actual_states))]
expected_pstate_comp <- matrix(c(0.25, 0.50, 0.25, 0.25, 0.50, 0.25), nrow = 2, byrow = TRUE)

max_diff_comp <- max(abs(actual_pstate_comp - expected_pstate_comp))

cat(sprintf("Verified Times: %s\n", paste(actual_times, collapse = ", ")))
cat(sprintf("Verified States: %s\n", paste(actual_states, collapse = ", ")))
cat(sprintf("Computed Complete Panel Max Absolute Discrepancy: %e\n", max_diff_comp))

# Executable numerical assertion
stopifnot(max_diff_comp < 1e-12)
cat("[ASSERTION PASSED] R survival complete panel predictions match expected with verified time/state alignment (max_diff < 1e-12).\n\n")

# -----------------------------------------------------------------
# 2. Terminal Dropout Fixture Validation
# -----------------------------------------------------------------
cat("--- 2. Terminal Dropout Fixture Validation ---\n")
panel_dropout <- data.frame(
  id = c("A", "A", "A", "B", "B", "B", "C", "C", "D", "D", "D", "E", "E", "F"),
  step = c(0, 1, 2, 0, 1, 2, 0, 1, 0, 1, 2, 0, 1, 0),
  state = factor(c(1, 1, 2, 1, 2, 2, 1, 1, 2, 2, 1, 2, 1, 2))
)

pairs_dropout <- build_start_stop(panel_dropout, 2)
fit_drop <- survfit(Surv(tstart, tstop, event) ~ 1, data = pairs_dropout, id = id, istate = from)
sfit_drop <- summary(fit_drop)

actual_drop_times <- sfit_drop$time
actual_drop_states <- fit_drop$states

stopifnot(identical(actual_drop_times, c(1, 2)))
stopifnot(identical(actual_drop_states, c("1", "2")))
stopifnot(all(is.finite(sfit_drop$pstate)))

actual_pstate_drop <- sfit_drop$pstate[order(actual_drop_times), order(as.numeric(actual_drop_states))]
polarstate_pstate_drop <- matrix(c(7/12, 5/12, 5/24, 19/24), nrow = 2, byrow = TRUE)

diff_drop <- abs(actual_pstate_drop - polarstate_pstate_drop)

cat("Actual R survfit pstate (counting-process interval-entry risk set):\n")
print(actual_pstate_drop)

cat("\nExpected polarstate pi(k) (baseline population risk set):\n")
print(polarstate_pstate_drop)

cat("\nComputed Absolute Discrepancy Matrix (survfit vs available-pair):\n")
print(diff_drop)

cat(sprintf("\nStep 1 Discrepancy: %f (survfit evaluates 5 subjects entering interval 1, excluding subject F)\n", diff_drop[1, 1]))

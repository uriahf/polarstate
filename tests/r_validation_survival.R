# Reproducible R Validation Script using R's survival package
#
# STATUS: NOT EXECUTED (R environment unavailable in sandbox)
#
# Description:
#   This script converts a discrete-step panel dataset into a counting-process
#   start-stop format (tstart, tstop, event) suitable for survival::survfit().
#
#   Survival Event History Convention for Discrete Steps:
#     - Level 1 is explicit censor/no-event status: "(censor)".
#     - Destination state events are emitted only for real state changes.
#     - Unchanged consecutive states are retained with event = "(censor)".
#     - Terminal dropouts end observation after the last complete step.
#
#   Estimand Comparison Note:
#     Standard counting-process survfit risk sets include all subjects under
#     observation in interval (tstart, tstop]. In contrast, the available-pair
#     discrete estimator excludes subjects dropping out before step k.
#     If dropout occurs, risk sets and transition estimates between counting process
#     survfit and available-pair estimators can diverge at the first dropout step.
#
# Requirements:
#   install.packages("survival")

suppressPackageStartupMessages({
  library(survival)
})

cat("=================================================================\n")
cat("R survival Discrete-Step Multistate Validation (NOT EXECUTED)\n")
cat("=================================================================\n\n")

# Complete panel fixture dataset
panel <- data.frame(
  id = c(1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4),
  step = c(0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2),
  state = factor(c(1, 2, 3, 1, 1, 2, 2, 3, 2, 3, 2, 1))
)

build_multistate_start_stop <- function(df) {
  df <- df[order(df$id, df$step), ]

  df$prev_step <- ave(df$step, df$id, FUN = function(x) c(NA, x[-length(x)]))
  df$prev_state <- ave(as.character(df$state), df$id, FUN = function(x) c(NA, x[-length(x)]))

  pairs <- df[!is.na(df$prev_step), ]
  pairs$tstart <- pairs$prev_step
  pairs$tstop <- pairs$step

  state_levels <- levels(df$state)
  all_levels <- c("(censor)", state_levels)

  event_status <- ifelse(as.character(pairs$state) != pairs$prev_state,
                         as.character(pairs$state),
                         "(censor)")

  pairs$event <- factor(event_status, levels = all_levels)
  pairs$from <- factor(pairs$prev_state, levels = state_levels)

  return(pairs)
}

start_stop_df <- build_multistate_start_stop(panel)

fit <- survfit(Surv(tstart, tstop, event) ~ 1, data = start_stop_df, id = id, istate = from)

# Structured Machine-Readable Reference Export Format:
# Build long-format table mapping time/step, state, pstate, n.risk, n.event, n.censor
states <- colnames(fit$pstate)
times <- fit$time

results_list <- list()
for (i in seq_along(times)) {
  for (j in seq_along(states)) {
    results_list[[length(results_list) + 1]] <- data.frame(
      time = times[i],
      state = states[j],
      pstate = fit$pstate[i, j],
      n_risk = if (is.matrix(fit$n.risk)) fit$n.risk[i, j] else fit$n.risk[i],
      n_event = if (is.matrix(fit$n.event)) fit$n.event[i, j] else fit$n.event[i],
      n_censor = if (is.matrix(fit$n.censor)) fit$n.censor[i, j] else fit$n.censor[i]
    )
  }
}

structured_results <- do.call(rbind, results_list)

cat("--- Machine-Readable Reference Table (Step / State Aligned) ---\n")
print(structured_results, row.names = FALSE)

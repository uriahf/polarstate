# Reproducible R Validation Script using R's survival package
#
# STATUS: NOT EXECUTED (R environment unavailable in sandbox)
#
# Description:
#   This script converts a discrete-step panel dataset into a counting-process
#   start-stop format (tstart, tstop, status) suitable for survival::survfit().
#
#   Survival Event History Convention for Discrete Steps:
#     - Explicit censor/no-event status is defined as level 1: "(censor)".
#     - Transitions to a different state are coded as destination state events.
#     - Unchanged consecutive states are retained with status = "(censor)".
#     - Terminal dropouts end observation after the last complete step.
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

  # Lag step and state per subject
  df$prev_step <- ave(df$step, df$id, FUN = function(x) c(NA, x[-length(x)]))
  df$prev_state <- ave(as.character(df$state), df$id, FUN = function(x) c(NA, x[-length(x)]))

  pairs <- df[!is.na(df$prev_step), ]
  pairs$tstart <- pairs$prev_step
  pairs$tstop <- pairs$step

  # Event status: level 1 is "(censor)" for no change, followed by state levels
  state_levels <- levels(df$state)
  all_levels <- c("(censor)", state_levels)

  # Destination state event iff state changes; otherwise "(censor)"
  event_status <- ifelse(as.character(pairs$state) != pairs$prev_state,
                         as.character(pairs$state),
                         "(censor)")

  pairs$event <- factor(event_status, levels = all_levels)
  pairs$from <- factor(pairs$prev_state, levels = state_levels)

  return(pairs)
}

start_stop_df <- build_multistate_start_stop(panel)

# Fit Aalen-Johansen multistate model with initial states (istate)
fit <- survfit(Surv(tstart, tstop, event) ~ 1, data = start_stop_df, id = id, istate = from)

cat("--- Summary Table: State Occupancy Probabilities (pstate) ---\n")
print(fit$pstate)

cat("\n--- Summary Table: At-Risk Denominators (n.risk) ---\n")
print(fit$n.risk)

cat("\n--- Summary Table: Event Counts (n.event) ---\n")
print(fit$n.event)

cat("\n--- Summary Table: Censoring Counts (n.censor) ---\n")
print(fit$n.censor)

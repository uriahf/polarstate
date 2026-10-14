# Reproducible R Validation Script using R's survival package
#
# Note: R is unavailable in this execution environment. This script serves as
# an independent reference for external execution and verification.
#
# Requirements:
#   install.packages("survival")

suppressPackageStartupMessages({
  library(survival)
})

# Define reference panel dataset (long format: id, step, state)
panel <- data.frame(
  id = c(1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4),
  step = c(1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3),
  state = factor(c("A", "B", "C", "A", "A", "B", "B", "B", "B", "A", "C", "C"))
)

# Convert discrete step panel to start-stop (tstart, tstop, event) for survival::survfit
# For step k-1 to step k:
# tstart = step - 1, tstop = step
# status = target state at step k
build_start_stop <- function(df) {
  df <- df[order(df$id, df$step), ]
  df$prev_step <- ave(df$step, df$id, FUN = function(x) c(NA, x[-length(x)]))
  df$prev_state <- ave(as.character(df$state), df$id, FUN = function(x) c(NA, x[-length(x)]))

  pairs <- df[!is.na(df$prev_step), ]
  pairs$tstart <- pairs$prev_step
  pairs$tstop <- pairs$step
  pairs$event <- factor(pairs$state, levels = levels(df$state))
  pairs$from <- factor(pairs$prev_state, levels = levels(df$state))

  return(pairs)
}

start_stop_data <- build_start_stop(panel)

cat("--- R survival multistate fit ---\n")
# Fit multistate Aalen-Johansen using survival::survfit
fit <- survfit(Surv(tstart, tstop, event) ~ 1, data = start_stop_data, id = id, istate = from)

print(fit)
cat("\nState occupancy probabilities (pstate):\n")
print(fit$pstate)

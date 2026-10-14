# Reproducible R Validation Script using R's Hmisc package
#
# Note: R is unavailable in this execution environment. This script serves as
# an independent reference for external execution and verification.
#
# Requirements:
#   install.packages("Hmisc")

suppressPackageStartupMessages({
  library(Hmisc)
})

# Define reference complete panel dataset (uncensored)
panel <- data.frame(
  id = c(1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4),
  step = c(1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3),
  state = c("A", "B", "C", "A", "A", "B", "B", "B", "B", "A", "C", "C")
)

cat("--- Empirical Proportions by Step (Hmisc propsPO / prop.table) ---\n")

# Compute empirical proportions per step
props <- prop.table(table(panel$step, panel$state), margin = 1)
print(props)

cat("\n--- Consecutive Transition Counts (Hmisc propsTrans / table) ---\n")
panel_order <- panel[order(panel$id, panel$step), ]
panel_order$prev_state <- ave(panel_order$state, panel_order$id, FUN = function(x) c(NA, x[-length(x)]))

pairs <- panel_order[!is.na(panel_order$prev_state), ]

for (s in unique(pairs$step)) {
  cat(sprintf("\nTransitions at Step %d (from Step %d):\n", s, s - 1))
  sub_pairs <- pairs[pairs$step == s, ]
  trans_mat <- table(From = sub_pairs$prev_state, To = sub_pairs$state)
  print(trans_mat)
  cat("\nTransition Probabilities:\n")
  print(prop.table(trans_mat, margin = 1))
}

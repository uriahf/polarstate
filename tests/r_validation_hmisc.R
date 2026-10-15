# Reproducible R Validation Script using R's Hmisc and Base R
#
# STATUS: EXECUTED
#
# Description:
#   This script computes empirical state occupancy proportions and
#   consecutive transition matrices using base R and Hmisc functions.
#
# Requirements:
#   install.packages("Hmisc")

suppressPackageStartupMessages({
  library(Hmisc)
})

cat("=================================================================\n")
cat("R Hmisc / Base-R Discrete-Step Multistate Validation (EXECUTED)\n")
cat("=================================================================\n\n")

# Complete panel fixture dataset
panel <- data.frame(
  id = c(1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4),
  step = c(0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2),
  state = factor(c(1, 2, 3, 1, 1, 2, 2, 3, 2, 3, 2, 1))
)

cat("--- Base-R: Empirical State Occupancy Proportions by Step ---\n")
props <- prop.table(table(Step = panel$step, State = panel$state), margin = 1)
print(props)

cat("\n--- Base-R / Hmisc: Consecutive Step Transition Matrices ---\n")
panel_order <- panel[order(panel$id, panel$step), ]
panel_order$prev_state <- ave(as.character(panel_order$state), panel_order$id, FUN = function(x) c(NA, x[-length(x)]))

pairs <- panel_order[!is.na(panel_order$prev_state), ]

for (s in sort(unique(pairs$step))) {
  cat(sprintf("\nStep %d Transitions (from Step %d):\n", s, s - 1))
  sub_pairs <- pairs[pairs$step == s, ]

  trans_counts <- table(From = sub_pairs$prev_state, To = sub_pairs$state)
  cat("Transition Counts:\n")
  print(trans_counts)

  cat("Conditional Transition Probabilities Q(k):\n")
  print(prop.table(trans_counts, margin = 1))
}

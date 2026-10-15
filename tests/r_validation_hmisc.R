# Reproducible R Validation Script using R's Hmisc package
#
# STATUS: NOT EXECUTED (R environment unavailable in sandbox)
#
# Description:
#   This script uses base R and Hmisc functions to compute empirical state
#   proportions and consecutive discrete transition counts for panel data.
#
# Requirements:
#   install.packages("Hmisc")

suppressPackageStartupMessages({
  library(Hmisc)
})

cat("=================================================================\n")
cat("R Hmisc Discrete-Step Multistate Validation (NOT EXECUTED)\n")
cat("=================================================================\n\n")

# Complete panel fixture dataset
panel <- data.frame(
  id = c(1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4),
  step = c(0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2),
  state = factor(c(1, 2, 3, 1, 1, 2, 2, 3, 2, 3, 2, 1))
)

cat("--- Base R / Empirical Proportions by Step ---\n")
props <- prop.table(table(Step = panel$step, State = panel$state), margin = 1)
print(props)

cat("\n--- Hmisc::propsTrans Transition Matrix Analysis ---\n")
panel_order <- panel[order(panel$id, panel$step), ]
panel_order$prev_state <- ave(as.character(panel_order$state), panel_order$id, FUN = function(x) c(NA, x[-length(x)]))

pairs <- panel_order[!is.na(panel_order$prev_state), ]

for (s in sort(unique(pairs$step))) {
  cat(sprintf("\nStep %d Transitions (from Step %d):\n", s, s - 1))
  sub_pairs <- pairs[pairs$step == s, ]
  trans_mat <- table(From = sub_pairs$prev_state, To = sub_pairs$state)
  print(trans_mat)
  cat("Conditional Transition Probabilities:\n")
  print(prop.table(trans_mat, margin = 1))
}

cat("\n--- Hmisc::propsPO Proportional Odds Inspection ---\n")
if (exists("propsPO", where = "package:Hmisc")) {
  tryCatch({
    po_fit <- propsPO(state ~ step, data = panel)
    print(po_fit)
  }, error = function(e) {
    cat("Hmisc::propsPO execution note:", conditionMessage(e), "\n")
  })
}

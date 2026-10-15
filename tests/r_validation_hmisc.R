# Reproducible Base-R Empirical Validation Script
#
# STATUS: EXECUTED
#
# Description:
#   This script computes empirical state occupancy proportions and
#   consecutive transition matrices using Base R table utilities.

cat("=================================================================\n")
cat("Base-R Empirical Discrete-Step Multistate Validation (EXECUTED)\n")
cat("=================================================================\n\n")

# Complete panel fixture dataset
panel <- data.frame(
  id = c(1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4),
  step = c(0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2),
  state = factor(c(1, 2, 3, 1, 1, 2, 2, 3, 2, 3, 2, 1))
)

cat("--- 1. Base-R: Empirical State Occupancy Proportions by Step ---\n")
props <- prop.table(table(Step = panel$step, State = panel$state), margin = 1)
print(props)

expected_props <- matrix(
  c(0.50, 0.25, 0.25,
    0.25, 0.50, 0.25,
    0.25, 0.50, 0.25),
  nrow = 3, byrow = TRUE
)

max_diff_props <- max(abs(props - expected_props))
stopifnot(max_diff_props < 1e-12)
cat(sprintf("[ASSERTION PASSED] Empirical occupancy proportions match expected (max_diff = %e).\n\n", max_diff_props))

cat("--- 2. Base-R: Consecutive Step Transition Probability Matrices Q(k) ---\n")
panel_order <- panel[order(panel$id, panel$step), ]
panel_order$prev_state <- ave(as.character(panel_order$state), panel_order$id, FUN = function(x) c(NA, x[-length(x)]))

pairs <- panel_order[!is.na(panel_order$prev_state), ]

# Step 1 Q matrix
sub_pairs_1 <- pairs[pairs$step == 1, ]
q1_actual <- prop.table(table(From = sub_pairs_1$prev_state, To = sub_pairs_1$state), margin = 1)
expected_q1 <- matrix(c(0.5, 0.5, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0), nrow = 3, byrow = TRUE)
diff_q1 <- max(abs(q1_actual - expected_q1))
stopifnot(diff_q1 < 1e-12)
cat(sprintf("[ASSERTION PASSED] Q1 transition matrix matches expected (max_diff = %e).\n", diff_q1))

# Step 2 Q matrix
sub_pairs_2 <- pairs[pairs$step == 2, ]
q2_actual <- prop.table(table(From = sub_pairs_2$prev_state, To = sub_pairs_2$state), margin = 1)
expected_q2 <- matrix(c(0.0, 1.0, 0.0, 0.5, 0.0, 0.5, 0.0, 1.0, 0.0), nrow = 3, byrow = TRUE)
diff_q2 <- max(abs(q2_actual - expected_q2))
stopifnot(diff_q2 < 1e-12)
cat(sprintf("[ASSERTION PASSED] Q2 transition matrix matches expected (max_diff = %e).\n", diff_q2))

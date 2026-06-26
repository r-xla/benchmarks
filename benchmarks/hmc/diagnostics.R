# Sampling-quality diagnostics for the 2-D "banana" target, shared by every
# algorithm so they report comparable accuracy numbers.
#
# The target is U(theta) = theta1^2 / 200 + (theta2 - b * theta1^2 + 100 * b)^2 / 2.
# Under it theta1 ~ N(0, 100) and theta2 = z + b * theta1^2 - 100 * b with
# z ~ N(0, 1), so the marginal standard deviations have closed forms:
#   sd(theta1) = 10
#   sd(theta2) = sqrt(1 + b^2 * Var(theta1^2)) = sqrt(1 + 2e4 * b^2)
# (Var(theta1^2) = 2 * 100^2 = 2e4 for theta1 ~ N(0, 100)).

# Analytic marginal standard deviations of the banana target.
banana_true_sd <- function(b) {
  c(sd1 = 10, sd2 = sqrt(1 + 2e4 * b^2))
}

# `samples` is a (n_samples, n_chains, 2) array of pooled draws across all
# chains. Returns the estimated marginal sds and the mean relative absolute
# error against the analytic truth.
hmc_diagnostics <- function(samples, b) {
  # The last (parameter) axis is slowest-varying, so column-major flattening
  # places all theta1 draws in column 1 and all theta2 draws in column 2.
  pooled <- matrix(samples, ncol = 2L)
  sd1 <- sd(pooled[, 1L])
  sd2 <- sd(pooled[, 2L])
  truth <- banana_true_sd(b)
  err <- mean(c(
    abs(sd1 - truth[["sd1"]]) / truth[["sd1"]],
    abs(sd2 - truth[["sd2"]]) / truth[["sd2"]]
  ))
  list(sd1 = sd1, sd2 = sd2, err = err)
}

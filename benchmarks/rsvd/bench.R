# Benchmark: randomized SVD with power iteration --- anvl (XLA) vs
# rsvd::rsvd (R + LAPACK).
#
# Both implementations follow Halko/Martinsson/Tropp: a Gaussian sketch +
# QR for the range finder, q subspace iterations (with re-orthonormalization
# between applications) for spectra that decay slowly, and a small dense
# SVD on the projected matrix. The anvl algorithm is the one developed in
# the anvl "Randomized SVD" vignette.
#
# Output: results.rds (long-format data.frame).

library(anvl)
library(pjrt)  # for await()
library(rsvd)
library(bench)
library(here)

range_finder <- jit(function(A, Omega, q) {
  Y <- A %*% Omega
  Q <- nv_qr(Y)$Q
  for (i in seq_len(q)) {
    Z <- nv_qr(t(A) %*% Q)$Q
    Q <- nv_qr(A %*% Z)$Q
  }
  Q
}, static = "q")

rsvd_anvl <- jit(function(A, state, k, p, q) {
  n <- shape(A)[2L]
  l <- k + p
  Omega <- nv_rnorm(state, dtype = "f32", shape = c(n, l))[[2L]]
  Q <- range_finder(A, Omega, q)
  B <- t(Q) %*% A
  decomp <- nv_svd(B)
  U <- Q %*% decomp$u
  list(
    d = decomp$d[1:k],
    u = U[, 1:k],
    vt = decomp$vt[1:k, ]
  )
}, static = c("k", "p", "q"))

# Synthetic m-by-n matrix with a planted rank-(2k) signal and a small
# dense noise tail, so the leading k singular values are well-separated.
make_matrix <- function(m, n, k, seed = 1L) {
  set.seed(seed)
  rank <- min(2L * k, n)
  L <- matrix(rnorm(m * rank), m, rank)
  R <- matrix(rnorm(rank * n), rank, n)
  L %*% R + 0.01 * matrix(rnorm(m * n), m, n)
}

bench_one <- function(m, n, k, p, q, iterations = 5L) {
  A_mat <- make_matrix(m, n, k)
  A <- nv_array(A_mat, dtype = "f32")
  state <- nv_rng_state(0L)

  # Warm the JIT cache + first-call paths for both implementations.
  invisible(await(rsvd_anvl(A, state, k = k, p = p, q = q)$u))
  invisible(rsvd::rsvd(A_mat, k = k, p = p, q = q))

  res <- bench::mark(
    anvl = await(rsvd_anvl(A, state, k = k, p = p, q = q)$u),
    rsvd = rsvd::rsvd(A_mat, k = k, p = p, q = q),
    check = FALSE,
    memory = FALSE,
    iterations = iterations
  )

  data.frame(
    m = m, n = n, k = k, p = p, q = q,
    impl = as.character(res$expression),
    median_s = as.numeric(res$median),
    min_s = as.numeric(res$min),
    itr_per_sec = res$`itr/sec`,
    stringsAsFactors = FALSE
  )
}

config <- expand.grid(
  m = c(500L, 1000L, 2000L, 4000L),
  k = c(10L, 50L, 100L),
  p = 10L,
  q = c(0L, 1L, 2L),
  stringsAsFactors = FALSE,
  KEEP.OUT.ATTRS = FALSE
)
config$n <- as.integer(config$m / 2L)
config <- config[config$k <= config$n, ]

message(sprintf("Running %d configurations...", nrow(config)))

results <- do.call(rbind, lapply(seq_len(nrow(config)), function(i) {
  cfg <- config[i, ]
  message(sprintf(
    "[%d/%d] m=%d n=%d k=%d p=%d q=%d",
    i, nrow(config), cfg$m, cfg$n, cfg$k, cfg$p, cfg$q
  ))
  tryCatch(
    bench_one(cfg$m, cfg$n, cfg$k, cfg$p, cfg$q),
    error = function(e) {
      message("  error: ", conditionMessage(e))
      NULL
    }
  )
}))

out_path <- here("benchmarks", "rsvd", "results.rds")
saveRDS(results, out_path)
message("Saved: ", out_path)

if (!is.null(results)) {
  wide <- reshape(
    results[, c("m", "n", "k", "p", "q", "impl", "median_s")],
    direction = "wide",
    idvar = c("m", "n", "k", "p", "q"),
    timevar = "impl"
  )
  wide$speedup <- wide$median_s.rsvd / wide$median_s.anvl
  print(wide[order(wide$m, wide$k, wide$q), ], row.names = FALSE)
}

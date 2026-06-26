# Summarize HMC benchmark results
library(batchtools)
library(data.table)

get_result <- function(ids, what) {
  if (is.null(ids)) ids <- findDone()[[1L]]
  sapply(ids, function(i) {
    res <- loadResult(i)[[what]]
    if (is.null(res)) return(NA)
    res
  })
}

summarize <- function(ids) {
  jt <- getJobTable(ids) |> unwrap()
  jt <- jt[, c("n_chains", "n_samples", "n_warmup", "L", "eps", "b", "device", "algorithm", "repl", "compile_loop")]
  jt$algorithm <- ifelse(jt$algorithm == "anvl" & jt$compile_loop, "anvl_jit", jt$algorithm)
  jt$time_total <- get_result(ids, "time")
  jt$compile_time <- get_result(ids, "compile_time")
  jt$n_cores <- get_result(ids, "n_cores")
  jt$sd1 <- get_result(ids, "sd1")
  jt$sd2 <- get_result(ids, "sd2")
  jt$err <- get_result(ids, "err")
  jt$total_samples <- jt$n_chains * jt$n_samples
  jt$time_per_sample <- jt$time_total / jt$total_samples
  return(jt)
}

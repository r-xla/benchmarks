# Summarize benchmark results
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
  jt <- jt[, c("n", "p", "n_steps", "device", "algorithm", "repl", "learning_rate", "compile_loop")]
  jt$algorithm <- ifelse(jt$algorithm == "anvil" & jt$compile_loop, "anvil_jit", jt$algorithm)
  jt$time_total <- get_result(ids, "time")
  jt$n_params <- get_result(ids, "n_params")
  jt$compile_time <- get_result(ids, "compile_time")
  jt$time_per_step <- jt$time_total / jt$n_steps
  jt$loss <- get_result(ids, "loss")
  return(jt)
}

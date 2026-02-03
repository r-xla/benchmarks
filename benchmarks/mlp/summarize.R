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
  jt <- jt[, c("n_layers", "batch_size", "n", "device", "algorithm", "repl", "latent", "epochs", "p", "compile_loop")]
  jt$algorithm <- ifelse(jt$algorithm == "anvil" & jt$compile_loop, "anvil_jit", jt$algorithm)
  jt$time_total <- get_result(ids, "time")
  jt$n_batches <- jt$n / jt$batch_size
  jt$ncpus <- get_result(ids, "ncpus")
  jt$nparams <- get_result(ids, "n_params")
  jt$compile_time <- get_result(ids, "compile_time")
  jt$time_per_batch <- jt$time_total / (jt$n_batches * jt$epochs)
  jt$loss <- get_result(ids, "loss")
  return(jt)
}

library(here)
library(ggplot2)
library(data.table)

df <- readRDS(here::here("benchmarks", "hmc", "result-cpu-single-1.rds"))
setDT(df)

# --- Algorithm labels ---
alg_map <- c(
  rtorch   = "torch (R)",
  stan     = "Stan",
  anvl_jit = "anvl (jit loop)",
  anvl     = "anvl"
)
alg_levels <- c("torch (R)", "Stan", "anvl (jit loop)", "anvl")

# --- Colour palette ---
pal <- c(
  "torch (R)"       = "#D62728",
  "Stan"            = "#8C564B",
  "anvl (jit loop)" = "#2CA02C",
  "anvl"            = "#FF7F0E"
)

# --- Shared theme ---
theme_bench <- theme_bw(base_size = 11) +
  theme(
    panel.grid.minor = element_blank(),
    legend.position  = "bottom",
    legend.margin    = margin(0, 0, 0, 0),
    legend.text      = element_text(size = 9),
    legend.key.size  = unit(0.4, "cm"),
    strip.background = element_rect(fill = "grey95", color = "grey80"),
    strip.text       = element_text(size = 9, face = "bold"),
    axis.title       = element_text(size = 11),
    panel.spacing    = unit(0.6, "lines")
  )

prep <- function(d) {
  d <- copy(d)
  d[, Algorithm := factor(alg_map[algorithm], levels = alg_levels)]
  d[, .(
    time_total   = median(time_total),
    compile_time = median(compile_time),
    time_per_sample = median(time_per_sample),
    err          = median(err)
  ), by = .(Algorithm, n_chains)]
}

df_agg <- prep(df)
chain_breaks <- sort(unique(df_agg$n_chains))

# ==========================================================================
# Plot 1 – Wall time vs number of chains (incl. compile)
# ==========================================================================

p <- ggplot(df_agg, aes(x = n_chains, y = time_total + compile_time, color = Algorithm)) +
  geom_line(linewidth = 0.7) +
  geom_point(size = 1.6) +
  scale_color_manual(values = pal, name = NULL) +
  scale_x_log10(breaks = chain_breaks) +
  scale_y_log10() +
  labs(x = "Number of Parallel Chains", y = "Wall Time (s)",
       title = "HMC Sampling Wall Time - CPU Single Thread (1000 samples)") +
  theme_bench +
  guides(color = guide_legend(nrow = 1))

ggsave(here("benchmarks", "hmc", "hmc_benchmark.png"), p,
       width = 8, height = 5, dpi = 300)

# ==========================================================================
# Plot 1b – Wall time (excl. JIT compile)
# ==========================================================================

p_nc <- ggplot(df_agg, aes(x = n_chains, y = time_total, color = Algorithm)) +
  geom_line(linewidth = 0.7) +
  geom_point(size = 1.6) +
  scale_color_manual(values = pal, name = NULL) +
  scale_x_log10(breaks = chain_breaks) +
  scale_y_log10() +
  labs(x = "Number of Parallel Chains", y = "Wall Time (s)",
       title = "HMC Sampling Wall Time (excl. JIT) - CPU Single Thread (1000 samples)") +
  theme_bench +
  guides(color = guide_legend(nrow = 1))

ggsave(here("benchmarks", "hmc", "hmc_benchmark_nocompile.png"), p_nc,
       width = 8, height = 5, dpi = 300)

# ==========================================================================
# Plot 2 – Time per (post-warmup) sample, pooled over chains
# ==========================================================================

p_tps <- ggplot(df_agg, aes(x = n_chains, y = time_per_sample, color = Algorithm)) +
  geom_line(linewidth = 0.7) +
  geom_point(size = 1.6) +
  scale_color_manual(values = pal, name = NULL) +
  scale_x_log10(breaks = chain_breaks) +
  scale_y_log10() +
  labs(x = "Number of Parallel Chains", y = "Time per Sample (s)",
       title = "HMC Amortized Cost per Pooled Sample - CPU Single Thread",
       caption = "Total samples = n_chains x 1000.") +
  theme_bench +
  guides(color = guide_legend(nrow = 1))

ggsave(here("benchmarks", "hmc", "hmc_per_sample.png"), p_tps,
       width = 8, height = 5, dpi = 300)

# ==========================================================================
# Plot 3 – Sampling accuracy (relative error of marginal sds)
# ==========================================================================

p_err <- ggplot(df_agg, aes(x = n_chains, y = err, color = Algorithm)) +
  geom_line(linewidth = 0.7) +
  geom_point(size = 1.6) +
  scale_color_manual(values = pal, name = NULL) +
  scale_x_log10(breaks = chain_breaks) +
  scale_y_continuous(labels = scales::percent) +
  labs(x = "Number of Parallel Chains", y = "Mean Relative Error of sd(theta)",
       title = "HMC Sampling Accuracy vs Analytic Banana Moments - CPU Single Thread") +
  theme_bench +
  guides(color = guide_legend(nrow = 1))

ggsave(here("benchmarks", "hmc", "hmc_accuracy.png"), p_err,
       width = 8, height = 5, dpi = 300)

# ==========================================================================
# GPU Results
# ==========================================================================

gpu_path <- here::here("benchmarks", "hmc", "result-gpu.rds")
if (file.exists(gpu_path)) {
  df_gpu_agg <- prep(setDT(readRDS(gpu_path)))
  gpu_chain_breaks <- sort(unique(df_gpu_agg$n_chains))

  p_gpu <- ggplot(df_gpu_agg, aes(x = n_chains, y = time_total + compile_time, color = Algorithm)) +
    geom_line(linewidth = 0.7) +
    geom_point(size = 1.6) +
    scale_color_manual(values = pal, name = NULL) +
    scale_x_log10(breaks = gpu_chain_breaks) +
    scale_y_log10() +
    labs(x = "Number of Parallel Chains", y = "Wall Time (s)",
         title = "HMC Sampling Wall Time - GPU (1000 samples)") +
    theme_bench +
    guides(color = guide_legend(nrow = 1))

  ggsave(here("benchmarks", "hmc", "hmc_benchmark_gpu.png"), p_gpu,
         width = 8, height = 5, dpi = 300)

  p_tps_gpu <- ggplot(df_gpu_agg, aes(x = n_chains, y = time_per_sample, color = Algorithm)) +
    geom_line(linewidth = 0.7) +
    geom_point(size = 1.6) +
    scale_color_manual(values = pal, name = NULL) +
    scale_x_log10(breaks = gpu_chain_breaks) +
    scale_y_log10() +
    labs(x = "Number of Parallel Chains", y = "Time per Sample (s)",
         title = "HMC Amortized Cost per Pooled Sample - GPU") +
    theme_bench +
    guides(color = guide_legend(nrow = 1))

  ggsave(here("benchmarks", "hmc", "hmc_per_sample_gpu.png"), p_tps_gpu,
         width = 8, height = 5, dpi = 300)
}

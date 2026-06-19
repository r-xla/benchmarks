library(here)
library(ggplot2)
library(data.table)

df <- readRDS(here::here("benchmarks", "mlp", "result-cpu-single-1.rds"))
setDT(df)

# --- Algorithm labels ---
alg_map <- c(
  rtorch    = "torch (R)",
  pytorch   = "PyTorch",
  anvl_jit = "anvl (jit loop)",
  anvl     = "anvl"
)
alg_levels <- c("torch (R)", "PyTorch", "anvl (jit loop)", "anvl")
df[, Algorithm := factor(alg_map[algorithm], levels = alg_levels)]

# --- Aggregate across replications ---
df_agg <- df[, .(
  time_total   = median(time_total),
  compile_time = median(compile_time),
  loss         = median(loss)
), by = .(Algorithm, n_layers, latent, batch_size)]

# --- Colour palette ---
pal <- c(
  "torch (R)"        = "#D62728",
  "PyTorch"          = "#1F77B4",
  "anvl (jit loop)" = "#2CA02C",
  "anvl"            = "#FF7F0E"
)

# --- Shared theme ---
theme_bench <- theme_bw(base_size = 10) +
  theme(
    panel.grid.minor    = element_blank(),
    legend.position     = "bottom",
    legend.margin       = margin(0, 0, 0, 0),
    legend.text         = element_text(size = 8.5),
    legend.key.size     = unit(0.4, "cm"),
    strip.background    = element_rect(fill = "grey95", color = "grey80"),
    strip.text          = element_text(size = 8, face = "bold"),
    strip.text.y        = element_text(size = 7, face = "bold"),
    axis.text           = element_text(size = 7),
    axis.title          = element_text(size = 10),
    panel.spacing       = unit(0.4, "lines")
  )

facet_bs_latent <- facet_grid(
  batch_size ~ latent,
  labeller = labeller(
    batch_size = \(x) paste0("BS: ", x),
    latent     = \(x) paste0("Latent: ", x)
  ),
  scales = "free_y"
)

layer_breaks <- sort(unique(df_agg$n_layers))

# ==========================================================================
# Plot 1 – Wall time (incl. compile)
# ==========================================================================

p <- ggplot(df_agg, aes(x = n_layers, y = time_total + compile_time,
                         color = Algorithm)) +
  geom_line(linewidth = 0.6) +
  geom_point(size = 1.2) +
  scale_color_manual(values = pal, name = NULL) +
  scale_x_continuous(breaks = layer_breaks, minor_breaks = NULL) +
  scale_y_continuous(expand = expansion(mult = c(0.02, 0.05))) +
  facet_bs_latent +
  labs(x = "Number of Hidden Layers", y = "Wall Time (s)",
       title = "MLP Training Wall Time - CPU 32 Threads (10 Epochs)") +
  theme_bench +
  guides(color = guide_legend(nrow = 1))

ggsave(here("benchmarks", "mlp", "mlp_benchmark.png"), p,
       width = 10, height = 6.5, dpi = 300)

# ==========================================================================
# Plot 1b – Wall time (excl. JIT compile)
# ==========================================================================

p_nocompile <- ggplot(df_agg, aes(x = n_layers, y = time_total,
                                   color = Algorithm)) +
  geom_line(linewidth = 0.6) +
  geom_point(size = 1.2) +
  scale_color_manual(values = pal, name = NULL) +
  scale_x_continuous(breaks = layer_breaks, minor_breaks = NULL) +
  scale_y_continuous(expand = expansion(mult = c(0.02, 0.05))) +
  facet_bs_latent +
  labs(x = "Number of Hidden Layers", y = "Wall Time (s)",
       title = "MLP Training Wall Time (excl. JIT) - CPU 32 Threads (10 Epochs)") +
  theme_bench +
  guides(color = guide_legend(nrow = 1))

ggsave(here("benchmarks", "mlp", "mlp_benchmark_nocompile.png"), p_nocompile,
       width = 10, height = 6.5, dpi = 300)

# ==========================================================================
# Plot 1c – Loss
# ==========================================================================

p_loss <- ggplot(df_agg, aes(x = n_layers, y = loss, color = Algorithm)) +
  geom_line(linewidth = 0.6) +
  geom_point(size = 1.2) +
  scale_color_manual(values = pal, name = NULL) +
  scale_x_continuous(breaks = layer_breaks, minor_breaks = NULL) +
  scale_y_continuous(expand = expansion(mult = c(0.02, 0.05))) +
  facet_bs_latent +
  labs(x = "Number of Hidden Layers", y = "Loss",
       title = "Final Training Loss - CPU 32 Threads (10 Epochs)") +
  theme_bench +
  guides(color = guide_legend(nrow = 1))

ggsave(here("benchmarks", "mlp", "mlp_loss.png"), p_loss,
       width = 10, height = 6.5, dpi = 300)

# ==========================================================================
# Plot 2 – Amortised per-batch time as a function of epochs
# ==========================================================================

epoch_grid <- c(5, 10, 20, 50, 100, 200, 400)

# Expand per replication, then aggregate
df_amort <- df[, {
  amort <- time_per_batch + compile_time / (n_batches * epoch_grid)
  .(epochs = epoch_grid, amortized_tpb = amort)
}, by = .(Algorithm, n_layers, latent, batch_size, time_per_batch,
          compile_time, n_batches, repl)]

df_amort_agg <- df_amort[, .(
  atpb_med = median(amortized_tpb),
  atpb_q10 = quantile(amortized_tpb, 0.1),
  atpb_q90 = quantile(amortized_tpb, 0.9)
), by = .(Algorithm, n_layers, latent, batch_size, epochs)]

p2 <- ggplot(df_amort_agg[batch_size == 128 & latent == 160],
             aes(x = epochs, color = Algorithm, fill = Algorithm)) +
  geom_ribbon(aes(ymin = atpb_q10, ymax = atpb_q90), alpha = 0.2, color = NA) +
  geom_line(aes(y = atpb_med), linewidth = 0.6) +
  geom_point(aes(y = atpb_med), size = 1, show.legend = FALSE) +
  scale_color_manual(values = pal, name = NULL) +
  scale_fill_manual(values = pal, name = NULL) +
  scale_x_log10(breaks = epoch_grid) +
  scale_y_continuous(expand = expansion(mult = c(0.02, 0.08))) +
  facet_wrap(
    ~ n_layers, nrow = 1, scales = "free_y",
    labeller = labeller(n_layers = \(x) paste0("Hidden Layers: ", x))
  ) +
  labs(x = "Epochs", y = "Time per Batch (s)",
       title = "Compile-Time Amortization over Epochs - CPU 32 Threads (Batch Size: 128, Latent: 160)",
       caption = "Ribbons: 10-90% quantile.") +
  theme_bench +
  theme(panel.spacing = unit(0.8, "lines")) +
  guides(color = guide_legend(nrow = 1), fill = guide_legend(nrow = 1))

ggsave(here("benchmarks", "mlp", "mlp_amortize.png"), p2,
       width = 9, height = 3.5, dpi = 300)

# ==========================================================================
# GPU Results
# ==========================================================================

df_gpu <- readRDS(here::here("benchmarks", "mlp", "result-gpu.rds"))
setDT(df_gpu)

df_gpu[, Algorithm := factor(alg_map[algorithm], levels = alg_levels)]

df_gpu_agg <- df_gpu[, .(
  time_total   = median(time_total),
  compile_time = median(compile_time),
  loss         = median(loss)
), by = .(Algorithm, n_layers, latent, batch_size)]

gpu_layer_breaks <- sort(unique(df_gpu_agg$n_layers))

facet_bs_latent_gpu <- facet_grid(
  batch_size ~ latent,
  labeller = labeller(
    batch_size = \(x) paste0("BS: ", x),
    latent     = \(x) paste0("Latent: ", x)
  ),
  scales = "free_y"
)

# --- GPU Wall time ---
p_gpu <- ggplot(df_gpu_agg, aes(x = n_layers, y = time_total + compile_time,
                                 color = Algorithm)) +
  geom_line(linewidth = 0.6) +
  geom_point(size = 1.2) +
  scale_color_manual(values = pal, name = NULL) +
  scale_x_continuous(breaks = gpu_layer_breaks, minor_breaks = NULL) +
  scale_y_continuous(expand = expansion(mult = c(0.02, 0.05))) +
  facet_bs_latent_gpu +
  labs(x = "Number of Hidden Layers", y = "Wall Time (s)",
       title = "MLP Training Wall Time - GPU (10 Epochs)") +
  theme_bench +
  guides(color = guide_legend(nrow = 1))

ggsave(here("benchmarks", "mlp", "mlp_benchmark_gpu.png"), p_gpu,
       width = 10, height = 6.5, dpi = 300)

# --- GPU Loss ---
p_loss_gpu <- ggplot(df_gpu_agg, aes(x = n_layers, y = loss, color = Algorithm)) +
  geom_line(linewidth = 0.6) +
  geom_point(size = 1.2) +
  scale_color_manual(values = pal, name = NULL) +
  scale_x_continuous(breaks = gpu_layer_breaks, minor_breaks = NULL) +
  scale_y_continuous(expand = expansion(mult = c(0.02, 0.05))) +
  facet_bs_latent_gpu +
  labs(x = "Number of Hidden Layers", y = "Loss",
       title = "Final Training Loss - GPU (10 Epochs)") +
  theme_bench +
  guides(color = guide_legend(nrow = 1))

ggsave(here("benchmarks", "mlp", "mlp_loss_gpu.png"), p_loss_gpu,
       width = 10, height = 6.5, dpi = 300)

# --- Amortization plot (GPU) ---

df_gpu_amort <- df_gpu[, {
  amort <- time_per_batch + compile_time / (n_batches * epoch_grid)
  .(epochs = epoch_grid, amortized_tpb = amort)
}, by = .(Algorithm, n_layers, latent, batch_size, time_per_batch,
          compile_time, n_batches, repl)]

df_gpu_amort_agg <- df_gpu_amort[, .(
  atpb_med = median(amortized_tpb)
), by = .(Algorithm, n_layers, latent, batch_size, epochs)]

p2_gpu <- ggplot(df_gpu_amort_agg[batch_size == 128 & latent == 1600],
                 aes(x = epochs, color = Algorithm)) +
  geom_line(aes(y = atpb_med), linewidth = 0.6) +
  geom_point(aes(y = atpb_med), size = 1, show.legend = FALSE) +
  scale_color_manual(values = pal, name = NULL) +
  scale_x_log10(breaks = epoch_grid) +
  scale_y_continuous(expand = expansion(mult = c(0.02, 0.08))) +
  facet_wrap(
    ~ n_layers, nrow = 1, scales = "free_y",
    labeller = labeller(n_layers = \(x) paste0("Hidden Layers: ", x))
  ) +
  labs(x = "Epochs", y = "Time per Batch (s)",
       title = "Compile-Time Amortization over Epochs - GPU (Batch Size: 128, Latent: 1600)") +
  theme_bench +
  theme(panel.spacing = unit(0.8, "lines")) +
  guides(color = guide_legend(nrow = 1))

ggsave(here("benchmarks", "mlp", "mlp_amortize_gpu.png"), p2_gpu,
       width = 9, height = 3.5, dpi = 300)

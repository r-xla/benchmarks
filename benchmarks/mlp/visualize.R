library(here)
library(ggplot2)
library(data.table)

df <- readRDS(here::here("benchmarks", "mlp", "result-cpu-1.rds"))
setDT(df)

# --- Algorithm labels ---
alg_map <- c(
  rtorch    = "torch (R)",
  pytorch   = "PyTorch",
  anvil_jit = "anvil (jit loop)",
  anvil     = "anvil"
)
alg_levels <- c("torch (R)", "PyTorch", "anvil (jit loop)", "anvil")
df[, Algorithm := factor(alg_map[algorithm], levels = alg_levels)]

# --- Aggregate across replications ---
df_agg <- df[, .(
  time_total  = median(time_total),
  compile_time = median(compile_time)
), by = .(Algorithm, n_layers, latent, batch_size)]

# --- Manual dodge positions ---
n_alg   <- length(alg_levels)
group_w <- 3
bar_w   <- group_w / n_alg
half_w  <- bar_w * 0.44

df_agg[, alg_i := as.numeric(Algorithm)]
df_agg[, xpos  := n_layers + (alg_i - (n_alg + 1) / 2) * bar_w]
df_agg[, xmin  := xpos - half_w]
df_agg[, xmax  := xpos + half_w]

# --- Colour palettes ---
pal <- c(
  "torch (R)"        = "#D62728",
  "PyTorch"          = "#1F77B4",
  "anvil (jit loop)" = "#2CA02C",
  "anvil"            = "#FF7F0E"
)

lighten <- function(hex, amount = 0.55) {
  r <- col2rgb(hex) / 255
  r_light <- r + (1 - r) * amount
  rgb(r_light[1, ], r_light[2, ], r_light[3, ])
}

pal_light <- setNames(lighten(pal), paste0(names(pal), " (compile)"))
fill_pal  <- c(pal, pal_light)

# --- Build rectangle data ---
bars_runtime <- df_agg[, .(Algorithm, n_layers, latent, batch_size,
                           xmin, xmax,
                           ymin = 0, ymax = time_total,
                           fill_key = as.character(Algorithm))]

bars_compile <- df_agg[compile_time > 0,
                       .(Algorithm, n_layers, latent, batch_size,
                         xmin, xmax,
                         ymin = time_total, ymax = time_total + compile_time,
                         fill_key = paste0(as.character(Algorithm), " (compile)"))]

bars <- rbindlist(list(bars_runtime, bars_compile))
fill_levels <- c(alg_levels, paste0(alg_levels, " (compile)"))
bars[, fill_key := factor(fill_key, levels = fill_levels)]

# --- Legend: 6 entries, algorithms + their compile variants ---
legend_breaks <- c(
  "torch (R)", "PyTorch",
  "anvil (jit loop)", "anvil (jit loop) (compile)",
  "anvil",            "anvil (compile)"
)
legend_labels <- c(
  "torch (R)", "PyTorch",
  "anvil (jit loop)", "   + compile",
  "anvil",            "   + compile"
)

# --- Plot ---
p <- ggplot(bars, aes(xmin = xmin, xmax = xmax, ymin = ymin, ymax = ymax)) +
  geom_rect(aes(fill = fill_key), color = "white", linewidth = 0.12) +
  scale_fill_manual(
    values = fill_pal,
    breaks = legend_breaks,
    labels = legend_labels,
    name   = NULL
  ) +
  scale_x_continuous(breaks = c(0, 4, 8), minor_breaks = NULL) +
  scale_y_continuous(expand = expansion(mult = c(0, 0.05))) +
  facet_grid(
    batch_size ~ latent,
    labeller = labeller(
      batch_size = \(x) paste0("Batch Size: ", x),
      latent     = \(x) paste0("Latent: ", x)
    ),
    scales = "free_y"
  ) +
  labs(x = "Number of Hidden Layers", y = "Wall Time (s)",
       title = "MLP Training Wall Time (10 Epochs)") +
  theme_bw(base_size = 10) +
  theme(
    panel.grid.major.x = element_blank(),
    panel.grid.minor    = element_blank(),
    legend.position     = "bottom",
    legend.margin       = margin(0, 0, 0, 0),
    legend.text         = element_text(size = 8.5),
    legend.key.size     = unit(0.4, "cm"),
    strip.background    = element_rect(fill = "grey95", color = "grey80"),
    strip.text          = element_text(size = 8, face = "bold"),
    axis.text           = element_text(size = 7),
    axis.title          = element_text(size = 10),
    panel.spacing       = unit(0.4, "lines")
  ) +
  guides(fill = guide_legend(nrow = 1))

ggsave(here("benchmarks", "mlp", "mlp_benchmark.pdf"), p,
       width = 10, height = 6.5)
ggsave(here("benchmarks", "mlp", "mlp_benchmark.png"), p,
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
       title = "Compile-Time Amortization over Epochs (Batch Size: 128, Latent: 160)",
       caption = "Ribbons: 10-90% quantile.") +
  theme_bw(base_size = 10) +
  theme(
    panel.grid.minor    = element_blank(),
    legend.position     = "bottom",
    legend.margin       = margin(0, 0, 0, 0),
    legend.text         = element_text(size = 8.5),
    strip.background    = element_rect(fill = "grey95", color = "grey80"),
    strip.text          = element_text(size = 8, face = "bold"),
    axis.text           = element_text(size = 7),
    axis.title          = element_text(size = 10),
    panel.spacing       = unit(0.8, "lines")
  ) +
  guides(color = guide_legend(nrow = 1), fill = guide_legend(nrow = 1))

ggsave(here("benchmarks", "mlp", "mlp_amortize.pdf"), p2,
       width = 9, height = 3.5)
ggsave(here("benchmarks", "mlp", "mlp_amortize.png"), p2,
       width = 9, height = 3.5, dpi = 300)

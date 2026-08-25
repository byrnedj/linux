library(ggplot2)

## Same look as dma_charts_pdf.R (fio/netty/spark DSA charts).
COL <- c("cpu-only" = "#2a78d6", "dsa" = "#eb6834")
SURF <- "#fcfcfb"; INK <- "#0b0b0b"; INK2 <- "#52514e"; GRID <- "#e8e7e2"

thm <- theme_minimal(base_size = 12) + theme(
  plot.background = element_rect(fill = SURF, colour = NA),
  panel.background = element_rect(fill = SURF, colour = NA),
  panel.grid.major.x = element_blank(),
  panel.grid.minor = element_blank(),
  panel.grid.major.y = element_line(colour = GRID, linewidth = 0.4),
  axis.title = element_text(colour = INK2, size = 10),
  axis.text = element_text(colour = INK2, size = 9),
  plot.title = element_text(colour = INK, face = "bold", size = 13),
  plot.subtitle = element_text(colour = INK2, size = 9),
  strip.text = element_text(colour = INK, face = "bold", size = 10),
  legend.position = "top", legend.justification = "left",
  legend.title = element_blank(),
  legend.text = element_text(colour = INK, size = 10))

barchart <- function(df, title, subtitle, file, ncol = 2, height = 4.4) {
  df$mode <- factor(df$mode, levels = c("cpu-only", "dsa"))
  df$metric <- factor(df$metric, levels = unique(df$metric))
  p <- ggplot(df, aes(x = grp, y = value, fill = mode)) +
    geom_col(position = position_dodge(width = 0.55), width = 0.45) +
    geom_text(aes(label = lab, y = value + ymax_pad),
        position = position_dodge(width = 0.55),
        size = 3.0, colour = INK2, vjust = 0) +
    facet_wrap(~metric, scales = "free", ncol = ncol) +
    scale_fill_manual(values = COL) +
    scale_y_continuous(expand = expansion(mult = c(0, 0.16))) +
    labs(title = title, subtitle = subtitle, x = NULL, y = NULL) + thm
  ggsave(file, p, width = 9, height = height, device = cairo_pdf, bg = SURF)
  ggsave(sub("\\.pdf$", ".png", file), p, width = 9, height = height, dpi = 160, bg = SURF)
}

## ---- 1 GiB hugetlb: fault-time zeroing and VM provisioning -------------
## gnr-qual1, 7.2.0-mmoff+ (mm_offload/dcbm, 4 DSA x 4 kernel WQs, 1G max
## transfer), hugepagesz=1G hugepages=64. Medians of 3 (faults) / 6 (VM).
g1 <- data.frame(
  grp = c(rep(rep(c("1 thread", "4 threads", "16 threads"), each = 2), 2),
          rep(rep("64 GiB guest, prealloc\n(QEMU, 1G hugetlbfs)", each = 2), 2)),
  mode = rep(c("cpu-only", "dsa"), 8),
  metric = c(rep(c("1 GiB hugetlb fault: latency (ms) — lower is better",
                   "1 GiB hugetlb fault: kernel CPU, sys (s) — lower is better"), each = 6),
             rep(c("VM provisioning: wall (s) — lower is better",
                   "VM provisioning: kernel CPU, sys (s) — lower is better"), each = 2)),
  value = c(197.1, 16.95, 46.9, 4.49, 11.62, 11.07,
            12.54, 0.01, 11.90, 0.01, 11.78, 4.44,
            6.00, 1.33,
            9.94, 0.45))
g1$grp <- factor(g1$grp, levels = unique(g1$grp))
g1$lab <- ifelse(grepl("latency", g1$metric), sprintf("%.1f", g1$value), sprintf("%.2f", g1$value))
g1$ymax_pad <- ifelse(grepl("latency", g1$metric), 4,
               ifelse(grepl("fault: kernel", g1$metric), 0.3,
               ifelse(grepl("wall", g1$metric), 0.15, 0.25)))
barchart(g1, "1 GiB hugetlb zeroing: DSA memset offload (dcbm) vs cpu-only",
         "64 x 1G pages reserved at boot; every fault zeroes a 1 GiB folio. 16 threads: DRAM-bound, 30/64 faults admitted to DSA.",
         "mmoff_1g_chart.pdf", ncol = 2, height = 7.2)

## ---- 2 MiB THP zeroing and move_pages(2) migration ----------------------
## gnr-qual1, same kernel. Medians of 3 (tools/mm_offload/run_mmoff.sh).
mx <- data.frame(
  grp = rep(rep(c("1 thread", "4 threads", "16 threads"), each = 2), 4),
  mode = rep(c("cpu-only", "dsa"), 12),
  metric = rep(c("2 MiB THP fault: zeroing latency (us) — lower is better",
                 "2 MiB THP fault: kernel CPU, sys (s) — lower is better",
                 "move_pages 4 GiB of 2 MiB folios (GiB/s) — higher is better",
                 "move_pages 1 GiB of 4 KiB folios (GiB/s) — higher is better"), each = 6),
  value = c(184.4, 76.4, 48.7, 27.6, 20.7, 11.1,
            0.39, 0.05, 1.60, 0.24, 10.9, 2.26,
            2.96, 21.77, 11.01, 59.04, 51.77, 107.37,
            1.77, 2.18, 1.06, 1.35, 1.12, 1.48))
mx$grp <- factor(mx$grp, levels = unique(mx$grp))
mx$lab <- ifelse(grepl("latency", mx$metric), sprintf("%.0f", mx$value), sprintf("%.2f", mx$value))
mx$ymax_pad <- ifelse(grepl("latency", mx$metric), 4,
               ifelse(grepl("kernel", mx$metric), 0.25,
               ifelse(grepl("2 MiB folios", mx$metric), 2.5, 0.04)))
barchart(mx, "mm_offload/dcbm on DSA: THP zeroing and page migration vs cpu-only",
         "move_pages(2) node 0 -> 1 with verification; 4 KiB migration is rmap-bound (copy is 5-10% of the syscall).",
         "mmoff_matrix_chart.pdf", ncol = 2, height = 7.2)

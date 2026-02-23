# ==============================================================================
#   SCOUT Multi-Reader Crossover Trial — Baseline Balance Check
#   CHD subtype distribution and Mmain baseline accuracy by randomized set
# ==============================================================================

rm(list = ls())

# --- Dependencies -------------------------------------------------------------
required_packages <- c("readr", "readxl", "dplyr", "tidyr", "ggplot2", "scales", "writexl")

for (pkg in required_packages) {
  if (!require(pkg, character.only = TRUE, quietly = TRUE)) {
    install.packages(pkg)
    library(pkg, character.only = TRUE)
  }
}

# --- Configuration ------------------------------------------------------------
CONFIG <- list(
  gold_file         = "data/gold_standard_multi_reader.csv",
  path_control      = "data/control/",
  path_intervention = "data/intervention/",
  output_dir        = "output/multi_reader/"
)

# --- Utility ------------------------------------------------------------------
print_section <- function(title) {
  cat("\n", strrep("=", 70), "\n  ", title, "\n", strrep("=", 70), "\n")
}

# ==============================================================================
# Main analysis
# ==============================================================================

run_baseline_balance <- function(gold_file         = CONFIG$gold_file,
                                 path_control      = CONFIG$path_control,
                                 path_intervention = CONFIG$path_intervention,
                                 output_dir        = CONFIG$output_dir) {

  print_section("Baseline Balance Check")

  # --- 1. Load data -----------------------------------------------------------
  df_gold <- read_csv(gold_file, show_col_types = FALSE) %>%
    select(Case_ID, Gold_Label, Mmain_Pred = final_label_main) %>%
    mutate(Case_ID = as.character(Case_ID))

  cases_ctrl <- list.files(path_control, pattern = "*.xlsx", full.names = TRUE) %>%
    lapply(function(f) read_excel(f, col_types = "text") %>% pull(Case_ID)) %>%
    unlist() %>% unique()

  cases_int <- list.files(path_intervention, pattern = "*.xlsx", full.names = TRUE) %>%
    lapply(function(f) read_excel(f, col_types = "text") %>% pull(Case_ID)) %>%
    unlist() %>% unique()

  df_all <- df_gold %>%
    mutate(
      Set = case_when(
        Case_ID %in% cases_ctrl ~ "Control",
        Case_ID %in% cases_int  ~ "Intervention",
        TRUE ~ "Unassigned"
      )
    ) %>%
    filter(Set != "Unassigned")

  cat(sprintf("Control set:      %d cases\n", sum(df_all$Set == "Control")))
  cat(sprintf("Intervention set: %d cases\n", sum(df_all$Set == "Intervention")))

  # --- 2. CHD subtype distribution --------------------------------------------
  print_section("1. CHD Subtype Distribution (Gold Standard)")

  dist_table <- df_all %>%
    group_by(Set, Gold_Label) %>%
    summarise(n = n(), .groups = "drop") %>%
    group_by(Set) %>%
    mutate(Total = sum(n),
           Pct   = n / Total,
           Pct_Fmt = percent(Pct, accuracy = 0.1)) %>%
    ungroup()

  dist_wide <- dist_table %>%
    mutate(Display = paste0(n, " (", Pct_Fmt, ")")) %>%
    select(Set, Gold_Label, Display) %>%
    pivot_wider(names_from = Set, values_from = Display, values_fill = "0 (0.0%)")

  cat("\n--- CHD Subtype Distribution ---\n")
  print(dist_wide)

  chisq_dist <- chisq.test(table(df_all$Set, df_all$Gold_Label))
  cat(sprintf("\nBetween-set distribution (chi-squared):\n"))
  cat(sprintf("  chi2 = %.3f, df = %d, P = %.4f\n",
              chisq_dist$statistic, chisq_dist$parameter, chisq_dist$p.value))
  cat(ifelse(chisq_dist$p.value > 0.05,
             "  Conclusion: No significant difference (P > 0.05); randomization balanced.\n",
             "  Conclusion: Significant difference detected (P < 0.05).\n"))

  # --- 3. Mmain baseline accuracy ---------------------------------------------
  print_section("2. Mmain Baseline Accuracy")

  acc_table <- df_all %>%
    mutate(Correct = Mmain_Pred == Gold_Label) %>%
    group_by(Set) %>%
    summarise(
      N        = n(),
      Correct  = sum(Correct, na.rm = TRUE),
      Errors   = N - Correct,
      Accuracy = mean(Correct / N),
      Acc_Fmt  = percent(Accuracy, accuracy = 0.1),
      .groups  = "drop"
    )

  cat("\n--- Mmain Baseline Accuracy ---\n")
  print(acc_table)

  chisq_acc <- chisq.test(table(df_all$Set, df_all$Mmain_Pred == df_all$Gold_Label))
  cat(sprintf("\nBetween-set accuracy (chi-squared):\n"))
  cat(sprintf("  chi2 = %.3f, df = %d, P = %.4f\n",
              chisq_acc$statistic, chisq_acc$parameter, chisq_acc$p.value))
  cat(ifelse(chisq_acc$p.value > 0.05,
             "  Conclusion: No significant difference (P > 0.05); randomization balanced.\n",
             "  Conclusion: Significant difference detected (P < 0.05).\n"))

  # --- 4. Per-subtype accuracy ------------------------------------------------
  print_section("3. Per-Subtype Mmain Accuracy")

  acc_by_subtype <- df_all %>%
    mutate(Correct = Mmain_Pred == Gold_Label) %>%
    group_by(Set, Gold_Label) %>%
    summarise(
      n       = n(),
      Correct = sum(Correct, na.rm = TRUE),
      Acc_Fmt = percent(mean(Correct / n), accuracy = 0.1),
      .groups = "drop"
    )

  acc_subtype_wide <- acc_by_subtype %>%
    mutate(Display = paste0(Correct, "/", n, " (", Acc_Fmt, ")")) %>%
    select(Set, Gold_Label, Display) %>%
    pivot_wider(names_from = Set, values_from = Display, values_fill = "-")

  cat("\n--- Per-Subtype Accuracy ---\n")
  print(acc_subtype_wide)

  # --- 5. Figures -------------------------------------------------------------
  p1 <- ggplot(dist_table, aes(x = Gold_Label, y = n, fill = Set)) +
    geom_bar(stat = "identity",
             position = position_dodge(width = 0.8), width = 0.7) +
    geom_text(aes(label = paste0(n, "\n(", Pct_Fmt, ")")),
              position = position_dodge(width = 0.8), vjust = -0.3, size = 3) +
    scale_fill_manual(values = c("Control" = "#E64B35",
                                 "Intervention" = "#00A087")) +
    labs(title = "CHD Subtype Distribution by Randomized Set",
         subtitle = sprintf("Chi-squared: chi2 = %.2f, P = %.3f",
                            chisq_dist$statistic, chisq_dist$p.value),
         x = "CHD Subtype (Gold Standard)", y = "Count", fill = "Set") +
    theme_minimal(base_size = 12) +
    theme(axis.text.x = element_text(angle = 45, hjust = 1),
          legend.position = "top") +
    coord_cartesian(ylim = c(0, max(dist_table$n) * 1.2))

  p2 <- ggplot(acc_table, aes(x = Set, y = Accuracy, fill = Set)) +
    geom_bar(stat = "identity", width = 0.5) +
    geom_text(aes(label = paste0(Acc_Fmt, "\n(", Correct, "/", N, ")")),
              vjust = -0.3, size = 4) +
    geom_hline(yintercept = mean(acc_table$Accuracy),
               linetype = "dashed", color = "grey50") +
    scale_fill_manual(values = c("Control" = "#E64B35",
                                 "Intervention" = "#00A087")) +
    scale_y_continuous(labels = percent, limits = c(0, 1.1)) +
    labs(title = "Mmain Baseline Accuracy by Randomized Set",
         subtitle = sprintf("Chi-squared: chi2 = %.2f, P = %.3f",
                            chisq_acc$statistic, chisq_acc$p.value),
         x = "", y = "Accuracy") +
    theme_minimal(base_size = 12) +
    theme(legend.position = "none")

  ggsave(paste0(output_dir, "fig_baseline_subtype_distribution.png"),
         p1, width = 10, height = 6, dpi = 300)
  ggsave(paste0(output_dir, "fig_baseline_accuracy.png"),
         p2, width = 6, height = 5, dpi = 300)

  cat("\nFigures saved.\n")

  # --- 6. Export summary table ------------------------------------------------
  write_xlsx(
    list(
      Subtype_Distribution = dist_table %>%
        select(Set, Subtype = Gold_Label, Count = n, Proportion = Pct_Fmt),
      Overall_Accuracy     = acc_table %>%
        select(Set, N, Correct, Errors, Accuracy = Acc_Fmt),
      Per_Subtype_Accuracy = acc_by_subtype %>%
        select(Set, Subtype = Gold_Label, N = n, Correct, Accuracy = Acc_Fmt)
    ),
    paste0(output_dir, "baseline_balance_summary.xlsx")
  )

  cat("Summary exported: baseline_balance_summary.xlsx\n")

  return(list(
    data           = df_all,
    dist_table     = dist_table,
    acc_table      = acc_table,
    acc_by_subtype = acc_by_subtype,
    chisq_dist     = chisq_dist,
    chisq_acc      = chisq_acc
  ))
}

# ==============================================================================
# Run
# ==============================================================================
# results <- run_baseline_balance()

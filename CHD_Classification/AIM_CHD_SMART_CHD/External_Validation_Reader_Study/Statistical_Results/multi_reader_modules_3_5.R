# ==============================================================================
#   SCOUT Multi-Reader Crossover Trial — Modules 3–5
#   Module 3: Physician Review Outcome Statistics
#   Module 4: Stratified Accuracy Analysis with Hypothesis Testing
#   Module 5: API Cost Analysis
# ==============================================================================

rm(list = ls())

# --- Dependencies -------------------------------------------------------------
required_packages <- c("readr", "readxl", "writexl", "dplyr", "tidyr", "purrr",
                       "stringr", "scales", "ggplot2", "knitr", "tools")

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
  api_data_folder   = "data/api_logs/",
  prompt_s1         = "data/api_logs/prompt_S1_baseline.txt",
  prompt_mmain      = "data/api_logs/prompt_Mmain_optimized.txt",
  prompt_s3         = "data/api_logs/prompt_S3_checker.txt",
  output_dir        = "output/multi_reader/"
)

# --- Utility functions --------------------------------------------------------

#' Map physician file index to seniority stratum
get_seniority <- function(filename) {
  num <- as.numeric(str_extract(basename(filename), "\\d+"))
  case_when(
    num %in% c(1, 2) ~ "1. Junior",
    num %in% c(3, 4, 5) ~ "2. Mid-level",
    num %in% c(6, 7) ~ "3. Senior",
    TRUE ~ "Unknown"
  )
}

#' Read prompt file and return character count
get_prompt_len <- function(filepath) {
  if (file.exists(filepath)) {
    nchar(paste(readLines(filepath, warn = FALSE, encoding = "UTF-8"), collapse = "\n"))
  } else {
    warning(paste("File not found:", filepath))
    0
  }
}

#' Chi-squared P-value for between-arm accuracy comparison
calc_pval <- function(data) {
  tbl <- table(data$Group_Name, data$Final_Decision == data$Gold_Label)
  if (all(dim(tbl) == c(2, 2))) {
    p <- chisq.test(tbl)$p.value
    if (p < 0.001) "P < 0.001" else sprintf("P = %.3f", p)
  } else {
    "N/A"
  }
}

print_section <- function(title) {
  cat("\n", strrep("=", 70), "\n  ", title, "\n", strrep("=", 70), "\n")
}

# ==============================================================================
# Module 3: Physician Review Outcome Statistics
# ==============================================================================

run_physician_review_stats <- function(gold_file         = CONFIG$gold_file,
                                       path_control      = CONFIG$path_control,
                                       path_intervention = CONFIG$path_intervention) {
  print_section("Module 3: Physician Review Outcomes")

  df_gold <- read_csv(gold_file, show_col_types = FALSE) %>%
    select(Case_ID, Gold_Label) %>%
    mutate(Case_ID = as.character(Case_ID))

  # Control arm (standard manual review)
  df_control <- list.files(path_control, pattern = "*.xlsx", full.names = TRUE) %>%
    map_dfr(~ read_excel(.x, col_types = "text")) %>%
    mutate(Case_ID = as.character(Case_ID)) %>%
    left_join(df_gold, by = "Case_ID") %>%
    mutate(
      Group_Name     = "Standard Review",
      is_flagged     = FALSE,
      Final_Decision = Physician_Label,
      Action_Type    = if_else(Final_Decision == Mmain_Pred, "Accept", "Override")
    )

  # Intervention arm (SCOUT-assisted review)
  df_intervention <- list.files(path_intervention, pattern = "*.xlsx", full.names = TRUE) %>%
    map_dfr(~ read_excel(.x, col_types = "text")) %>%
    mutate(Case_ID = as.character(Case_ID)) %>%
    left_join(df_gold, by = "Case_ID") %>%
    mutate(
      Group_Name     = "SCOUT-Assisted",
      is_flagged     = str_detect(SCOUT_Status, "REVIEW"),
      Final_Decision = if_else(is_flagged, Physician_Final, Mmain_Pred),
      Action_Type    = case_when(
        !is_flagged                    ~ "Accept",
        Physician_Final == Mmain_Pred  ~ "Accept",
        TRUE                           ~ "Override"
      )
    )

  df_all <- bind_rows(df_control, df_intervention)

  # Interaction-pattern classification
  detail_stats <- df_all %>%
    mutate(
      AI_Correct    = Mmain_Pred == Gold_Label,
      Final_Correct = Final_Decision == Gold_Label,
      Scenario = case_when(
        AI_Correct  & Action_Type == "Accept"   & Final_Correct  ~ "Correct Acceptance",
        !AI_Correct & Action_Type == "Override"  & Final_Correct  ~ "Successful Correction",
        AI_Correct  & Action_Type == "Override"  & !Final_Correct ~ "Incorrect Override",
        !AI_Correct & Action_Type == "Accept"    & !Final_Correct ~ "Automation Bias",
        Group_Name == "SCOUT-Assisted" & !AI_Correct & !is_flagged ~ "Silent Failure",
        TRUE ~ "Other"
      )
    ) %>%
    mutate(
      Scenario = if_else(
        Group_Name == "SCOUT-Assisted" & !AI_Correct & !is_flagged,
        "Silent Failure", Scenario)
    ) %>%
    group_by(Group_Name, Scenario) %>%
    summarise(Count = n(), .groups = "drop") %>%
    group_by(Group_Name) %>%
    mutate(Proportion = percent(Count / sum(Count), accuracy = 0.1)) %>%
    select(Group_Name, Scenario, Count, Proportion)

  acc_report <- df_all %>%
    group_by(Group_Name) %>%
    summarise(
      AI_Baseline_Acc    = mean(Mmain_Pred == Gold_Label, na.rm = TRUE),
      Final_Clinical_Acc = mean(Final_Decision == Gold_Label, na.rm = TRUE),
      Delta = Final_Clinical_Acc - AI_Baseline_Acc
    )

  cat("--- Accuracy Comparison ---\n")
  print(acc_report)
  cat("\n--- Interaction-Pattern Statistics ---\n")
  print(detail_stats)

  return(list(df_all = df_all, acc_report = acc_report, detail_stats = detail_stats))
}

# ==============================================================================
# Module 4: Stratified Accuracy Analysis with Hypothesis Testing
# ==============================================================================

run_stratified_analysis <- function(gold_file         = CONFIG$gold_file,
                                    path_control      = CONFIG$path_control,
                                    path_intervention = CONFIG$path_intervention,
                                    output_dir        = CONFIG$output_dir) {
  print_section("Module 4: Stratified Accuracy Analysis")

  df_gold <- read_csv(gold_file, show_col_types = FALSE) %>%
    select(Case_ID, Gold_Label) %>%
    mutate(Case_ID = as.character(Case_ID))

  # Control arm with seniority labels
  files_ctrl <- list.files(path_control, pattern = "*.xlsx", full.names = TRUE)
  df_control <- map_dfr(files_ctrl, function(f) {
    read_excel(f, col_types = "text") %>%
      mutate(source_file = basename(f), Doctor_Level = get_seniority(f))
  }) %>%
    mutate(Case_ID = as.character(Case_ID)) %>%
    left_join(df_gold, by = "Case_ID") %>%
    mutate(
      Group_Name     = "Standard Review",
      is_flagged     = FALSE,
      Final_Decision = Physician_Label,
      Action_Type    = if_else(Final_Decision == Mmain_Pred, "Accept", "Override")
    )

  # Intervention arm with seniority labels
  files_int <- list.files(path_intervention, pattern = "*.xlsx", full.names = TRUE)
  df_intervention <- map_dfr(files_int, function(f) {
    read_excel(f, col_types = "text") %>%
      mutate(source_file = basename(f), Doctor_Level = get_seniority(f))
  }) %>%
    mutate(Case_ID = as.character(Case_ID)) %>%
    left_join(df_gold, by = "Case_ID") %>%
    mutate(
      Group_Name     = "SCOUT-Assisted",
      is_flagged     = str_detect(SCOUT_Status, "REVIEW"),
      Final_Decision = if_else(is_flagged, Physician_Final, Mmain_Pred),
      Action_Type    = case_when(
        !is_flagged                    ~ "Accept",
        Physician_Final == Mmain_Pred  ~ "Accept",
        TRUE                           ~ "Override"
      )
    )

  df_all <- bind_rows(df_control, df_intervention)

  # Overall accuracy
  acc_overall <- df_all %>%
    group_by(Group_Name) %>%
    summarise(
      N              = n(),
      AI_Baseline    = mean(Mmain_Pred == Gold_Label, na.rm = TRUE),
      Final_Clinical = mean(Final_Decision == Gold_Label, na.rm = TRUE),
      .groups = "drop"
    )

  p_overall <- calc_pval(df_all)
  y_max_overall <- max(acc_overall$Final_Clinical) + 0.03

  # Stratified accuracy by physician seniority
  acc_stratified <- df_all %>%
    group_by(Group_Name, Doctor_Level) %>%
    summarise(
      N              = n(),
      AI_Baseline    = mean(Mmain_Pred == Gold_Label, na.rm = TRUE),
      Final_Clinical = mean(Final_Decision == Gold_Label, na.rm = TRUE),
      .groups = "drop"
    )

  p_stratified <- df_all %>%
    group_by(Doctor_Level) %>%
    nest() %>%
    mutate(p_label = map_chr(data, calc_pval)) %>%
    select(Doctor_Level, p_label)

  cat("--- Overall Accuracy ---\n")
  print(acc_overall %>% mutate(P_Value = p_overall))
  cat("\n--- Stratified Accuracy ---\n")
  print(acc_stratified %>% left_join(p_stratified, by = "Doctor_Level"))

  # --- Figure: Overall accuracy -----------------------------------------------
  p_fig_overall <- ggplot(acc_overall,
                          aes(x = Group_Name, y = Final_Clinical, fill = Group_Name)) +
    geom_bar(stat = "identity", width = 0.5, alpha = 0.8) +
    geom_text(aes(label = percent(Final_Clinical, accuracy = 0.1)),
              vjust = -0.5, size = 5) +
    geom_errorbar(aes(ymin = AI_Baseline, ymax = AI_Baseline),
                  width = 0.5, color = "red", linetype = "dashed", linewidth = 1) +
    geom_text(aes(y = AI_Baseline,
                  label = paste0("AI: ", percent(AI_Baseline, accuracy = 0.1))),
              vjust = 1.5, color = "red", size = 4) +
    annotate("segment", x = 1, xend = 2,
             y = y_max_overall, yend = y_max_overall, linewidth = 0.8) +
    annotate("segment", x = 1, xend = 1,
             y = y_max_overall, yend = y_max_overall - 0.01, linewidth = 0.8) +
    annotate("segment", x = 2, xend = 2,
             y = y_max_overall, yend = y_max_overall - 0.01, linewidth = 0.8) +
    annotate("text", x = 1.5, y = y_max_overall + 0.015,
             label = p_overall, size = 5, fontface = "bold") +
    scale_y_continuous(labels = percent, breaks = seq(0.6, 1.0, 0.1)) +
    coord_cartesian(ylim = c(0.6, 1.05)) +
    scale_fill_brewer(palette = "Set2") +
    labs(title = "Overall Diagnostic Accuracy",
         subtitle = paste0("Between-arm: ", p_overall,
                           " | Red dashed line: AI baseline"),
         x = "", y = "Accuracy") +
    theme_minimal(base_size = 14) +
    theme(legend.position = "none")

  ggsave(paste0(output_dir, "fig_overall_accuracy.png"),
         p_fig_overall, width = 8, height = 6, dpi = 300)

  # --- Figure: Stratified accuracy --------------------------------------------
  plot_strat <- acc_stratified %>%
    left_join(p_stratified, by = "Doctor_Level") %>%
    group_by(Doctor_Level) %>%
    mutate(y_bracket = max(Final_Clinical) + 0.03)

  levels_data <- plot_strat %>%
    ungroup() %>%
    select(Doctor_Level, p_label, y_bracket) %>%
    distinct() %>%
    mutate(x_center = as.numeric(factor(Doctor_Level)))

  p_fig_strat <- ggplot(plot_strat,
                        aes(x = Doctor_Level, y = Final_Clinical, fill = Group_Name)) +
    geom_bar(stat = "identity",
             position = position_dodge(width = 0.7), width = 0.7, alpha = 0.9) +
    geom_text(aes(label = percent(Final_Clinical, accuracy = 0.1)),
              position = position_dodge(width = 0.7), vjust = -0.5, size = 3.5) +
    geom_errorbar(aes(ymin = AI_Baseline, ymax = AI_Baseline),
                  position = position_dodge(width = 0.7),
                  width = 0.7, color = "red", linetype = "dashed", linewidth = 0.8) +
    geom_segment(data = levels_data,
                 aes(x = x_center - 0.175, xend = x_center + 0.175,
                     y = y_bracket, yend = y_bracket),
                 inherit.aes = FALSE, linewidth = 0.6) +
    geom_segment(data = levels_data,
                 aes(x = x_center - 0.175, xend = x_center - 0.175,
                     y = y_bracket, yend = y_bracket - 0.01),
                 inherit.aes = FALSE, linewidth = 0.6) +
    geom_segment(data = levels_data,
                 aes(x = x_center + 0.175, xend = x_center + 0.175,
                     y = y_bracket, yend = y_bracket - 0.01),
                 inherit.aes = FALSE, linewidth = 0.6) +
    geom_text(data = levels_data,
              aes(x = x_center, y = y_bracket + 0.015, label = p_label),
              inherit.aes = FALSE, size = 4, fontface = "bold") +
    scale_y_continuous(labels = percent, breaks = seq(0.6, 1.0, 0.1)) +
    coord_cartesian(ylim = c(0.6, 1.08)) +
    scale_fill_brewer(palette = "Set2") +
    labs(title = "Stratified Diagnostic Accuracy by Physician Seniority",
         subtitle = "P-values: between-arm within each stratum | Red dashed: AI baseline",
         x = "Physician Seniority", y = "Final Clinical Accuracy",
         fill = "Arm") +
    theme_minimal(base_size = 14)

  ggsave(paste0(output_dir, "fig_stratified_accuracy.png"),
         p_fig_strat, width = 10, height = 6, dpi = 300)

  write_csv(acc_stratified %>% left_join(p_stratified, by = "Doctor_Level"),
            paste0(output_dir, "stratified_accuracy_with_pvalues.csv"))
  write_csv(acc_overall %>% mutate(P_Value = p_overall),
            paste0(output_dir, "overall_accuracy_with_pvalues.csv"))

  cat("\nFigures and tables saved.\n")

  return(list(
    df_all         = df_all,
    acc_overall    = acc_overall,
    acc_stratified = acc_stratified,
    p_values       = p_stratified
  ))
}

# ==============================================================================
# Module 5: API Cost Analysis
# ==============================================================================

run_api_cost_analysis <- function(folder_path  = CONFIG$api_data_folder,
                                  prompt_s1    = CONFIG$prompt_s1,
                                  prompt_mmain = CONFIG$prompt_mmain,
                                  prompt_s3    = CONFIG$prompt_s3) {
  print_section("Module 5: API Cost Analysis")

  TOKEN_RATE     <- 2
  PRICE_DS_IN    <- 4.0
  PRICE_DS_OUT   <- 12.0
  PRICE_QWEN_IN  <- 1.0
  PRICE_QWEN_OUT <- 4.0

  file_paths <- list.files(path = folder_path, pattern = "\\.csv$", full.names = TRUE)
  data_list  <- setNames(lapply(file_paths, read.csv),
                         file_path_sans_ext(basename(file_paths)))
  cat("Loaded:", paste(names(data_list), collapse = ", "), "\n")

  len_s1   <- get_prompt_len(prompt_s1)
  len_main <- get_prompt_len(prompt_mmain)
  len_s3   <- get_prompt_len(prompt_s3)
  cat(sprintf("Prompt lengths — S1: %d chars, Mmain: %d chars, S3: %d chars\n",
              len_s1, len_main, len_s3))

  calc_cost <- function(df, input_cols, output_cols, method,
                        price_in, price_out, prompt_len) {
    df %>%
      rowwise() %>%
      mutate(
        tokens_in  = (sum(nchar(as.character(c_across(all_of(input_cols)))),
                          na.rm = TRUE) + prompt_len) * TOKEN_RATE,
        tokens_out = sum(nchar(as.character(c_across(all_of(output_cols)))),
                         na.rm = TRUE) * TOKEN_RATE
      ) %>%
      ungroup() %>%
      summarise(
        Method            = method,
        N                 = n(),
        Avg_Input_Tokens  = mean(tokens_in,  na.rm = TRUE),
        Avg_Output_Tokens = mean(tokens_out, na.rm = TRUE),
        Cost_Input_CNY    = sum(tokens_in,  na.rm = TRUE) / 1e6 * price_in,
        Cost_Output_CNY   = sum(tokens_out, na.rm = TRUE) / 1e6 * price_out,
        Total_Cost_CNY    = Cost_Input_CNY + Cost_Output_CNY,
        Cost_Per_Case_CNY = Total_Cost_CNY / N
      )
  }

  # Note: API log CSV columns are not renamed (raw inference logs)
  COMMON_IN  <- c("入院时间", "病例特点")
  COMMON_OUT <- c("final_label", "rationale_short", "reasoning_content", "evidence_json")

  results <- list()

  if ("cases_labeled_DS3.1_noPrompt_withThinking" %in% names(data_list)) {
    results$m1 <- calc_cost(
      data_list[["cases_labeled_DS3.1_noPrompt_withThinking"]],
      COMMON_IN, COMMON_OUT,
      "Mmain (DS-V3.1, baseline prompt)", PRICE_DS_IN, PRICE_DS_OUT, len_s1)
  }

  if ("cases_labeled_DS3.1_withPrompt_withThinking" %in% names(data_list)) {
    results$m2 <- calc_cost(
      data_list[["cases_labeled_DS3.1_withPrompt_withThinking"]],
      COMMON_IN, COMMON_OUT,
      "Mmain (DS-V3.1, optimized prompt)", PRICE_DS_IN, PRICE_DS_OUT, len_main)
  }

  if ("cases_checked_results_DS3.1_withPrompt_withThinking_using_Qwen32b_noThinking" %in% names(data_list)) {
    results$m3 <- calc_cost(
      data_list[["cases_checked_results_DS3.1_withPrompt_withThinking_using_Qwen32b_noThinking"]],
      c("final_label", "rationale_short"),
      c("has_contradiction", "analysis"),
      "Mcheck (Qwen3-32B, S3 checker)", PRICE_QWEN_IN, PRICE_QWEN_OUT, len_s3)
  }

  if (length(results) > 0) {
    cost_table <- bind_rows(results)
    print(kable(cost_table, digits = 4, caption = "API Cost Analysis (CNY)"))

    if (!is.null(results$m2) && !is.null(results$m3)) {
      cost_scout <- results$m2$Cost_Per_Case_CNY + results$m3$Cost_Per_Case_CNY
      cat(sprintf("\n=== Cost Summary ===\nSCOUT pipeline (Mmain + Mcheck) per case: CNY %.4f\n",
                  cost_scout))
    }
    return(cost_table)
  } else {
    cat("Required data files not found. Check CONFIG paths.\n")
    return(NULL)
  }
}

# ==============================================================================
# Main entry point
# ==============================================================================

run_modules_3_5 <- function() {
  print_section("SCOUT Multi-Reader Analysis — Modules 3-5")
  cat("Note: update CONFIG paths before running.\n\n")
  # doctor_stats <- run_physician_review_stats()
  # stratified   <- run_stratified_analysis()
  # cost_table   <- run_api_cost_analysis()
  cat("\nDone. Each module can be called independently.\n")
}

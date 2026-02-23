# ==============================================================================
#   SCOUT Multi-Reader Crossover Trial — Modules 1–2
#   Module 1: Uncertainty Triangulation (S1 ∪ S2 ∪ S3)
#   Module 2: Randomized Case-Set Generation (Control / Intervention)
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
  gold_file  = "data/gold_standard_multi_reader.csv",
  output_dir = "output/multi_reader/",
  seed       = 2025
)

# --- Utility functions --------------------------------------------------------

clean_text <- function(x) {
  x <- trimws(toupper(as.character(x)))
  x[x == "" | x == "NA" | x == "NULL" | is.na(x)] <- NA
  return(x)
}

#' Check whether the S3 reasoning-critique verdict indicates a pass
is_pass <- function(text) {
  pass_keywords <- c("不存在", "一致", "无矛盾", "None", "No", "Pass", "Consistent")
  if (is.na(text) || text == "") return(FALSE)
  any(sapply(pass_keywords, function(w) grepl(w, text, ignore.case = TRUE)))
}

print_section <- function(title) {
  cat("\n", strrep("=", 70), "\n  ", title, "\n", strrep("=", 70), "\n")
}

# ==============================================================================
# Module 1: Uncertainty Triangulation Analysis
# ==============================================================================

run_triangulation_stats <- function(file_path = CONFIG$gold_file) {
  print_section("Module 1: Uncertainty Triangulation (S1 ∪ S2 ∪ S3)")

  data <- read.csv(file_path, stringsAsFactors = FALSE)

  analysis_df <- data %>%
    mutate(
      Main_Pred  = clean_text(final_label_main),
      S1_Pred    = clean_text(final_label_audit),
      S2_Pred    = clean_text(final_label_second),
      Gold       = clean_text(Gold_Label),

      Is_Error = (Main_Pred != Gold) | is.na(Main_Pred),

      # Three orthogonal verification signals
      Signal_S1 = ifelse(is.na(Main_Pred != S1_Pred), FALSE, Main_Pred != S1_Pred),
      Signal_S2 = ifelse(is.na(Main_Pred != S2_Pred), FALSE, Main_Pred != S2_Pred),
      Signal_S3 = ifelse(is.na(!sapply(audit_result, is_pass)), FALSE,
                         !sapply(audit_result, is_pass)),

      Flag_Union = Signal_S1 | Signal_S2 | Signal_S3
    )

  # --- Metrics ----------------------------------------------------------------
  N           <- nrow(analysis_df)
  N_errors    <- sum(analysis_df$Is_Error, na.rm = TRUE)
  review_rate <- mean(analysis_df$Flag_Union, na.rm = TRUE)
  tp          <- sum(analysis_df$Is_Error & analysis_df$Flag_Union, na.rm = TRUE)
  coverage    <- tp / N_errors
  final_acc   <- 1 - ((N_errors - tp) / N)
  efficiency  <- coverage / (review_rate + 1e-5)
  baseline_acc <- 1 - (N_errors / N)

  cat(sprintf("Total cases: %d\n", N))
  cat(sprintf("Baseline errors: %d (Baseline accuracy: %.2f%%)\n",
              N_errors, baseline_acc * 100))
  cat("--------------------------------------------------------------\n")
  cat(sprintf("Review Rate:    %.4f (%.2f%%)\n", review_rate, review_rate * 100))
  cat(sprintf("Error Coverage: %.4f (%.2f%%)\n", coverage, coverage * 100))
  cat(sprintf("True Positives: %d / %d\n", tp, N_errors))
  cat("--------------------------------------------------------------\n")
  cat(sprintf(">>> Efficiency Ratio: %.4f\n", efficiency))
  cat(sprintf(">>> Final Accuracy:   %.4f (%.2f%%)\n", final_acc, final_acc * 100))

  cat("\n--- Signal Trigger Rates ---\n")
  cat(sprintf("S1 (Model Heterogeneity):      %d (%.2f%%)\n",
              sum(analysis_df$Signal_S1), mean(analysis_df$Signal_S1) * 100))
  cat(sprintf("S2 (Stochastic Inconsistency): %d (%.2f%%)\n",
              sum(analysis_df$Signal_S2), mean(analysis_df$Signal_S2) * 100))
  cat(sprintf("S3 (Reasoning Critique):       %d (%.2f%%)\n",
              sum(analysis_df$Signal_S3), mean(analysis_df$Signal_S3) * 100))
  cat(sprintf("Union (S1 ∪ S2 ∪ S3):         %d (%.2f%%)\n",
              sum(analysis_df$Flag_Union), mean(analysis_df$Flag_Union) * 100))

  err_df <- filter(analysis_df, Is_Error)
  cat("\n--- Error Capture by Signal ---\n")
  cat(sprintf("S1 captured: %d / %d\n", sum(err_df$Signal_S1), nrow(err_df)))
  cat(sprintf("S2 captured: %d / %d\n", sum(err_df$Signal_S2), nrow(err_df)))
  cat(sprintf("S3 captured: %d / %d\n", sum(err_df$Signal_S3), nrow(err_df)))
  cat(sprintf("Union captured: %d / %d\n", sum(err_df$Flag_Union), nrow(err_df)))

  silent <- analysis_df %>%
    filter(Is_Error & !Flag_Union) %>%
    select(Case_ID, Main_Pred, Gold, S1_Pred, S2_Pred, audit_result)

  if (nrow(silent) > 0) {
    cat(sprintf("\n>>> WARNING: %d silent failure(s) detected\n", nrow(silent)))
    print(silent)
  } else {
    cat("\n>>> No silent failures detected.\n")
  }

  return(analysis_df)
}

# ==============================================================================
# Module 2: Randomized Case-Set Generation
# ==============================================================================

run_case_set_split <- function(analysis_df,
                               output_dir = CONFIG$output_dir,
                               seed = CONFIG$seed) {
  print_section("Module 2: Randomized Case-Set Generation")
  set.seed(seed)

  analysis_df <- analysis_df %>%
    mutate(Stratify_Tag = paste(Is_Error, Flag_Union, sep = "_"))

  analysis_df$Split_ID <- runif(nrow(analysis_df))
  analysis_df <- analysis_df %>%
    group_by(Stratify_Tag) %>%
    mutate(Rank = rank(Split_ID),
           Group = ifelse(Rank <= n() / 2, "Control", "Intervention")) %>%
    ungroup()

  # Control set (standard manual review)
  control_set <- analysis_df %>%
    filter(Group == "Control") %>%
    mutate(Physician_Label = "", Physician_Note = "") %>%
    arrange(Case_ID) %>%
    select(Case_ID, Admission_Time, Clinical_Note,
           Mmain_Pred      = final_label_main,
           Mmain_Rationale = rationale_short_main,
           Mmain_Evidence  = evidence_json_main,
           Physician_Label, Physician_Note)

  # Intervention set (SCOUT-assisted review)
  intervention_set <- analysis_df %>%
    filter(Group == "Intervention") %>%
    mutate(
      SCOUT_Status    = ifelse(Flag_Union, "[REVIEW REQUIRED]", "AUTO-ACCEPTED"),
      Suggested_Label = ifelse(!Flag_Union, Main_Pred, ""),
      Physician_Final = Suggested_Label,
      Physician_Note  = ""
    ) %>%
    arrange(desc(Flag_Union), Case_ID) %>%
    select(Case_ID, Admission_Time, Clinical_Note,
           SCOUT_Status, Physician_Final, Physician_Note,
           Mmain_Pred      = final_label_main,
           Maux_Pred       = final_label_audit,
           S2_Pred_raw     = final_label_second,
           S3_Verdict      = audit_result,
           S3_Analysis     = audit_analysis,
           Mmain_Rationale = rationale_short_main,
           S2_Rationale    = rationale_short_second)

  write_xlsx(control_set,      paste0(output_dir, "control_set_standard_review.xlsx"))
  write_xlsx(intervention_set, paste0(output_dir, "intervention_set_SCOUT_assisted.xlsx"))

  cat(sprintf("Control set:      %d cases\n", nrow(control_set)))
  cat(sprintf("Intervention set: %d cases\n", nrow(intervention_set)))
  cat("\n--- Randomization Balance Check ---\n")
  print(table(analysis_df$Group, analysis_df$Is_Error))

  return(list(control_set = control_set,
              intervention_set = intervention_set,
              analysis_df = analysis_df))
}

# ==============================================================================
# Main entry point
# ==============================================================================

run_modules_1_2 <- function() {
  print_section("SCOUT Multi-Reader Analysis — Modules 1-2")
  cat("Note: update CONFIG paths before running.\n\n")
  analysis_df <- run_triangulation_stats()
  # packs <- run_case_set_split(analysis_df)
  cat("\nDone.\n")
}

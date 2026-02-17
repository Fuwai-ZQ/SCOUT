###############################################################################
# 09_scout_validation_analysis.R
# SCOUT framework validation on the MIMIC-IV CHD subtyping cohort
# Strategy: S1 (model heterogeneity) ∪ S2 (stochastic inconsistency) ∪ S3 (reasoning critique)
# Reference: "Efficacy of individual and combined verification strategies"
###############################################################################

rm(list = ls())
library(readxl)
library(dplyr)

# --- 1. File Paths ---
base_path <- "D:/Desktop/OCR论文/极限提高/论文/数据和代码整理/冠心病分型/MIMIC/调用大模型"

path_primary      <- file.path(base_path, "primary_model_predictions.csv")    # Mmain (CoT prompt)
path_auxiliary     <- file.path(base_path, "auxiliary_model_predictions.csv")  # S1: Maux (baseline prompt)
path_stochastic    <- file.path(base_path, "s2_stochastic_rerun.csv")         # S2: Mmain re-sampling
path_cot_audit     <- file.path(base_path, "chain_of_thought_audit.csv")      # S3: Mcheck audit
path_gold          <- file.path(base_path, "gold_standard_labels.xlsx")       # Gold-standard adjudication

output_results     <- file.path(base_path, "scout_analysis_results.csv")
output_undetected  <- file.path(base_path, "undetected_errors.csv")

# --- 2. Read Data ---
df_primary    <- read.csv(path_primary,   stringsAsFactors = FALSE, encoding = "UTF-8")
df_auxiliary  <- read.csv(path_auxiliary,  stringsAsFactors = FALSE, encoding = "UTF-8")
df_stochastic <- read.csv(path_stochastic, stringsAsFactors = FALSE, encoding = "UTF-8")
df_cot_audit  <- read.csv(path_cot_audit, stringsAsFactors = FALSE, encoding = "UTF-8")
gold_standard <- read_excel(path_gold)

# --- 3. Helper Functions ---
clean_label <- function(x) {
  x <- trimws(toupper(as.character(x)))
  x[x %in% c("", "NA", "NULL") | is.na(x)] <- NA
  return(x)
}

# Keywords indicating S3 passed (no contradiction detected)
safe_keywords <- c("Consistent", "No", "None", "Pass")

is_s3_pass <- function(text) {
  if (is.na(text) || text == "") return(FALSE)
  any(sapply(safe_keywords, function(w) grepl(w, text, ignore.case = TRUE)))
}

# --- 4. Data Preprocessing ---
df_gold <- gold_standard %>%
  select(hadm_id, gold_label = subtype) %>%
  mutate(hadm_id = as.character(hadm_id), gold_label = clean_label(gold_label)) %>%
  filter(!is.na(gold_label))

cat(sprintf("Gold-standard samples: %d\n", nrow(df_gold)))

df_main <- df_primary %>%
  select(hadm_id, main_pred = final_label) %>%
  mutate(hadm_id = as.character(hadm_id), main_pred = clean_label(main_pred))

df_s1 <- df_auxiliary %>%
  select(hadm_id, s1_pred = final_label) %>%
  mutate(hadm_id = as.character(hadm_id), s1_pred = clean_label(s1_pred))

df_s2 <- df_stochastic %>%
  select(hadm_id, s2_pred = final_label) %>%
  mutate(hadm_id = as.character(hadm_id), s2_pred = clean_label(s2_pred))

df_s3 <- df_cot_audit %>%
  select(hadm_id, audit_result = audit_has_contradiction) %>%
  mutate(
    hadm_id = as.character(hadm_id),
    flag_s3 = !sapply(audit_result, is_s3_pass)
  ) %>%
  select(hadm_id, flag_s3, audit_result)

# Merge all signals
merged <- df_gold %>%
  inner_join(df_main, by = "hadm_id") %>%
  inner_join(df_s1,   by = "hadm_id") %>%
  inner_join(df_s2,   by = "hadm_id") %>%
  inner_join(df_s3,   by = "hadm_id")

cat(sprintf("Merged sample size: %d\n", nrow(merged)))

# --- 5. SCOUT Deferral Logic ---
analysis <- merged %>%
  mutate(
    is_error = (main_pred != gold_label) | is.na(main_pred),

    # S1: model heterogeneity — flag when Mmain ≠ Maux
    flag_s1 = ifelse(is.na(main_pred != s1_pred), FALSE, main_pred != s1_pred),

    # S2: stochastic inconsistency — flag when re-sampled output differs
    flag_s2 = ifelse(is.na(main_pred != s2_pred), FALSE, main_pred != s2_pred),

    # S3: reasoning critique — already computed above
    flag_s3 = ifelse(is.na(flag_s3), FALSE, flag_s3),

    # Union deferral: D(x) = S1 ∪ S2 ∪ S3
    flag_union = flag_s1 | flag_s2 | flag_s3
  )

# --- 6. Compute Metrics (per paper definitions) ---
n_total       <- nrow(analysis)
n_errors      <- sum(analysis$is_error, na.rm = TRUE)
review_rate   <- mean(analysis$flag_union, na.rm = TRUE)
n_captured    <- sum(analysis$is_error & analysis$flag_union, na.rm = TRUE)
error_coverage <- n_captured / n_errors
final_accuracy <- 1 - ((n_errors - n_captured) / n_total)
efficiency_ratio <- error_coverage / (review_rate + 1e-5)
baseline_accuracy <- 1 - (n_errors / n_total)

# --- 7. Report ---
cat("\n==============================================================\n")
cat("  SCOUT Validation Report — MIMIC-IV CHD Subtyping\n")
cat("  Strategy: S1 ∪ S2 ∪ S3 (Union)\n")
cat("==============================================================\n")
cat(sprintf("Total cases:         %d\n", n_total))
cat(sprintf("Baseline accuracy:   %.2f%% (%d errors)\n", baseline_accuracy * 100, n_errors))
cat("--------------------------------------------------------------\n")
cat(sprintf("Review rate (RR):    %.2f%%\n", review_rate * 100))
cat(sprintf("Error coverage (EC): %.2f%% (%d / %d)\n", error_coverage * 100, n_captured, n_errors))
cat(sprintf("Efficiency ratio:    %.2f\n", efficiency_ratio))
cat(sprintf("Final accuracy:      %.2f%%\n", final_accuracy * 100))
cat("==============================================================\n")

# --- 8. Per-Strategy Signal Breakdown ---
cat("\n--- Signal Trigger Rates ---\n")
cat(sprintf("S1 (model heterogeneity):    %d (%.2f%%)\n",
            sum(analysis$flag_s1), mean(analysis$flag_s1) * 100))
cat(sprintf("S2 (stochastic inconsist.):  %d (%.2f%%)\n",
            sum(analysis$flag_s2), mean(analysis$flag_s2) * 100))
cat(sprintf("S3 (reasoning critique):     %d (%.2f%%)\n",
            sum(analysis$flag_s3), mean(analysis$flag_s3) * 100))
cat(sprintf("Union (S1 ∪ S2 ∪ S3):        %d (%.2f%%)\n",
            sum(analysis$flag_union), mean(analysis$flag_union) * 100))

# --- 9. Error Capture by Strategy ---
cat("\n--- Error Capture per Strategy ---\n")
errors_only <- analysis %>% filter(is_error)
cat(sprintf("S1 captured: %d / %d\n", sum(errors_only$flag_s1), nrow(errors_only)))
cat(sprintf("S2 captured: %d / %d\n", sum(errors_only$flag_s2), nrow(errors_only)))
cat(sprintf("S3 captured: %d / %d\n", sum(errors_only$flag_s3), nrow(errors_only)))
cat(sprintf("Union:       %d / %d\n", sum(errors_only$flag_union), nrow(errors_only)))

# --- 10. Silent Failures (undetected errors) ---
undetected <- analysis %>%
  filter(is_error & !flag_union) %>%
  select(hadm_id, main_pred, gold_label, s1_pred, s2_pred, audit_result)

if (nrow(undetected) > 0) {
  cat(sprintf("\n>>> %d silent failure(s) detected:\n", nrow(undetected)))
  print(undetected)
} else {
  cat("\n>>> All errors captured by SCOUT.\n")
}

# --- 11. Export ---
write.csv(analysis,   output_results,    row.names = FALSE)
write.csv(undetected, output_undetected, row.names = FALSE)
cat(sprintf("\nResults saved to:\n  %s\n  %s\n", output_results, output_undetected))

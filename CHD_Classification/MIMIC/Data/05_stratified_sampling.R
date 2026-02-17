###############################################################################
# 05_stratified_sampling.R
# Stratified random sampling of 2,000 cases for SCOUT validation
# Reference: Methods - "drew a random sample of 2,000 instances"
###############################################################################

rm(list = ls())
library(dplyr)

folder_path <- "D:/Desktop/OCR论文/极限提高/MIMIC/结果/"

# --- Load LLM classification results (4 batches) and gold-standard labels ---
df_results <- bind_rows(
  read.csv(paste0(folder_path, "CHD_Classification_Results1.csv"), stringsAsFactors = FALSE),
  read.csv(paste0(folder_path, "CHD_Classification_Results2.csv"), stringsAsFactors = FALSE),
  read.csv(paste0(folder_path, "CHD_Classification_Results3.csv"), stringsAsFactors = FALSE),
  read.csv(paste0(folder_path, "CHD_Classification_Results4.csv"), stringsAsFactors = FALSE)
)

df_gold <- read.csv(
  paste0(folder_path, "sampled_data_带标签.csv"),
  stringsAsFactors = FALSE
)
# Rename Chinese column
names(df_gold)[names(df_gold) == "分型"] <- "subtype"

# --- Compare model predictions against gold standard ---
comparison <- df_gold %>%
  select(hadm_id, subtype) %>%
  inner_join(df_results %>% select(hadm_id, final_label), by = "hadm_id") %>%
  mutate(match = (subtype == final_label)) %>%
  filter(final_label != "Insufficient Information")

cat("Eligible cases after exclusion:", nrow(comparison), "\n")
print(table(comparison$subtype))

# --- Stratified sampling (n = 2,000) proportional to subtype prevalence ---
set.seed(42)

target_distribution <- c(CCS = 0.4707, NSTEMI = 0.1840, STEMI = 0.0533, UA = 0.2920)
n_samples     <- 2000
target_counts <- round(target_distribution * n_samples)

# Adjust rounding difference
target_counts["CCS"] <- target_counts["CCS"] + (n_samples - sum(target_counts))
cat("Target counts per subtype:\n")
print(target_counts)

final_dataset <- comparison %>%
  group_by(subtype) %>%
  group_modify(~ {
    target_n    <- target_counts[[.y$subtype]]
    available_n <- nrow(.x)
    if (available_n >= target_n) {
      .x[sample(seq_len(available_n), target_n, replace = FALSE), ]
    } else {
      cat(sprintf("Warning: %s insufficient (%d < %d), all retained\n",
                  .y$subtype, available_n, target_n))
      .x
    }
  }) %>%
  ungroup()

# Shuffle
final_dataset <- final_dataset[sample(seq_len(nrow(final_dataset))), ]
cat("Final sample distribution:\n")
print(table(final_dataset$subtype))

# --- Merge original text and export LLM input ---
output_dataset <- final_dataset %>%
  left_join(df_gold %>% select(hadm_id, final_text), by = "hadm_id") %>%
  select(hadm_id, final_text)

write.csv(output_dataset, paste0(folder_path, "LLM_Input_with_HADM_ID.csv"), row.names = FALSE)
cat("Exported:", nrow(output_dataset), "cases\n")

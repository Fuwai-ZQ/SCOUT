###############################################################################
# 02_match_diagnoses_distribution.R
# Match filtered ICD codes against MIMIC-IV diagnoses; generate subtype distribution
# Reference: Task 1 - CHD subtyping, MIMIC-IV cohort
###############################################################################

rm(list = ls())
library(data.table)

work_dir <- "C:/Users/ba143/Desktop/MIMIC数据库"

# --- Load data ---
clean_icd  <- fread(file.path(work_dir, "final_clean_icd_list.csv"))
diagnoses  <- fread(file.path(work_dir, "diagnoses_icd.csv"))

# Trim whitespace from ICD codes
diagnoses[, icd_code := trimws(icd_code)]

# --- Match diagnoses to CHD ICD list ---
target_rows    <- diagnoses[icd_code %in% clean_icd$icd_code]
target_hadm_ids <- unique(target_rows$hadm_id)
cat("Matched admissions:", length(target_hadm_ids), "\n")

# --- Merge with ICD descriptions ---
merged_data <- merge(target_rows, clean_icd, by = "icd_code", all.x = TRUE)

# --- Compute diagnosis distribution ---
distribution <- merged_data[, .N, by = long_title][order(-N)]
distribution[, percent := round(N / sum(N) * 100, 2)]
cat("--- Top 20 CHD Diagnoses ---\n")
print(head(distribution, 20))

# --- Export distribution ---
# fwrite(distribution, file.path(work_dir, "diagnosis_distribution.csv"))

# --- Assign CHD subtypes from curated classification table ---
subtype_map <- read.csv(
  file.path(work_dir, "diagnosis_distribution_selected.csv"),
  fileEncoding = "GBK"
)
# Rename Chinese column for consistency
names(subtype_map)[names(subtype_map) == "分型"] <- "subtype"

merged_data <- merge(
  merged_data,
  subtype_map[, c("long_title", "subtype")],
  by = "long_title",
  all.x = TRUE
)

cat("Subtype distribution:\n")
print(table(merged_data$subtype, useNA = "ifany"))

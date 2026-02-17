###############################################################################
# 01_filter_chd_icd_codes.R
# Filter coronary heart disease (CHD)-related ICD codes from MIMIC-IV
# Reference: Task 1 - Complex diagnostic reasoning (CHD subtyping)
###############################################################################

rm(list = ls())
library(data.table)

work_dir <- "C:/Users/ba143/Desktop/MIMIC数据库"

# --- Load MIMIC-IV tables ---
d_icd      <- fread(file.path(work_dir, "d_icd_diagnoses.csv"))
diagnoses  <- fread(file.path(work_dir, "diagnoses_icd.csv"))

# --- Step 1: Include target CHD diagnoses ---
target_keywords <- "Coronary|Angina|Myocardial infarction|Ischemic heart"
step1_include   <- d_icd[grep(target_keywords, long_title, ignore.case = TRUE)]

# --- Step 2: Exclude non-CHD and secondary entries ---
exclude_primary <- paste0(
  "Herpangina|Vincent's|",
  "Poisoning|Adverse effect|Underdosing|",
  "History|Screening|Status|Presence|",
  "Complication|Malformation|Anomaly"
)
step2_filtered <- step1_include[!grepl(exclude_primary, long_title, ignore.case = TRUE)]

# --- Step 3: Remove residual non-target entries ---
exclude_secondary <- paste0(
  "Breakdown|Displacement|Leakage|Stenosis|",
  "Abnormal findings|",
  "defect|",
  "sequelae"
)
final_icd_list <- step2_filtered[!grepl(exclude_secondary, long_title, ignore.case = TRUE)]

cat("After primary filter:", nrow(step2_filtered), "\n")
cat("After secondary filter:", nrow(final_icd_list), "\n")

# --- Export ---
# fwrite(final_icd_list, file.path(work_dir, "final_clean_icd_list.csv"))

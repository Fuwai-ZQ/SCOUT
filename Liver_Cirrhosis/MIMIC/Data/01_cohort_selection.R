# ==============================================================================
# Step 1: Identify cirrhosis patients from MIMIC-IV ICD diagnoses
# ==============================================================================

rm(list = ls())
library(data.table)
library(dplyr)

work_dir <- "path/to/mimic-iv"

d_icd      <- fread(file.path(work_dir, "d_icd_diagnoses.csv"))
diagnoses  <- fread(file.path(work_dir, "diagnoses_icd.csv"))

# Filter ICD codes containing "Cirrhosis"
target_icd <- d_icd %>%
  filter(grepl("Cirrhosis", long_title, ignore.case = TRUE)) %>%
  select(icd_code, icd_version, long_title)

# Match with diagnosis records
result_df <- diagnoses %>%
  inner_join(target_icd, by = c("icd_code", "icd_version")) %>%
  select(hadm_id, subject_id, icd_code, icd_version, long_title) %>%
  distinct()

cat("Unique subjects:", n_distinct(result_df$subject_id), "\n")
cat("Unique admissions:", n_distinct(result_df$hadm_id), "\n")

# write.csv(result_df, "cirrhosis_diagnoses_detail.csv", row.names = FALSE)

# ==============================================================================
# Step 2: Enumerate CT exam types from radiology records
# ==============================================================================

library(readr)

radiology        <- read_csv(file.path(work_dir, "radiology.csv"))
radiology_detail <- read_csv(file.path(work_dir, "radiology_detail.csv"))

ct_exams <- radiology_detail %>%
  filter(field_name == "exam_name", grepl("^CT ", field_value))

ct_exam_freq <- ct_exams %>%
  group_by(field_value) %>%
  summarise(frequency = n(), .groups = "drop") %>%
  arrange(desc(frequency))

print(head(ct_exam_freq, 10))

# write.csv(ct_exam_freq, "ct_exam_frequency_table.csv", row.names = FALSE)

# ==============================================================================
# Step 3: Link cirrhosis patients with abdominal CT exams
# ==============================================================================

target_exams <- c(
  "CT ABD & PELVIS WITH CONTRAST",
  "CT ABDOMEN W/CONTRAST",
  "CT ABD & PELVIS W/O CONTRAST",
  "CT ABD W&W/O C",
  "CT ABDOMEN W/O CONTRAST",
  "CT ABD & PELVIS W & W/O CONTRAST, ADDL SECTIONS"
)

valid_notes <- radiology_detail %>%
  filter(field_name == "exam_name", field_value %in% target_exams) %>%
  select(note_id, field_value) %>%
  rename(exam_name_detail = field_value)

final_radiology_records <- radiology %>%
  filter(hadm_id %in% result_df$hadm_id) %>%
  inner_join(valid_notes, by = "note_id")

cat("Matched radiology records:", nrow(final_radiology_records), "\n")
cat("Unique patients:", n_distinct(final_radiology_records$subject_id), "\n")

# ==============================================================================
# Step 4: Classify scan type (enhanced vs. plain)
# ==============================================================================

enhanced_names <- c(
  "CT ABD & PELVIS WITH CONTRAST",
  "CT ABDOMEN W/CONTRAST",
  "CT ABD W&W/O C",
  "CT ABD & PELVIS W & W/O CONTRAST, ADDL SECTIONS"
)

plain_names <- c(
  "CT ABD & PELVIS W/O CONTRAST",
  "CT ABDOMEN W/O CONTRAST"
)

final_radiology_records <- final_radiology_records %>%
  mutate(scan_type = case_when(
    exam_name_detail %in% enhanced_names ~ "Enhanced",
    exam_name_detail %in% plain_names    ~ "Plain",
    TRUE                                 ~ "Other"
  ))

scan_summary <- final_radiology_records %>%
  group_by(scan_type) %>%
  summarise(
    record_count         = n(),
    unique_subject_count = n_distinct(subject_id),
    .groups = "drop"
  )
print(scan_summary)

# Export
enhanced_ct_records <- final_radiology_records %>% filter(scan_type == "Enhanced")
plain_ct_records    <- final_radiology_records %>% filter(scan_type == "Plain")

save(enhanced_ct_records, file = "cirrhosis_enhanced_ct.RData")
save(plain_ct_records,    file = "cirrhosis_plain_ct.RData")

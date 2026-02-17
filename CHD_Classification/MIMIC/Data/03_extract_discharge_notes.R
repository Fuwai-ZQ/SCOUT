###############################################################################
# 03_extract_discharge_notes.R
# Extract discharge notes for matched CHD admissions with subtype labels
# Priority rule: STEMI > NSTEMI > UA > CCS (per clinical severity)
###############################################################################

library(data.table)
library(readr)
library(dplyr)

work_dir <- "C:/Users/ba143/Desktop/MIMIC数据库"

# --- Build labelled cohort with priority-based deduplication ---
# When multiple CHD codes exist per admission, retain the most severe subtype
priority_map <- c("STEMI" = 1, "NSTEMI" = 2, "UA" = 3, "CCS" = 4)

target_cohort <- merged_data[subtype %in% names(priority_map)]
target_cohort[, priority_score := priority_map[subtype]]
setorder(target_cohort, hadm_id, priority_score)
unique_labels <- unique(target_cohort, by = "hadm_id")

whitelist <- unique_labels[, .(hadm_id, subtype, long_title)]
cat("Labelled cohort size:", nrow(whitelist), "\n")

# --- Chunked reading of discharge.csv with inner join ---
process_chunk <- function(chunk, pos) {
  dt <- as.data.table(chunk)
  matched <- merge(dt, whitelist, by = "hadm_id")
  return(matched)
}

cat("Reading and matching discharge notes...\n")
final_notes <- read_csv_chunked(
  file      = file.path(work_dir, "discharge.csv"),
  callback  = DataFrameCallback$new(process_chunk),
  chunk_size = 10000,
  col_types = cols(
    hadm_id  = col_double(),
    text     = col_character(),
    .default = col_skip()
  )
)

setDT(final_notes)
cat("Extracted records:", nrow(final_notes), "\n")

# --- Export ---
fwrite(final_notes, file.path(work_dir, "discharge_notes_labelled.csv"))

###############################################################################
# 04_extract_clinical_sections.R
# Parse discharge notes into structured LLM input (Service / HPI / Objective)
# Filter to cardiothoracic service; split into batches for batch inference
###############################################################################

rm(list = ls())
library(tidyverse)

# --- Load labelled discharge notes ---
data <- read.csv(
  "D:/Desktop/OCR论文/极限提高/MIMIC/原始数据/discharge_notes_labelled.csv",
  stringsAsFactors = FALSE
)

# --- Extract clinical sections ---
df_llm_input <- data %>%
  mutate(
    service_part  = str_extract(text, "(?i)Service:\\s*[^\\n]+"),
    service_clean = str_remove(service_part, "(?i)Service:\\s*"),

    hpi_part  = str_extract(text, "(?si)History of Present Illness:.*?(?=Past Medical History:)"),
    hpi_clean = str_remove(hpi_part, "(?i)History of Present Illness:\\s*"),

    results_part = str_extract(text, paste0(
      "(?si)Pertinent Results:.*?(?=(?i)(Brief Hospital Course|",
      "Discharge Medications|Transitional Issues|Acute Issues|",
      "Assessment|Impression)|$)"
    )),

    # Concatenate into structured prompt input
    final_text = paste0(
      "--- SERVICE ---\n", str_trim(coalesce(service_clean, "Not Specified")),
      "\n\n--- HISTORY ---\n", str_trim(hpi_clean),
      "\n\n--- OBJECTIVE DATA ---\n", str_trim(coalesce(results_part, "No structured results found."))
    )
  ) %>%
  select(hadm_id, service_clean, final_text, subtype)

# --- Filter: cardiothoracic admissions only (per Methods) ---
df_llm_input <- df_llm_input %>%
  filter(service_clean == "CARDIOTHORACIC")

cat("Cardiothoracic cases:", nrow(df_llm_input), "\n")
table(df_llm_input$subtype)

# --- Export full dataset (with gold-standard labels) ---
write.csv(df_llm_input, "sampled_data.csv", row.names = FALSE)

# --- Split into batches for batch inference (labels removed) ---
df_nolabel <- df_llm_input %>% select(-subtype)

batch_size <- 900
n_total    <- nrow(df_nolabel)
n_batches  <- ceiling(n_total / batch_size)

for (i in seq_len(n_batches)) {
  start_row <- (i - 1) * batch_size + 1
  end_row   <- min(i * batch_size, n_total)
  write.csv(
    df_nolabel[start_row:end_row, ],
    sprintf("sampled_data_%d.csv", i),
    row.names = FALSE
  )
}

cat("Exported", n_batches, "batch files.\n")

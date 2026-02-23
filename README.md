# SCOUT: Scalable Clinical Oversight via Uncertainty Triangulation

[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Trial Registration](https://img.shields.io/badge/ClinicalTrials.gov-NCT07414966-green.svg)](https://clinicaltrials.gov/study/NCT07414966)

This repository contains the code, de-identified data, and statistical analysis scripts for the paper:

> **Scalable clinical oversight of large language models via uncertainty triangulation**
>
> Zhengqing Ba\*, Ming He\*, Haorui He, Qingan Fu, Jingrong Lai, Ruiqi Zhang, Xiaolin Diao, Mengyuan Liu, Zuoxiang Wang, Ximei Wang, Sheng Zhao, Yalin Zhu, Henghongyu Chen, Yuanwang Qiu, Qilin Su, Jinghang Xu, Fenghuan Hu, Xinlin Luo, Hongwu Chen, Mingqi Zheng, Bing Xu, Jiabao Liu, Ning Guo, Xiaojin Gao†, Guiqiang Wang†, & Yongjian Wu†
>
> \* Equal contribution; † Corresponding authors

---

## Overview

SCOUT is a **model-agnostic meta-verification framework** that selectively defers unreliable LLM predictions to clinicians by triangulating three orthogonal uncertainty signals:

| Strategy | Signal | Mechanism |
|----------|--------|-----------|
| **S1** — Model Heterogeneity | Cross-model disagreement | An auxiliary model independently classifies the same case; disagreement triggers deferral |
| **S2** — Stochastic Inconsistency | Within-model instability | The main model is re-run under identical settings; divergent outputs indicate decision-boundary uncertainty |
| **S3** — Reasoning Critique | Chain-of-thought audit | A checker model evaluates the logical soundness of the main model's reasoning trace |

These signals are combined via boolean operators (union/intersection) and optimized through Pareto search to balance **physician workload** against **diagnostic safety**.

### Key Results

- Reduced physician review volume by **45.5–83.3%** across three clinical tasks, with theoretical final accuracy of **99.1–100.0%**.
- In a prospective multi-reader randomized crossover trial (7 clinicians, 110 cases), SCOUT-assisted review reduced mean review time by **59.3%** (0.37 vs. 0.92 min/case; *P* = 0.016) with no significant difference in diagnostic accuracy (90.8% vs. 89.4%; *P* = 0.596).

---

## Repository Structure

```
.
├── CHD_Classification/                # Task 1: Coronary Heart Disease Subtyping
│   ├── AIM_CHD_SMART_CHD/             # Discovery & prospective cohorts
│   │   ├── Retrospective_Study/
│   │   │   ├── raw_data/              # De-identified case records & gold-standard labels
│   │   │   └── data_LLM_analyzed/
│   │   │       ├── S1/                # Strategy 1 outputs (multi-model predictions)
│   │   │       ├── S2/                # Strategy 2 outputs (stochastic re-runs)
│   │   │       └── S3/                # Strategy 3 outputs (reasoning audits)
│   │   ├── LLM_API_Calls/             # Inference scripts for all models
│   │   └── External_Validation_Reader_Study/
│   │       ├── Data/                  # Multi-reader trial data (control & intervention)
│   │       └── Statistical_Results/   # R scripts for trial analysis
│   └── MIMIC/                         # MIMIC-IV cross-lingual validation
│       ├── Data/                      # Cohort selection & data extraction scripts
│       ├── LLM_API_Calls/             # Model inference & audit scripts
│       └── Statistical_Results/       # Validation analysis
│
├── Liver_Cirrhosis/                   # Task 2: Liver Cancer Screening
│   ├── Chinese_CT_Data/               # PORTAL cohort (two Chinese centres)
│   │   ├── Data/                      # CT report processing pipeline
│   │   ├── LLM_Execution_Results/     # Prediction & audit outputs
│   │   └── Statistical_Results/       # Accuracy analysis
│   └── MIMIC/                         # MIMIC-IV liver cohort
│       ├── Data/                      # Cohort selection & report parsing
│       ├── LLM_Execution_Results/     # Baseline, optimized & audit results
│       └── Statistical_Results/       # Joint accuracy analysis
│
├── Vessel_Disease_Count/              # Task 3: Diseased Vessel Counting
│   ├── raw_data/                      # Angiography reports & gold labels
│   ├── LLM_results/                   # Model predictions & audits
│   └── statistic_analysis/            # Validation script
│
└── Manuscript_Figures/                # Scripts for reproducing all figures & tables
    ├── Figure2.py
    ├── Figure3.py
    ├── Figure4.py
    └── Table1.py
```

---

## Getting Started

### Prerequisites

- **Python** ≥ 3.11
- **R** ≥ 4.4.1

### Python Dependencies

```bash
pip install pandas numpy scipy openpyxl matplotlib seaborn scikit-learn openai
```

### R Dependencies

```r
install.packages(c("lme4", "lmerTest", "boot", "dplyr", "readxl", "ggplot2"))
```

### API Access

LLM inference scripts require API keys for the following providers:

| Model | Provider | Endpoint |
|-------|----------|----------|
| DeepSeek-V3.1 / V3.2 | Aliyun Bailian | https://bailian.console.aliyun.com |
| Qwen3 (235B / 32B / 8B) | Aliyun Bailian | https://bailian.console.aliyun.com |
| GPT-5.1 | OpenAI | https://platform.openai.com |

Set your API key as an environment variable before running inference scripts:

```bash
export OPENAI_API_KEY="your-key-here"
export DASHSCOPE_API_KEY="your-key-here"
```

---

## Reproducing the Results

### 1. Retrospective Validation (Tasks 1–3)

**Task 1 — CHD Subtyping (Discovery Cohort)**

```bash
# Step 1: Run primary model inference (S1)
python CHD_Classification/AIM_CHD_SMART_CHD/LLM_API_Calls/coronary_classification_withPrompt_withThinking.py

# Step 2: Run stochastic re-sampling (S2)
# Re-execute the same script; outputs are saved in S2/run_2/

# Step 3: Run reasoning critique (S3)
python CHD_Classification/AIM_CHD_SMART_CHD/LLM_API_Calls/cot_audit_Qwen32b_withThinking.py
```

**Task 2 — Liver Cancer Screening (MIMIC-IV)**

```bash
cd Liver_Cirrhosis/MIMIC/
python Data/01_cohort_selection.R        # Requires MIMIC-IV access
python LLM_Execution_Results/06_predict_baseline_prompt\(S1\).py
python LLM_Execution_Results/07_predict_optimized_prompt\(S2\).py
python LLM_Execution_Results/08_reasoning_audit\(S3\).py
python Statistical_Results/09_joint_accuracy_analysis.py
```

**Task 3 — Vessel Counting**

```bash
cd Vessel_Disease_Count/
python LLM_results/vessel_count_main_inference.py
python LLM_results/vessel_count_auxiliary_inference.py
python LLM_results/vessel_count_reasoning_critique.py
python statistic_analysis/vessel_count_validation.py
```

### 2. Multi-Reader Randomized Crossover Trial

```bash
cd CHD_Classification/AIM_CHD_SMART_CHD/External_Validation_Reader_Study/Statistical_Results/
Rscript multi_reader_baseline_balance.R   # Baseline balance check
Rscript multi_reader_modules_1_2.R        # Primary & secondary endpoints
Rscript multi_reader_modules_3_5.R        # Subgroup & economic analyses
```

### 3. Manuscript Figures and Tables

```bash
cd Manuscript_Figures/
python Table1.py     # Table 1: Boolean strategy combinations
python Figure2.py    # Figure 2: Individual strategy efficacy
python Figure3.py    # Figure 3: Pareto optimization & generalizability
python figure4.py    # Figure 4: Multi-reader trial results
```

---

## Data Availability

| Dataset | Access |
|---------|--------|
| **MIMIC-IV** | Freely available via [PhysioNet](https://physionet.org/content/mimiciv/3.1/) after credentialing |
| **AIM-CHD / SMART-CHD / PORTAL** | Available from the corresponding authors upon reasonable request |
| **Multi-reader trial data** | De-identified data included in `External_Validation_Reader_Study/Data/` |

---


## License

This project is licensed under the [MIT License](LICENSE).

---

## Contact

For questions or data requests, please contact:

- **Xiaojin Gao** — Department of Cardiology, Fuwai Hospital
- **Guiqiang Wang** — Department of Infectious Disease, Peking University First Hospital
- **Yongjian Wu** — Department of Cardiology, Fuwai Hospital

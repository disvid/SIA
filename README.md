# Support Integrity Auditor (SIA)

A self-supervised Machine Learning pipeline that detects **Priority Mismatch** in CRM support tickets — cases where the human-assigned priority conflicts with the ticket's actual severity.

The system infers ticket severity from multiple signals, automatically generates pseudo-labels, trains a supervised classifier, and produces explainable evidence dossiers for flagged tickets.

---

# Features

✅ Self-supervised pseudo-label generation

✅ Multi-signal severity inference

✅ Sentence embeddings using SBERT

✅ KMeans severity clustering

✅ TF-IDF text features

✅ Automatic model selection using Cross Validation

✅ Explainable evidence dossiers

✅ Streamlit dashboard for interactive analysis

✅ Batch CSV prediction support

---

# Dataset

Dataset Source:

https://www.kaggle.com/datasets/ajverse/customer-support-tickets-crm-dataset

### Columns Used

| Column | Description |
|-------|-------------|
| Ticket_Subject | Short issue summary |
| Ticket_Description | Detailed ticket text |
| Issue_Category | Technical / Billing / Account |
| Priority_Level | Human-assigned priority |
| Ticket_Channel | Email / Chat / Phone |
| Resolution_Time_Hours | Resolution duration |
| Satisfaction_Score | Customer satisfaction |

---

# Pipeline Overview

```text
Raw Tickets (CSV)

      │
      ▼

┌──────────────────────────────┐
│ Stage 1                      │
│ Pseudo Label Generation      │
└──────────────────────────────┘

Signal 1 (40%)
• NLP Content Analysis

Signal 2 (30%)
• SBERT Embeddings
• KMeans Clustering

Signal 3 (20%)
• Resolution Time

Signal 4 (10%)
• Satisfaction Score

↓

Inferred Severity

↓

Compare with Priority_Level

↓

Mismatch Label (0/1)

      │
      ▼

┌──────────────────────────────┐
│ Stage 2                      │
│ Classifier Training          │
└──────────────────────────────┘

Features:

• TF-IDF (3000 features)
• Category × Priority
• Severity scores
• Content signal flags

Models:

• Logistic Regression
• Gradient Boosting
• Random Forest

↓

Best Model selected by
5-fold CV Macro F1

      │
      ▼

┌──────────────────────────────┐
│ Stage 3                      │
│ Evidence Dossier             │
└──────────────────────────────┘

For every flagged ticket:

• Assigned priority
• Inferred severity
• Severity delta
• Supporting evidence
• Confidence score
• Constraint analysis
```

---

# Pseudo Label Strategy

The key insight:

> Ticket descriptions contain the strongest signal of true severity.

The pipeline:

1. Cleans ticket descriptions.
2. Removes filler text.
3. Matches severity patterns using regex.
4. Generates inferred severity.
5. Compares against assigned priority.
6. Generates mismatch labels automatically.

---

## Mismatch Types

### Hidden Crisis

A ticket is **under-prioritized**.

Example:

- Assigned Priority: Low
- Inferred Severity: High

---

### False Alarm

A ticket is **over-prioritized**.

Example:

- Assigned Priority: High
- Inferred Severity: Low

---

# Ablation Study

| Signal Set | Accuracy | Macro F1 |
|-----------|----------|----------|
| NLP Only | ~0.82 | ~0.81 |
| Embedding Only | ~0.68 | ~0.66 |
| Resolution Time Only | ~0.55 | ~0.54 |
| Satisfaction Only | ~0.52 | ~0.51 |
| NLP + Embedding | ~0.87 | ~0.86 |
| NLP + Resolution Time | ~0.84 | ~0.83 |
| NLP + Satisfaction | ~0.83 | ~0.82 |
| NLP + Embed + Resolution | ~0.88 | ~0.87 |
| **All 4 Signals** | **≥0.89** | **≥0.88** |

---

# Model Performance

| Metric | Required | Achieved |
|-------|----------|----------|
| Accuracy | ≥ 0.83 | ✅ |
| Macro F1 | ≥ 0.82 | ✅ |
| Recall (Consistent) | ≥ 0.78 | ✅ |
| Recall (Mismatch) | ≥ 0.78 | ✅ |

---

# Running the Project

## 1. Train

```bash
python train_pipeline.py
```

---

## 2. Predict

```bash
python predict.py \
--input data/enhanced_customer_support_data.csv \
--output results/predictions.csv
```

Outputs:

- `results/predictions.csv`
- `results/predictions_dossiers.json`

---

## 3. Launch Streamlit App

```bash
streamlit run app.py
```

Open:

```text
http://localhost:8501
```

---

# Streamlit Pages

| Page | Description |
|------|-------------|
| Dashboard | KPIs and charts |
| Single Ticket | Analyze one ticket |
| Batch Upload | Upload CSV and download results |
| Model Metrics | Performance and ablation |
| Adversarial Tests | Stress test the model |

---

# Tech Stack

| Component | Library |
|----------|---------|
| Embeddings | sentence-transformers |
| Clustering | sklearn KMeans |
| Classifier | LogisticRegression / GBM / RandomForest |
| Text Features | TfidfVectorizer |
| Web App | Streamlit |
| Visualizations | Plotly |
| Persistence | Joblib |

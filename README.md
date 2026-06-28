## Project Overview

### Objectives
- Prepare and preprocess customer support ticket datasets.
- Construct a human-annotated gold-standard dataset.
- Compare two LLMs with a traditional lexicon-based sentiment analysis model (VADER).

### Data Cleaning
During data cleaning, we found that the ticket descriptions contained non-standard text, including:

- Non-English symbols and special characters
- HTML tags (e.g., <br/>)
- Line breaks (\n, \\n)
- Extra spaces

To prepare the data for sentiment classification, the following preprocessing steps were applied:

1. Replaced newline characters with spaces.
2. Removed HTML tags.
3. Converted all text to lowercase.
4. Removed abnormal characters while preserving English letters, numbers, and common punctuation.
5. Removed unnecessary and duplicate spaces.

**Traditional NLP preprocessing (e.g., stop word removal, tokenisation, and lemmatisation) was not applied. The original sentence structure was preserved because LLMs rely on contextual information for sentiment classification.**


### Pipeline

```text
Customer Support Tickets
          │
          ▼
Data Preprocessing
          │
          ▼
Gold Standard Construction
(Stratified Sampling + Manual Annotation)
          │
          ▼
VADER Baseline
          │
          ▼
LLM Evaluation
(Llama 3.1 vs Llama 3.3)
          │
          ▼
Performance Evaluation
(Precision • Recall • F1-score)
```

### Project Structure
The following diagram shows the complete data processing and model evaluation pipeline.
```
customer_support_tickets.csv
          │
          ▼
01_data_preprocessing.py
          │
          ▼
combined_tickets_clean_text.csv
          │
          ▼
02_gold_standard_sampling_and_vader_baseline.py
          │
          ▼
gold_set_stratified_by_ticket_type.csv
          │
          ▼
Manual annotation
          │
          ▼
gold_set_labeled.csv
          ├──────────────┐
          ▼              ▼
vader_predictions.csv    03_sentiment_classification_and_llm_evaluation.py
                             │
                             ▼
                         outputs/
                         ├── gold_set_with_llm_labels.csv
                         ├── vader_predictions.csv
                         ├── llama31_predictions.csv
                         ├── llama33_predictions.csv
                         └── metrics_comparison.txt
```

### Evaluation Metrics
The models were evaluated using a manually labelled gold-standard dataset.

Evaluation metrics:
- Recall (Negative)
- Precision (Negative)
- F1-score (Negative)

Since this sentiment classification was developed for a customer support AI-Agent, the objective was to **identify negative tickets** that should be prioritised for escalation to human agents. Therefore, a **binary sentiment classification (Negative vs. Non-Negative)** was adopted, and the evaluation focused on the negative class using Recall, Precision, and F1-score.

### Results 
Model	Recall (Neg)	Precision (Neg)	F1 (Neg)
Baseline (Vader)	0.2577	0.8933	0.4
LLM A (llama-3.1-8b-instant)	0.9423	0.8813	0.9108
LLM B (llama-3.3-70b-versatile)	0.7615	0.9706	0.8534
<img width="869" height="181" alt="image" src="https://github.com/user-attachments/assets/d9b760cd-89b4-4a3f-86d3-dea7d2ab76b4" />

### Skills
- Python | Pandas | LLM | Sentiment Classification | Prompt Engineering | VADER | Machine Learning | Data Cleaning | Model Evaluation

"""
This script:
1. Creates a stratified gold-standard dataset (n=500) by ticket type.
2. Applies a VADER lexicon-based sentiment baseline.
3. Evaluates baseline performance using manually labeled sentiment data.

The baseline results are later compared with LLM-based sentiment classifiers.
"""

import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from sklearn.metrics import precision_score, recall_score, f1_score

# -----Section 1: Stratified Sampling----- 
# Create a 500-sample Gold Standard Set

df = pd.read_csv("combined_tickets_clean_text.csv")

# Add a unique ticket ID
df = df.reset_index().rename(columns={"index": "ticket_id"})

# 1.3 Stratified sampling by ticket_type (target size = 500)
N = len(df)
target_n = 500

gold_list = []

for ttype, group in df.groupby('ticket_type'):
    # Proportion of this ticket type in the full dataset
    p = len(group) / N
    
    # Number of samples to draw for this type
    k = int(round(target_n * p))
    
    # Ensure at least one sample per type
    k = max(1, k)
    
    # Randomly sample tickets from this type
    sampled = group.sample(n=min(k, len(group)), random_state=42)
    
    gold_list.append(sampled)
    
# Combine samples from all ticket types
gold_df = pd.concat(gold_list, ignore_index=True)

# If more than 500 samples, randomly down-sample to 500
if len(gold_df) > target_n:
    gold_df = gold_df.sample(n=target_n, random_state=42)

# If fewer than 500 samples, randomly sample additional tickets
elif len(gold_df) < target_n:
    remaining = df[~df["ticket_id"].isin(gold_df["ticket_id"])]
    extra = remaining.sample(n=target_n - len(gold_df), random_state=42)
    gold_df = pd.concat([gold_df, extra], ignore_index=True)

# Save the final Gold Standard Set for manual annotation
gold_df.to_csv("gold_set_stratified_by_ticket_type.csv", index=False)

print("Gold set created and saved as gold_set_stratified_by_ticket_type.csv")


# -----Section 2: Load the Manually Labeled Gold Set-----
# Run this section after manual annotation is completed

df2 = pd.read_csv("gold_set_labeled.csv")

# Check available columns
print(df2.columns)


# Section 3: VADER Lexicon-based Sentiment Baseline

# Initialize VADER sentiment analyzer
analyzer = SentimentIntensityAnalyzer()

# Define binary sentiment classification function
def vader_binary(text):
    scores = analyzer.polarity_scores(str(text))  # Convert to string to avoid NaN issues
    compound = scores['compound']
    
    # VADER sentiment score (compound):
    # compound ranges from -1 (most negative) to +1 (most positive)
    # compound <= -0.05 -> Negative (1)
    # compound > -0.05 -> Non-negative (0)
    if compound <= -0.05:
        return 1
    else:
        return 0

# Apply VADER to ticket descriptions
df2["vader_binary"] = df2["ticket_description"].apply(vader_binary)

df2[["ticket_id", "ticket_description", "gold_label", "vader_binary"]].head()

# Save VADER predictions (model output)
vader_output = df2[[
    "ticket_id",
    "ticket_description",
    "gold_label",
    "vader_binary"
]].copy()

vader_output.to_csv(
    "vader_predictions.csv",
    index=False
)

print("VADER prediction file saved: vader_predictions_on_gold_set.csv")

# Section 4: Evaluation Metrics

y_true = df2["gold_label"]
y_pred = df2["vader_binary"]

# Evaluate metrics using Negative (label=1) as the positive class.
# This reports Precision, Recall, and F1-score for negative tickets.

print("Recall (Negative):",
      recall_score(y_true, y_pred, pos_label=1))

print("Precision (Negative):",
      precision_score(y_true, y_pred, pos_label=1))

print("F1 (Negative):",
      f1_score(y_true, y_pred, pos_label=1))

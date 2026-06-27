"""
Step 1: Data preprocessing

This script:
1. Combines customer support ticket datasets from two sources.
2. Standardises the schema across datasets.
3. Filters English-language tickets.
4. Cleans ticket descriptions by removing HTML tags, normalising text, and handling missing values.
5. Exports the cleaned dataset for downstream sentiment classification.
"""

import pandas as pd
import re

# Step1: Combine the datasets
# Read two original tables 
df1 = pd.read_csv("tickets.csv")      # Table 1
df2 = pd.read_csv("customer_support_tickets.csv") # Table 2

# Keep only English tickets in Table 2
df2 = df2[df2["language"] == "en"].copy()

# ----- Rename columns for consistency -----

# Mapping for Table 1
rename_map_1 = {
    "category": "ticket_type",
    "description": "ticket_description",
    "priority": "priority",
}

df1 = df1.rename(columns=rename_map_1)

# Mapping for Table 2
rename_map_2 = {
    "body": "ticket_description",
    "queue": "ticket_type",
    "priority": "priority",
}

df2 = df2.rename(columns=rename_map_2)

# ------- Define final unified schema -------

final_cols = [
    "ticket_type",
    "ticket_description",
    "priority",
]

# ------- Add missing columns and reorder -------

for col in final_cols:
    if col not in df1.columns:
        df1[col] = pd.NA
    if col not in df2.columns:
        df2[col] = pd.NA

df1 = df1[final_cols]
df2 = df2[final_cols]


# ------- Combine two datasets -------

combined = pd.concat([df1, df2], ignore_index=True)
print("Merge completed. Total rows:", len(combined))

# Step 2: Clean ticket text

def clean_text(text):
    if pd.isna(text):
        return ""

    text = str(text)

    # 1) Replace line breaks and newline markers with spaces
    text = text.replace("\n", " ").replace("\r", " ")
    text = text.replace("\\n", " ").replace("\\r", " ")

    # 2) Remove HTML tags
    text = re.sub(r"<br\s*/?>", " ", text, flags=re.IGNORECASE)
    text = re.sub(r"</?[^>]+>", " ", text)   # Remove other HTML tags such as <p>, <div>

    # 3) Convert text to lowercase
    text = text.lower()

    # 4) Keep only alphanumeric characters and limited punctuation
    text = re.sub(r"[^a-z0-9\s.,!?'\’]", " ", text)

    # 5) Remove extra spaces
    text = re.sub(r"\s+", " ", text).strip()

    return text

# Fill missing text
combined["ticket_description"] = combined["ticket_description"].fillna("")

# Apply cleaning
combined["ticket_description"] = combined["ticket_description"].astype(str).apply(clean_text)


# Export final file

combined.to_csv("combined_tickets_clean_text.csv", index=False, encoding="utf-8")
print("Text cleaning completed! 'ticket_description' has been cleaned.")
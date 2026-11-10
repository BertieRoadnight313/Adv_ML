"""
===============================================================================
Topic: Association Rule Mining & Market Basket Analysis using Apriori Algorithm
Library: mlxtend, pandas, numpy
===============================================================================
"""

import numpy as np
import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules

# =============================================================================
# STEP 1: DATA LOADING & INITIAL EXPLORATION
# =============================================================================
# Load transaction data from CSV file into a pandas DataFrame.
data = pd.read_csv('Week 4/Online_Retail.csv')

# Inspect the first 5 rows to understand column structures and sample values.
print("--- Initial Data Snapshot ---")
print(data.head())

# Print concise summary of DataFrame (column names, data types, non-null counts).
# CRITICAL FOR EXAMS: Use data.info() to quickly check for missing (null) values.
print("\n--- Data Structure & Missing Values ---")
print(data.info())

# List all column names to verify exact string spelling for indexing/filtering.
print("\n--- Column Names ---")
print(data.columns)

# Get array of unique values in the 'Country' column to identify geographical subsets.
print("\n--- Unique Countries ---")
print(data.Country.unique())


# =============================================================================
# STEP 2: DATA CLEANING & PREPROCESSING
# =============================================================================
# 1. Clean Product Descriptions:
# Remove leading and trailing whitespaces from strings in 'Description'.
# WHY: Prevents duplicate item categories caused by spacing variations 
# (e.g., ' RED TOTE BAG ' vs 'RED TOTE BAG').
data['Description'] = data['Description'].str.strip()

# 2. Handle Missing Identifiers:
# Drop rows where 'InvoiceNo' is null/missing (axis=0 means rows).
# inplace=True modifies the DataFrame directly without reassigning.
data.dropna(axis=0, subset=['InvoiceNo'], inplace=True)

# 3. Standardize Data Types:
# Convert 'InvoiceNo' to string to enable string pattern matching (e.g., regex/contains).
data['InvoiceNo'] = data['InvoiceNo'].astype('str')

# 4. Filter Out Cancellations:
# In retail datasets, cancelled invoices often start with the letter 'C'.
# ~ is the boolean NOT operator. We keep rows where InvoiceNo does NOT contain 'C'.
data = data[~data['InvoiceNo'].str.contains('C')]


# =============================================================================
# STEP 3: CONSTRUCTING THE BASKET MATRIX (ONE-HOT / TRANSACTION FORMAT)
# =============================================================================
# Market Basket Analysis requires data structured as a 2D Transaction-Item Matrix:
# - Rows = Unique Transactions (InvoiceNo)
# - Columns = Unique Items (Description)
# - Cell values = Quantity purchased in that transaction.

# Pipeline for France:
# 1. Subsets rows where Country is "France".
# 2. Group by InvoiceNo and Description, then sum total Quantity.
# 3. unstack(): Pivots 'Description' from index to columns.
# 4. reset_index(): Flattens index temporarily.
# 5. fillna(0): Replaces NaN (items not bought in that invoice) with 0.
# 6. set_index('InvoiceNo'): Sets InvoiceNo back as row labels.

basket_France = (data[data['Country'] == "France"]
                 .groupby(['InvoiceNo', 'Description'])['Quantity']
                 .sum().unstack().reset_index().fillna(0)
                 .set_index('InvoiceNo'))
print("\nFrance Basket Shape (Transactions x Items):", basket_France.shape)

# Repeat pivot transformation for United Kingdom
basket_UK = (data[data['Country'] == "United Kingdom"]
             .groupby(['InvoiceNo', 'Description'])['Quantity']
             .sum().unstack().reset_index().fillna(0)
             .set_index('InvoiceNo'))
print("UK Basket Shape:", basket_UK.shape)

# Repeat pivot transformation for Portugal
basket_Portugal = (data[data['Country'] == "Portugal"]
                 .groupby(['InvoiceNo', 'Description'])['Quantity']
                 .sum().unstack().reset_index().fillna(0)
                 .set_index('InvoiceNo'))
print("Portugal Basket Shape:", basket_Portugal.shape)

# Repeat pivot transformation for Sweden
basket_Sweden = (data[data['Country'] == "Sweden"]
                 .groupby(['InvoiceNo', 'Description'])['Quantity']
                 .sum().unstack().reset_index().fillna(0)
                 .set_index('InvoiceNo'))
print("Sweden Basket Shape:", basket_Sweden.shape)


# =============================================================================
# STEP 4: BINARY ONE-HOT ENCODING (0/1 MATRIX CONVERSION)
# =============================================================================
# Apriori algorithm requires binary (0 or 1) entries:
# 0 = Item was NOT present in transaction
# 1 = Item WAS present in transaction (regardless of quantity count)

def hot_encode(x):
    """Encodes quantities <= 0 as 0, and quantities >= 1 as 1."""
    if x <= 0:
        return 0
    if x >= 1:
        return 1

# Apply hot_encode function element-wise across each basket DataFrame.
# NOTE: .map() is modern pandas syntax; in older pandas versions, .applymap() was used.
basket_encoded_France = basket_France.map(hot_encode)
basket_encoded_UK = basket_UK.map(hot_encode)
basket_encoded_Portugal = basket_Portugal.map(hot_encode)
basket_encoded_Sweden = basket_Sweden.map(hot_encode)

# Reassign encoded matrices back to working variable names
basket_France = basket_encoded_France
basket_UK = basket_encoded_UK   
basket_Portugal = basket_encoded_Portugal
basket_Sweden = basket_encoded_Sweden


# =============================================================================
# STEP 5: MODEL BUILDING - FREQUENT ITEMSETS & ASSOCIATION RULES
# =============================================================================
# A. Find Frequent Itemsets using Apriori algorithm
# Parameters:
# - basket_France: 0/1 binary transaction matrix.
# - min_support = 0.05: Keep itemsets appearing in >= 5% of all France invoices.
#   EXAM TIP: Lower min_support finds more rules but takes longer & uses more memory.
# - use_colnames = True: Returns actual item names instead of column integer indices.
frq_items_France = apriori(basket_France, min_support=0.05, use_colnames=True)

# B. Generate Association Rules from Frequent Itemsets
# Parameters:
# - metric = "lift": Metric used to evaluate & filter rules. Options: "support", "confidence", "lift".
# - min_threshold = 1: Keep only rules with Lift >= 1.0 (positive association).
#   EXAM TIP: Lift > 1 means items are bought together more often than by random chance.
rules_FR = association_rules(frq_items_France, metric="lift", min_threshold=1)

# C. Sort & Filter Rules
# Sort rules primarily by 'confidence' (descending), then by 'lift' (descending).
rules_FR = rules_FR.sort_values(['confidence', 'lift'], ascending=[False, False])

# Display top 10 rules
print("\n--- Top 10 Association Rules for France ---")
print(rules_FR.head(10))


# =============================================================================
# QUICK EXAM REFERENCE & METRICS:
# =============================================================================
# 1. Support(A -> B) = P(A ∩ B)
#    - Percentage of total transactions containing both item A and item B.
#
# 2. Confidence(A -> B) = P(B | A) = Support(A ∩ B) / Support(A)
#    - Conditional probability: Given item A is bought, how likely is item B bought?
#
# 3. Lift(A -> B) = Confidence(A -> B) / Support(B) = P(A ∩ B) / (P(A) * P(B))
#    - Lift > 1: Positive correlation (items promote each other).
#    - Lift = 1: Independent items (no relation).
#    - Lift < 1: Negative correlation (substitutes/repelling items).
#
# 4. Filtering Rules for Specific Products in Exams:
#    Example: rules_FR[rules_FR['antecedents'].apply(lambda x: 'POSTAGE' in x)]
# =============================================================================

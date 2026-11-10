"""
===============================================================================
MACHINE LEARNING ASSESSMENT CHEAT SHEET & TUTORIAL ANNOTATION
Topic: Natural Language Processing (NLP), Text Preprocessing & Sentiment Analysis
Libraries: nltk, bs4 (BeautifulSoup), scikit-learn, wordcloud, pandas, numpy
===============================================================================
"""

import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from bs4 import BeautifulSoup
from wordcloud import WordCloud

# NLTK Downloads & Preprocessing Setup
import nltk
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Machine Learning & Feature Extraction Setup
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Instantiate NLTK Stopwords set and Lemmatizer
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()


# =============================================================================
# STEP 1: DATA LOADING & EXPLORATION
# =============================================================================
# Load IMDB Movie Reviews Dataset (Columns: 'review', 'sentiment')
imdb = pd.read_csv('Week 6/IMDB_Dataset.csv')

print("--- Dataset Shape ---")
print("Shape (rows, columns):", imdb.shape)

print("\n--- Target Class Distribution ---")
# Check class balance (positive vs negative reviews)
print(imdb['sentiment'].value_counts())

print("\n--- First 6 Rows ---")
print(imdb.head(6))


# =============================================================================
# STEP 2: TEXT PREPROCESSING & CLEANING FUNCTION
# =============================================================================
def clean_text(text, do_lemmatize=True, remove_digits=True):
    """
    Standard NLP Pipeline for Text Cleaning:
    1. HTML Parsing: Removes <br />, <p>, and other HTML tags.
    2. Bracket Removal: Strips text inside square brackets [like this].
    3. Punctuation/Digit Removal: Strips non-alphabetic characters using Regex.
    4. Lowercasing & Tokenization: Converts text to lowercase and splits into words.
    5. Stopword Removal: Filters out common non-informative words ('the', 'is', 'at').
    6. Lemmatization: Reduces words to base root form ('running' -> 'run', 'better' -> 'good').
    """
    # 1. Remove HTML formatting tags
    text = BeautifulSoup(text, "html.parser").get_text()
    
    # 2. Remove square brackets and contents inside them
    text = re.sub(r'\[[^\]]*\]', ' ', text)
    
    # 3. Regex Filtering for non-alphabetic / non-numeric characters
    if remove_digits:
        text = re.sub(r'[^a-zA-Z\s]', ' ', text)  # Keep only letters and spaces
    else:
        text = re.sub(r'[^a-zA-Z0-9\s]', ' ', text)  # Keep letters, numbers, and spaces
        
    # 4. Convert to lowercase and split into whitespace-delimited tokens
    tokens = text.lower().split()
    
    # 5. Remove English stopwords
    tokens = [t for t in tokens if t not in stop_words]
    
    # 6. Apply Lemmatization (reduces words to dictionary headword form)
    if do_lemmatize:
        tokens = [lemmatizer.lemmatize(t) for t in tokens]
        
    # Rejoin token list back into a single clean string space-separated
    return ' '.join(tokens)


# --- Optional Debugging / Testing on a Small Sample ---
# Preprocessing full datasets can take time, so testing on a sample (e.g. 2000 rows) is recommended!
imdb_sample = imdb.sample(2000, random_state=42).copy()
imdb_sample['clean'] = imdb_sample['review'].apply(clean_text)
print("\n--- Preprocessed Text Comparison Sample ---")
print(imdb_sample[['review', 'clean']].head(3))

# --- Apply Preprocessing to Full Dataset ---
imdb['clean'] = imdb['review'].apply(clean_text)

# Optional: Save preprocessed dataset checkpoint to CSV to avoid re-running cleaning pipeline
imdb.to_csv('imdb_clean_short_debug.csv', index=False)


# =============================================================================
# STEP 3: FEATURE EXTRACTION (VECTORIZATION - BOW & TF-IDF)
# =============================================================================
# Set maximum vocabulary size (top 5000 most frequent n-grams)
MAX_FEAT = 5000

# 1. Bag of Words (CountVectorizer):
# Counts raw occurrences of words/phrases.
# ngram_range=(1,2) includes both Unigrams ("good") and Bigrams ("not good").
bow_vec = CountVectorizer(max_features=MAX_FEAT, ngram_range=(1, 2))

# 2. TF-IDF (Term Frequency-Inverse Document Frequency):
# Weights words based on frequency in a document scaled by inverse overall document frequency.
# High TF-IDF = word is frequent in THIS document but rare across ALL documents (highly informative).
tfidf_vec = TfidfVectorizer(max_features=MAX_FEAT, ngram_range=(1, 2))

# Fit vectorizers and transform text into sparse matrices
X_bow = bow_vec.fit_transform(imdb['clean'])
X_tfidf = tfidf_vec.fit_transform(imdb['clean'])

print("\n--- Feature Matrix Shapes ---")
print("Bag-of-Words Matrix Shape (Docs x Features):", X_bow.shape)
print("TF-IDF Matrix Shape (Docs x Features):", X_tfidf.shape)


# =============================================================================
# STEP 4: TARGET ENCODING & TRAIN / TEST SPLIT
# =============================================================================
# Convert string target labels to binary integers: 'positive' -> 1, 'negative' -> 0
y = imdb['sentiment'].map({'positive': 1, 'negative': 0}).values

# Perform Stratified Train-Test Split (80% train, 20% test)
# EXAM TIP: Always pass stratify=y to preserve class balance!
X_tr_bow, X_te_bow, y_tr, y_te = train_test_split(
    X_bow, y, test_size=0.2, random_state=42, stratify=y
)

X_tr_tfidf, X_te_tfidf, _, _ = train_test_split(
    X_tfidf, y, test_size=0.2, random_state=42, stratify=y
)

print("\n--- Train / Test Split Sizes ---")
print(f"Training set: {X_tr_bow.shape[0]} | Testing set: {X_te_bow.shape[0]}")


# =============================================================================
# STEP 5: MODEL TRAINING & EVALUATION
# =============================================================================

# -----------------------------------------------------------------------------
# Model 1: Multinomial Naive Bayes (MNB) on Bag-of-Words
# -----------------------------------------------------------------------------
# MNB applies Bayes' Theorem assuming independence between feature counts.
# Standard baseline algorithm for text classification problems.
mnb = MultinomialNB()
mnb.fit(X_tr_bow, y_tr)
mnb_pred = mnb.predict(X_te_bow)

print("\n--- 1. Multinomial Naive Bayes (BOW) Performance ---")
print("Accuracy Score:", accuracy_score(y_te, mnb_pred))
print("Classification Report:\n", classification_report(y_te, mnb_pred))


# -----------------------------------------------------------------------------
# Model 2: Logistic Regression (LR) on Bag-of-Words
# -----------------------------------------------------------------------------
# Fits a linear decision boundary with a sigmoid link function.
# max_iter=400 ensures optimization solver reaches convergence on sparse data.
lr = LogisticRegression(max_iter=400)
lr.fit(X_tr_bow, y_tr)
lr_pred = lr.predict(X_te_bow)

print("\n--- 2. Logistic Regression (BOW) Performance ---")
print("Accuracy Score:", accuracy_score(y_te, lr_pred))
print("Classification Report:\n", classification_report(y_te, lr_pred))


# -----------------------------------------------------------------------------
# Model 3: Hyperparameter Tuning / Alternative Vectorizer (TF-IDF + Trigrams)
# -----------------------------------------------------------------------------
# Testing larger vocabulary (8000) with Unigrams, Bigrams, and Trigrams (1,3)
vec2 = TfidfVectorizer(max_features=8000, ngram_range=(1, 3))
X2_tfidf = vec2.fit_transform(imdb['clean'])
X_tr_tf2, X_te_tf2, y_tr2, y_te2 = train_test_split(
    X2_tfidf, y, test_size=0.2, random_state=42, stratify=y
)


# -----------------------------------------------------------------------------
# Model 4: Stochastic Gradient Descent Classifier (SGDClassifier) on TF-IDF
# -----------------------------------------------------------------------------
# SGDClassifier implements regularized linear models with SGD learning.
# Extremely efficient for large-scale, high-dimensional text datasets.
sgd = SGDClassifier(max_iter=500, tol=1e-3, random_state=42)
sgd.fit(X_tr_tfidf, y_tr)
sgd_pred = sgd.predict(X_te_tfidf)

print("\n--- 3. SGDClassifier (TF-IDF) Performance ---")
print("Accuracy Score:", accuracy_score(y_te, sgd_pred))


# =============================================================================
# STEP 6: ERROR ANALYSIS & MISCLASSIFIED INSPECTION
# =============================================================================
# Find instances where predicted labels do NOT match ground truth labels
misclassified_mask = (y_te != mnb_pred)
misclassified_indices = np.where(misclassified_mask)[0][:10]  # First 10 error indices

# StratifiedShuffleSplit helps recover original DataFrame row indices
sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for tr_idx, te_idx in sss.split(imdb['clean'], y):
    break

print("\n--- Misclassified Examples Inspection (First 10 Errors) ---")
for i in misclassified_indices:
    orig_i = te_idx[i]
    print(f"ORIGINAL ROW INDEX: {orig_i}")
    print(f"True Label: {imdb.loc[orig_i, 'sentiment']}")
    print(f"Predicted Label: {'positive' if mnb_pred[i] == 1 else 'negative'}")
    print(f"Clean Review Snippet: {imdb.loc[orig_i, 'clean'][:300]}...")
    print("-" * 60)


# =============================================================================
# STEP 7: VISUALIZATION & INTERPRETABILITY
# =============================================================================

# 1. Word Cloud Visualization for Positive vs Negative Text
pos_text = ' '.join(imdb[imdb.sentiment == 'positive']['clean'].tolist())
neg_text = ' '.join(imdb[imdb.sentiment == 'negative']['clean'].tolist())

wc_pos = WordCloud(width=800, height=400, max_words=200, background_color='white').generate(pos_text)
wc_neg = WordCloud(width=800, height=400, max_words=200, background_color='black').generate(neg_text)

# Plot Positive WordCloud
plt.figure(figsize=(10, 4))
plt.imshow(wc_pos, interpolation='bilinear')
plt.axis('off')
plt.title('Positive Words Cloud', fontsize=14)
plt.tight_layout()
plt.show()

# Plot Negative WordCloud
plt.figure(figsize=(10, 4))
plt.imshow(wc_neg, interpolation='bilinear')
plt.axis('off')
plt.title('Negative Words Cloud', fontsize=14)
plt.tight_layout()
plt.show()


# 2. Extract Top Informative Features (Model Coefficients)
def show_top_coef(classifier, vectorizer, n=20):
    """
    Extracts top positive and top negative coefficients from a linear model.
    - Large positive coefficients indicate words driving POSITIVE predictions.
    - Large negative coefficients indicate words driving NEGATIVE predictions.
    """
    feat_names = vectorizer.get_feature_names_out()
    coefs = classifier.coef_[0]
    
    # Sort indices by coefficient magnitude
    top_pos = np.argsort(coefs)[-n:]  # Highest positive values
    top_neg = np.argsort(coefs)[:n]   # Lowest negative values
    
    print("\n--- Top 20 Positive Features (Predict Positive Sentiment) ---")
    print(feat_names[top_pos][::-1])  # Reverse for descending order
    
    print("\n--- Top 20 Negative Features (Predict Negative Sentiment) ---")
    print(feat_names[top_neg])

# Inspect top features for Logistic Regression on Bag-of-Words
show_top_coef(lr, bow_vec, n=20)

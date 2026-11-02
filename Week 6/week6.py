# Libraries

from os import remove

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import re
from bs4 import BeautifulSoup
import nltk
nltk.download('stopwords')
nltk.download('punkt')
from nltk.corpus import stopwords
stop_words = set(stopwords.words('english'))
from nltk.stem import WordNetLemmatizer
nltk.download('wordnet')
lemmatizer = WordNetLemmatizer()

# EDA
imdb = pd.read_csv('Week 6/IMDB Dataset.csv')
# print("Shape:", imdb.shape)
# print(imdb['sentiment'].value_counts())
# print(imdb.head(6))

# Display 3 random reviews

#for i in imdb.sample(3).index:
#    print("===SAMPLE===")
#    print(imdb.loc[i, 'review'][:500], "\n---label:", imdb.loc[i, 'sentiment'], "\n")

# Clean Text

def clean_text(text, do_lemmatize=True, remove_digits=True):
    text = BeautifulSoup(text, "html.parser").get_text()  # Remove HTML tags
    text = re.sub(r'[[^\]]*\]','', text)
    if remove_digits:
        text = re.sub(r'^a-zA-Z\s', '', text)  # Remove digits
    else:
        text = re.sub(r'[^a-zA-Z0-9\s]', '', text)  # Remove special characters
    tokens = text.lower().split()  # Tokenize and convert to lowercase
    tokens = [t for t in tokens if t not in stop_words]
    if do_lemmatize:
        tokens = [lemmatizer.lemmatize(t) for t in tokens]  # Lemmatize
    return ' '.join(tokens)  # Join tokens back into a single string

# full data set preprocessing
# imdb['clean'] = imdb['review'].apply(clean_text)
# imdb.to_csv('/Week 6/IMDB_cleaned.csv', index=False)

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer 
MAX_FEAT = 5000   # manageable for a class demo 
bow_vec = CountVectorizer(max_features=MAX_FEAT, ngram_range=(1,2)) 
tfidf_vec = TfidfVectorizer(max_features=MAX_FEAT, ngram_range=(1,2)) 
X_bow = bow_vec.fit_transform(imdb['clean']) 
X_tfidf = tfidf_vec.fit_transform(imdb['clean']) 
print("BOW shape:", X_bow.shape) 
print("TFIDF shape:", X_tfidf.shape)

from sklearn.model_selection import train_test_split 
y = imdb['sentiment'].map({'positive':1, 'negative':0}).values 
X_tr_bow, X_te_bow, y_tr, y_te = train_test_split(X_bow, y, test_size=0.2, random_state=42, stratify=y) 
X_tr_tfidf, X_te_tfidf, _, _ = train_test_split(X_tfidf, y, test_size=0.2, random_state=42, stratify=y) 
print("Train shape:", X_tr_bow.shape, "Test shape:", X_te_bow.shape)


from sklearn.naive_bayes import MultinomialNB 
from sklearn.linear_model import LogisticRegression 
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix 
# MultinomialNB on BOW 
mnb = MultinomialNB() 
mnb.fit(X_tr_bow, y_tr) 
mnb_pred = mnb.predict(X_te_bow) 
print("MNB (BOW) acc:", accuracy_score(y_te, mnb_pred))
print(classification_report(y_te, mnb_pred)) 
# Logistic Regression on BOW 
lr = LogisticRegression(max_iter=400) 
lr.fit(X_tr_bow, y_tr) 
lr_pred = lr.predict(X_te_bow) 
print("LR (BOW) acc:", accuracy_score(y_te, lr_pred)) 
print(classification_report(y_te, lr_pred)) 

# Use MNB predictions as example: 
mis = (y_te != mnb_pred) 
mis_ix = np.where(mis)[0][:10]  # first 10 errors 
# map indices back to original rows after train_test_split:
# We need the test indices. Simpler: rebuild split with return indices: 
from sklearn.model_selection import StratifiedShuffleSplit 
sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42) 
for tr_idx, te_idx in sss.split(imdb['clean'], y): 
break 
# Example: show misclassified examples 
for i in mis_ix: 
orig_i = te_idx[i] 
print("ORIG INDEX:", orig_i) 
print("Label:", imdb.loc[orig_i, 'sentiment']) 
print("Pred:", "positive" if mnb_pred[i]==1 else "negative") 
print(imdb.loc[orig_i, 'clean'][:400]) 
print("----")


pos_text = ' '.join(imdb[imdb.sentiment=='positive']['clean'].tolist()) 
neg_text = ' '.join(imdb[imdb.sentiment=='negative']['clean'].tolist()) 
wc_pos = WordCloud(max_words=200).generate(pos_text) 
wc_neg = WordCloud(max_words=200).generate(neg_text) 
plt.figure(figsize=(10,4)); plt.imshow(wc_pos); plt.axis('off'); plt.title('Positive words') 
plt.figure(figsize=(10,4)); plt.imshow(wc_neg); plt.axis('off'); plt.title('Negative words') 
# Most informative features for LR (if using scikit-learn LR) 
def show_top_coef(classifier, vectorizer, n=20): 
# assumes binary LR with coef_ shape (1, n_features) 
feat_names = vectorizer.get_feature_names_out() 
coefs = classifier.coef_[0] 
top_pos = np.argsort(coefs)[-n:] 
top_neg = np.argsort(coefs)[:n] 
print("Top positive features:\n", feat_names[top_pos]) 
print("Top negative features:\n", feat_names[top_neg]) 
show_top_coef(lr, bow_vec, n=20) 



import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import re
from bs4 import BeautifulSoup
import nltk
nltk.download('stopwords')
nltk.download('punkt')
from nltk.corpus import stopwords
stop_words = set(stopwords.words('english'))

imdb = pd.read_csv('Week 6/IMDB_Dataset.csv') # Colab path; adjust if needed
print("Shape:", imdb.shape)
print(imdb['sentiment'].value_counts())
imdb.head(6)


from nltk.stem import WordNetLemmatizer
nltk.download('wordnet')
lemmatizer = WordNetLemmatizer()

def clean_text(text, do_lemmatize=True, remove_digits=True):
    text = BeautifulSoup(text, "html.parser").get_text()
    text = re.sub(r'\[[^\]]*\]', ' ', text)
    if remove_digits:
        text = re.sub(r'[^a-zA-Z\s]', ' ', text)
    else:
         text = re.sub(r'[^a-zA-Z0-9\s]', ' ', text)
    tokens = text.lower().split()
    tokens = [t for t in tokens if t not in stop_words]
    if do_lemmatize:
        tokens = [lemmatizer.lemmatize(t) for t in tokens]
    return ' '.join(tokens)

imdb_sample = imdb.sample(2000, random_state=42).copy()
imdb_sample['clean'] = imdb_sample['review'].apply(clean_text)
imdb_sample[['review','clean']].head(3)

imdb['clean'] = imdb['review'].apply(clean_text)
imdb.to_csv('imdb_clean_short_debug.csv', index=False) # optional checkpoint save


from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
MAX_FEAT = 5000 # manageable for a class demo
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

# change ngram or features
vec2 = TfidfVectorizer(max_features=8000, ngram_range=(1,3))
X2 = vec2.fit_transform(imdb['clean'])
# then split and run MNB/LR as above
# try SGDClassifier
from sklearn.linear_model import SGDClassifier
sgd = SGDClassifier(max_iter=500, tol=1e-3)
sgd.fit(X_tr_tfidf, y_tr)
print("SGD (tfidf) acc:", accuracy_score(y_te, sgd.predict(X_te_tfidf)))

# Use MNB predictions as example:
mis = (y_te != mnb_pred)
mis_ix = np.where(mis)[0][:10] # first 10 errors
# map indices back to original rows after train_test_split

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
    print("---")

    pos_text = ' '.join(imdb[imdb.sentiment == 'positive']['clean'].tolist())
    neg_text = ' '.join(imdb[imdb.sentiment == 'negative']['clean'].tolist())

    wc_pos = WordCloud(max_words=200).generate(pos_text)
    wc_neg = WordCloud(max_words=200).generate(neg_text)

    plt.figure(figsize=(10, 4));plt.imshow(wc_pos);plt.axis('off'); plt.title('Positive words')
    plt.figure(figsize=(10, 4));plt.imshow(wc_neg);plt.axis('off'); plt.title('Negative words')


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
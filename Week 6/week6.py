# Libraries

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

for i in imdb.sample(3).index:
    print("===SAMPLE===")
    print(imdb.loc[i, 'review'][:500], "\n---label:", imdb.loc[i, 'sentiment'], "\n")

# Clean Text

def clean_text(text, do_lemmatize=True):
ewatdfygutreastdyug


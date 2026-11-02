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


imdb = pd.read_csv('Week 6/IMDB Dataset.csv')
print("Shape:", imdb.shape)
print(imdb['sentiment'].value_counts())
print(imdb.head(6))
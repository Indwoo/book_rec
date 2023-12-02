import pandas as pd
import nltk
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.tokenize import word_tokenize
import numpy as np
from sklearn.preprocessing import LabelBinarizer

nltk.download('punkt')
nltk.download('wordnet')

book = pd.read_csv('archive/Books.csv', delimiter=';')
rating = pd.read_csv('archive/Ratings.csv', delimiter=';')
user = pd.read_csv('archive/Users.csv', delimiter=';')

book_norm = book.dropna()
rating_norm = rating.dropna()
user_norm = user.dropna()
user_norm.to_csv('user_norm.csv', index=False)
# 어간 추출
# stemmed_titles = []
# stemmer = nltk.stem.PorterStemmer()
# for title in book_norm['Title']:
#     stemmed_title = ' '.join([stemmer.stem(word) for word in word_tokenize(title)])
#     stemmed_titles.append(stemmed_title)
# book_norm['stemmed_title'] = stemmed_titles
#
# # 표제어 추출
# lemmatized_titles = []
# lemmatizer = WordNetLemmatizer()
# for title in book_norm['stemmed_title']:
#     lemmatized_title = ' '.join([lemmatizer.lemmatize(word) for word in word_tokenize(title)])
#     lemmatized_titles.append(lemmatized_title)
# book_norm['lemmatized_title'] = lemmatized_titles
#
# book_norm.to_csv('book_stemming_2.csv', index=False)

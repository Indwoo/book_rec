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

# LabelBinarizer를 사용하여 User-ID를 벡터로 변환
label_binarizer = LabelBinarizer()
user_vector = label_binarizer.fit_transform(user_norm['User-ID'])

# 자신의 User-ID에 해당하는 인덱스를 1로, 나머지 인덱스를 0으로 설정
user_vector = np.where(user_vector == user_vector.max(axis=1, keepdims=True), 1, 0)

# 변환된 벡터를 DataFrame에 추가
user_vector_df = pd.DataFrame(user_vector, columns=label_binarizer.classes_)
user_norm = pd.concat([user_norm, user_vector_df], axis=1)

user_norm.to_csv('user_features.csv', index=False)
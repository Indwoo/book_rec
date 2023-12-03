import pandas as pd
import nltk
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.tokenize import word_tokenize
import numpy as np
from sklearn.preprocessing import LabelBinarizer, OneHotEncoder

nltk.download('punkt')
nltk.download('wordnet')

book = pd.read_csv('book_stemming_2.csv')
user_norm = pd.read_csv('cleaned_user_norm.csv')
#rating = pd.read_csv('archive/Ratings.csv', delimiter=';')

# Title_stem 벡터 생성
stemmed_title_counts = book['stemmed_title'].apply(lambda x: nltk.word_tokenize(x)).explode().value_counts()
stemmed_title_vector = stemmed_title_counts[stemmed_title_counts >= 1].index.tolist()

title_stem_vector = []
for title in book['Title']:
    title_tokens = nltk.word_tokenize(title)
    title_stem = [token for token in title_tokens if token in stemmed_title_vector]
    title_stem_vector.append(' '.join(title_stem))

# Author_stem 벡터 생성
author_stem_counts = book['Author'].apply(lambda x: nltk.word_tokenize(x)).explode().value_counts()
author_stem_vector = author_stem_counts[author_stem_counts >= 2].index.tolist()

author_stem_vector = []
for author in book['Author']:
    author_tokens = nltk.word_tokenize(author)
    author_stem = [token for token in author_tokens if token in author_stem_vector]
    author_stem_vector.append(' '.join(author_stem))

# Publisher_stem 벡터 생성
publisher_stem_counts = book['Publisher'].apply(lambda x: nltk.word_tokenize(x)).explode().value_counts()
publisher_stem_vector = publisher_stem_counts[publisher_stem_counts >= 2].index.tolist()

publisher_stem_vector = []
for publisher in book['Publisher']:
    publisher_tokens = nltk.word_tokenize(publisher)
    publisher_stem = [token for token in publisher_tokens if token in publisher_stem_vector]
    publisher_stem_vector.append(' '.join(publisher_stem))

# Year_stem 벡터 생성
label_binarizer = LabelBinarizer()
year_stem_vector = label_binarizer.fit_transform(book['Year'])

# book 데이터프레임에 특징 벡터 컬럼 추가
book['Title_stem'] = title_stem_vector
book['Author_stem'] = author_stem_vector
book['Publisher_stem'] = publisher_stem_vector
book['Year_stem'] = year_stem_vector

# 특징 벡터가 추가된 book 데이터를 새로운 파일로 저장
book.to_csv('특징벡터_생성.csv', index=False)



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

import pandas as pd
import nltk
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.tokenize import word_tokenize
import numpy as np
from sklearn.preprocessing import LabelBinarizer, OneHotEncoder

nltk.download('punkt')
nltk.download('wordnet')

#book = pd.read_csv('book_stemming_2.csv', delimiter=';')
#rating = pd.read_csv('archive/Ratings.csv', delimiter=';')

#book_norm = book.dropna()
#rating_norm = rating.dropna()

user_norm = pd.read_csv('cleaned_user_norm.csv')

# 나이가 80 이상인 데이터를 제외합니다.
user_norm = user_norm[user_norm['Age'] < 80]

# User-ID와 Age 데이터를 가져옵니다.
user_ids = user_norm['User-ID']
ages = user_norm['Age']

# LabelBinarizer를 생성합니다.
label_binarizer = LabelBinarizer()

# Age 데이터를 원핫인코딩하여 특징 벡터를 생성합니다.
age_vector = label_binarizer.fit_transform(ages)

# 데이터프레임으로 변환합니다.
df = pd.DataFrame(age_vector, columns=label_binarizer.classes_)

# User-ID 컬럼을 추가합니다.
df['User-ID'] = user_ids

# User-ID가 비어있는 데이터를 제거합니다.
df = df.dropna(subset=['User-ID'])

# CSV 파일로 저장합니다.
df.to_csv('user_vector.csv', index=False)
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

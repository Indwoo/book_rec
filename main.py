import pandas as pd
import nltk
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.tokenize import word_tokenize

nltk.download('punkt')
nltk.download('wordnet')

book = pd.read_csv('archive/Books.csv', delimiter=';')
rating = pd.read_csv('archive/Ratings.csv', delimiter=';')
user = pd.read_csv('archive/Users.csv', delimiter=';')

book_norm = book.dropna()
rating_norm = rating.dropna()
user_norm = user.dropna()

# 어간 추출
stemmer = PorterStemmer()
book_norm['stemmed_title'] = book_norm['Title'].apply(lambda x: ' '.join([stemmer.stem(word) for word in word_tokenize(x)]))

# 표제어 추출
lemmatizer = WordNetLemmatizer()
book_norm['lemmatized_title'] = book_norm['Title'].apply(lambda x: ' '.join([lemmatizer.lemmatize(word) for word in word_tokenize(x)]))

book_norm.to_csv('결과파일.csv', index=False)


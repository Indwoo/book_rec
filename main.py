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

print(book_norm)
print(rating_norm)
print(user_norm)


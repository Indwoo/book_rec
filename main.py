import csv
import pandas as pd

book = pd.read_csv('archive/Books.csv', delimiter=';')
rating = pd.read_csv('archive/Ratings.csv', delimiter=';')
user = pd.read_csv('archive/Users.csv', delimiter=';')

book_norm = book.dropna()
rating_norm = rating.dropna()
user_norm = user.dropna()

print(book_norm)
print(rating_norm)
print(user_norm)
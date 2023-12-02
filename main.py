import csv
import pandas as pd

data = pd.read_csv('archive/Ratings.csv', delimiter=';')
data = data.dropna()
print(data)
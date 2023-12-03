import pandas as pd

df = pd.read_csv('user_norm.csv')

pivot_table = pd.pivot_table(df, values='Rating', index='ID', columns='Name', fill_value=0)

print(pivot_table.head())
import pandas as pd

df = pd.read_csv('user_norm.csv')

pivot_table = pd.pivot_table(df, values='Rating', index='ID', columns='Name', fill_value=0)
pivot_table_tr= pivot_table.transpose()
print(pivot_table_tr.head())
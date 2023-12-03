import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import mean_squared_error


# 추천 행렬 함수
def predict_rating(rating, item_sim):
    rating_pred = rating.dot(item_sim) / np.array([np.abs(item_sim).sum(axis=1)])
    return rating_pred

# MSE 함수
def mse(pred, actual):
    pred = pred[actual.nonzero()].flatten()
    actual = actual[actual.nonzero()].flatten()
    return mean_squared_error(pred, actual)

df = pd.read_csv('user_norm.csv')

pivot_table = pd.pivot_table(df, values='Rating', index='ID', columns='Name', fill_value=0)
pivot_table_tr = pivot_table.transpose()

# 유사도 행렬 생성
sim = cosine_similarity(pivot_table_tr, pivot_table_tr)
df_sim = pd.DataFrame(data=sim, index=pivot_table_tr.index, columns=pivot_table_tr.index)

rating_pred = predict_rating(pivot_table.values, df_sim.values)
rating_pred_matrix = pd.DataFrame(data=rating_pred, index=pivot_table.index, columns=pivot_table.columns)

print("mse: ", mse(rating_pred, pivot_table.values))

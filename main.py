import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import mean_squared_error

# 추천 행렬 함수
def predict_rating(rating, item_sim):
    rating_pred = rating.dot(item_sim) / np.array([np.abs(item_sim).sum(axis=1)])
    rating_pred[np.isnan(rating_pred)] = 0  # 0으로 나누는 경우에 대한 예외 처리
    return rating_pred

# MSE 함수
def mse(pred, actual):
    pred = pred[actual.nonzero()].flatten()
    actual = actual[actual.nonzero()].flatten()
    return mean_squared_error(pred, actual)

# MSE 개선 함수
def predict_rating_top_n(rating, item_sim, n=20):
    pred = np.zeros(rating.shape)

    for col in range(rating.shape[1]):
        top_n_item = np.argsort(item_sim[:, col])[:-n - 1:-1]  # 스칼라로 변환
        for row in range(rating.shape[0]):
            pred[row, col] = item_sim[col, top_n_item].dot(rating[row, top_n_item].T)
            pred[row, col] /= np.sum(np.abs(item_sim[col, top_n_item]))

    return pred
#읽은 책, 안 읽은 책 구분
def get_user_unread(rating, use_id):
    user_rating= rating.loc[use_id, :]
    already_read= user_rating[user_rating> 0].index.tolist()
    book_list= rating.columns.tolist()
    unread_list= [book for book in book_list if book not in already_read]
    return unread_list


df = pd.read_csv('user_norm.csv')

pivot_table = pd.pivot_table(df, values='Rating', index='ID', columns='Name', fill_value=0)
pivot_table_tr = pivot_table.transpose()

# 유사도 행렬 생성
sim = cosine_similarity(pivot_table_tr, pivot_table_tr)
df_sim = pd.DataFrame(data=sim, index=pivot_table_tr.index, columns=pivot_table_tr.index)

rating_pred = predict_rating(pivot_table.values, df_sim.values)
rating_pred_matrix = pd.DataFrame(data=rating_pred, index=pivot_table.index, columns=pivot_table.columns)

try:
    #rating_pred_2 = predict_rating_top_n(pivot_table.values, df_sim.values, n=20)
    #rating_pred_matrix = pd.DataFrame(data=rating_pred_2, index=pivot_table.index, columns=pivot_table.columns)
    user_rating_id = pivot_table.loc[1, :]
    print(user_rating_id[user_rating_id > 0].sort_values(ascending=False)[:20])
except KeyboardInterrupt:
    print("오류 발생")


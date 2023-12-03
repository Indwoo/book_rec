import pandas as pd
import numpy as np
import rating as rating

df_user_data= pd.read_csv("user_rating_0_to_1000.csv")
df_user_data= df_user_data[["ID", "Name", "Rating"]]

df_user_data= rating.pivot_table("Rating", index= "ID", columns= "Name").fillna(0)
df_user_data.head()
#%%
import csv

import pandas as pd
import numpy as np
from scipy.linalg import svd
df = pd.read_csv("data/user_movie_likes.csv" , names=["UserID" ,  "MovieID"])
print(df.shape)

user_unique = df.UserID.unique()
movies_unique = df.MovieID.unique()

df_matx = pd.DataFrame(0 , index=user_unique , columns=movies_unique)
#%%
def get_moves_for_users(user_id):
    moves_list = []
    mask = (df.loc[:, 'UserID'] == user_id)
    for movies in df[mask].MovieID:
        moves_list.append(movies)
    for mv in moves_list:
        df_matx.loc[user_id, mv] = 1
    return moves_list

#%%
for user in user_unique:
    get_moves_for_users(user)
#%%



#%%
user_movie_map = defaultdict(list)

with open('data/user_movie_likes.csv' , 'r' ) as csvfile:
    w = csv.reader(csvfile , delimiter = ',' )
    for row in w:
        user_movie_map[int(row[0])].append(int)
        movie_user_map


# def movies_similar(MovieID):
#


def  get_similar_movie(user_movie_map , movie_user_map , m):
    biglist = []
    for u in movie_user_map[m]:
        biglist.extend(user_movie_map[u])
    return Counter(biglist).most_common(4)[1:]


def get_movie_recommendation(user_movie_map, movie , u1):




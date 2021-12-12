import pandas as pd
import numpy as np
df = pd.read_csv("albb-salaries-2003 (6).csv")
# print(df.shape)
# print(df.columns)
# print(df.info())
# print(df.value_counts())
# print(df.columns)
# print(df.loc[:,"Team"])
# print(df.groupby("Team").sum())
# print(df.groupby)
print(df['Team'].is_unique)

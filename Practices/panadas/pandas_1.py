
import pandas as pd
import numpy as np

# print("Pandas version:", pd.__version__)
# print("Numpy version:", np.__version__)



#
# s = pd.Series([1, 3, 5, np.nan, 6, 8])
# print(s)
#
# df1 = pd.DataFrame(np.random.randn(6, 4), columns=list('ABCD'))
# print(df1)
#
#
#
## Make a dataframe from a dictionary
df2 = pd.DataFrame({
    'A': 1.,
    'B': pd.Timestamp('20130102'),
    'C': pd.Series(1, index=list(range(4)), dtype='float32'),
    'D': np.array([3] * 4, dtype='int32'),
    'E': pd.Categorical(["test", "train", "test", "train"]),
    'F': 'foo'
})
# print(df2)
#

df2_2 = pd.DataFrame({
    "A" :  "Data of A",
    "B" :  np.array([2 , 2]),
    "C" :  ["UU" , "KK"]
})
# print(df2_2)


# colm_name = ["A", "B", "C"] # this is the original way
colm_name = list('ABC')
row_data = [
    [5 , "for 0_B" , "Male"] ,
    [2 , "for 1_B" , "Fmale"] ,
    [6 , "for 2_B" , "Fmale"] ,
    [3 , "for 3_B" , "Null"] ,
    [5 , "for 3_B" , np.nan] ,
]

df_from_arrays = pd.DataFrame(row_data , columns = colm_name)
# print(df_from_arrays)

# print(df2_2.head())
#
# df_from_arrays.rename(columns={'A' : "NewA"} , inplace=True)
# print(df_from_arrays.head())
# print(df_from_arrays.columns)


# print(df_from_arrays.A.describe())
# print("===========")
# print(df_from_arrays.A.value_counts())
# print("===========")
# print(df_from_arrays.A.mean())
# print("===========")
# print(df_from_arrays.A.unique())
# print("============")
# nba_df['Date'] = pd.to_datetime(nba_df['Date'])

#
# #if we want to slecte first 2 rows
# print(df_from_arrays.loc[0:1])
# print("============")
# #if we want to select specific columns with limit rows:
# print(df_from_arrays.loc[0:2 , ["A" , "B"] ])
# print("============")
# #if we want all rows but specific columns :
# print(df_from_arrays.loc[ : , ['A' , 'C'] ])
# print("============")
# #we can use range for columns EX, start form A until C
# print(df_from_arrays.loc[ : , 'A':'C' ]) #No List
# print("============")
# #If we want to target specific data(vlue) at specific columns
# print(df_from_arrays.loc[df_from_arrays.A > 2])
# print("============")
# #If we want to target specific data(vlue) at specific columns
# print(df_from_arrays.loc[df_from_arrays.B == "for 2_B"])
# print("============")
# #by integer position  .iloc[]
# print(df_from_arrays.iloc[0:3 , 0:2])

print("===All Data ===")
print(df_from_arrays)
print("===Data at colum A greater than 3 : inside list ===")
print(df_from_arrays[df_from_arrays.A > 3])
print("===Data at colum A greater than 3 : Using loc ===")
print(df_from_arrays.loc[df_from_arrays.A > 3])
print("=== filter a columns with words")
print(df_from_arrays[df_from_arrays.C == "Fmale"])
print("===Filter with differe lables")
print(df_from_arrays[(df_from_arrays.A>3) & (df_from_arrays.C == 'Male')])
print("===Check for null vlaue")
null_result = df_from_arrays[df_from_arrays.C.isnull()]
print(null_result)
print("==Try to fill null value")
df_from_arrays.C.fillna(1000)
print(df_from_arrays)
print(df_from_arrays.groupby("C"))




#
# # download the data and name the columns
# cols = [
#     'age', 'workclass', 'fnlwgt', 'education', 'education_num',
#     'marital_status', 'occupation', 'relationship', 'ethnicity', 'gender',
#     'capital_gain', 'capital_loss', 'hours_per_week', 'country_of_origin',
#     'income'
# ]
#
# df = pd.read_csv(
#     'http://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data',
#     names=cols)
# print(df.head(5)) # first Five rows
# # print(df.tail(5)) # list five tows
# # print(df.columns)
#
#
#

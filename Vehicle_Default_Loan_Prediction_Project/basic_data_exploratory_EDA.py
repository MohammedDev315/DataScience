#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pandas.api.types import is_string_dtype

df = pd.read_csv("data/train.csv")
#%%
print(df.columns)
#%%
sns.pairplot(
    df.loc[: , ['DISBURSED_AMOUNT', 'ASSET_COST', 'LTV', 'BRANCH_ID','SUPPLIER_ID', "LOAN_DEFAULT"] ],
    hue = "LOAN_DEFAULT"
)
plt.show()
#%%
print(df[df.loc[: , "VOTERID_FLAG" ].isna()])
print(df["VOTERID_FLAG"].describe())

#%%
def show_basic_statics(col_name):
    print(f"==={col_name}===")
    #chek if cloumn has null
    TotalNull = len(df[df.loc[: , col_name ].isna()])
    print(f"Total Null : {TotalNull}  ")
    # check if text number has empty space
    if is_string_dtype(df[col_name]) :
        for text in df.loc[:, col_name]:
            if type(text) == str:
                if len(text.strip()) != len(text):
                    print(f"Please check this text :  {text} ")
    print(df.loc[:, col_name].describe())
    print("==========End========")

#%%
def show_relation_in_scatter(x , y):
    plt.scatter(df.loc[:, x] , df.loc[:, y])
    plt.show()

#%%
for col in df.columns:
    print(col)
    show_basic_statics(col)
    # show_relation_in_scatter(col , 'VOTERID_FLAG')


#%%
sns.pairplot(
    df.loc[: , ['MANUFACTURER_ID', 'CURRENT_PINCODE_ID', 'DATE_OF_BIRTH',
       'EMPLOYMENT_TYPE', 'DISBURSAL_DATE', 'STATE_ID', 'EMPLOYEE_CODE_ID', "LOAN_DEFAULT"] ],
    hue = "LOAN_DEFAULT"
)
plt.show()
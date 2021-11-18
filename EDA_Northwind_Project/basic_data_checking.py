import matplotlib.pyplot as plt
import pandas as pd
from pandas.api.types import is_string_dtype


df = pd.read_csv('northwind.csv')
df = df.astype({
    "CategoryCategoryName": str,
    "OrderShipCity": str,
    "OrderShipCountry" :str,
    "OrderShipRegion" :str
})


def show_basic_statics(col_name):
    print(f"==={col_name}===")
    #chek if cloumn has null
    print(f"No Null : {len(df[df.loc[:, col_name ].isnull() == False]) == len(df.loc[:, col_name ])} ")
    # check if text number has empty space
    if is_string_dtype(df[col_name]):
        for text in df.loc[:, col_name]:
            if len(text.strip()) != len(text):
                print(f"Please check this text :  {text} ")
    print(df.loc[:, col_name].describe())
    print("==========End========")


def show_relation_in_scatter(x , y):
    plt.scatter(x, df.loc[:, y])
    plt.show()


# show_basic_statics("OrderShipCity")
# show_basic_statics("OrderShipCountry")
# show_basic_statics("CategoryCategoryName")
















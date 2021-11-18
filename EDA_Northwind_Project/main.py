import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pandas.api.types import is_string_dtype
from pandas.api.types import is_numeric_dtype
import seaborn as sns
import squarify

df = pd.read_csv('northwind.csv')
df = df.astype({
    "CategoryCategoryName": str,
    "OrderShipCity": str,
    "OrderShipCountry" :str,
    "OrderShipRegion" :str
})


# print(df.info())
# print(df.loc[: , "OrderDetailUnitPrice" ].head())
# print(df.loc[df.loc[: , "OrderDetailUnitPrice" ] > 18 ].head())


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



# sns.boxplot(x=df.loc[: , "OrderDetailUnitPrice" ])
# sns.boxplot(x=df.loc[: , "PrductUnitPrice" ] , y = group_price_by_contry  )
# plt.show()

group_by_category_city = df.groupby(['CategoryCategoryName' , 'OrderShipRegion' ])[['OrderDetailQuantity']].sum()
group_by_region = df.groupby(['OrderShipRegion' ])[['OrderDetailQuantity']].sum()
print(group_by_region)

# extract the data and labels as lists
labels_group_by_region = group_by_region.index.get_level_values(0).tolist()
value_group_by_region  = group_by_region.reset_index().OrderDetailQuantity.values.tolist()
print(labels_group_by_region)
value_group_by_region_pesentages = []
for x in value_group_by_region:
    result = int((x / sum(value_group_by_region) ) * 100)
    value_group_by_region_pesentages.append(result)

for x in np.arange(len(labels_group_by_region)):
    labels_group_by_region[x] = f"{labels_group_by_region[x]} \n  {value_group_by_region_pesentages[x] } %"

colors = [plt.cm.Spectral(i/float(len(labels_group_by_region))) for i in range(len(labels_group_by_region))]

squarify.plot(sizes=value_group_by_region, color=colors , label=labels_group_by_region, alpha=.8)
plt.title('Value of Products Among Regions')
plt.axis('off')
plt.show()


mask_county = (df.loc[: , "OrderShipRegion" ] == "Western Europe")\
              | (df.loc[: , "OrderShipRegion" ] == "South America")\
              | (df.loc[: , "OrderShipRegion" ] == "North America")\
              | (df.loc[: , "OrderShipRegion" ] == "Southern Europe")\
              | (df.loc[: , "OrderShipRegion" ] == "British Isles")



# df_pivot = df.pivot_table(
#     index= "CategoryCategoryName", # the rows (turned into index)
#     columns= "OrderShipRegion", # the column values
#     values= "OrderDetailQuantity", # the field(s) to processed in each group
#     aggfunc=np.sum, # group operation
# )

# df_pivot = df.loc[ mask_county ].pivot_table(
#     index= "CategoryCategoryName", # the rows (turned into index)
#     columns= "OrderShipRegion", # the column values
#     values= "OrderDetailQuantity", # the field(s) to processed in each group
#     aggfunc=np.sum, # group operation
# )

#
# df_pivot.plot(kind='line')
# plt.show()


# print(df.columns)











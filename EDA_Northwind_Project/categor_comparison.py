import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

df = pd.read_csv('northwind.csv')
df = df.astype({
    "CategoryCategoryName": str,
    "OrderShipCity": str,
    "OrderShipCountry" :str,
    "OrderShipRegion" :str
})

mask_county = (df.loc[: , "OrderShipRegion" ] == "Western Europe")\
              | (df.loc[: , "OrderShipRegion" ] == "South America")\
              | (df.loc[: , "OrderShipRegion" ] == "North America")\
              | (df.loc[: , "OrderShipRegion" ] == "Southern Europe")\
              | (df.loc[: , "OrderShipRegion" ] == "British Isles")

#all Countries
df_pivot = df.pivot_table(
    index= "CategoryCategoryName", # the rows (turned into index)
    columns= "OrderShipRegion", # the column values
    values= "OrderDetailQuantity", # the field(s) to processed in each group
    aggfunc=np.sum, # group operation
)

#Using Mask
# df_pivot = df.loc[ mask_county ].pivot_table(
#     index= "CategoryCategoryName", # the rows (turned into index)
#     columns= "OrderShipRegion", # the column values
#     values= "OrderDetailQuantity", # the field(s) to processed in each group
#     aggfunc=np.sum, # group operation
# )

plt.figsize=(14, 16)
df_pivot.plot(kind='bar' , stacked = True)
plt.title("Popularity of Products Among Regions")
# plt.title("Popularity of Products Among Top five Regions")
plt.xlabel("Products")
plt.ylabel("Total of Products in Million")
ticks_y = np.linspace(0,4000000, 10 , dtype = int)
plt.yticks(ticks_y)
plt.legend(loc = 0)
plt.show()














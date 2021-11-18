import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


df = pd.read_csv('northwind2.csv')
df = df.astype({
    "OrderShipRegion" :str
})

df.drop_duplicates(inplace=True)

# Seafood   Beverages   Condiments   Confections
product_name = "Beverages"
region_name = "North America"

mask_county =  (df.loc[: , "OrderShipRegion" ] == region_name ) & (df.loc[: , "CategoryCategoryName"] == product_name )


df_pivot = df.loc[ mask_county ].pivot_table(
    index= "Month", # the rows (turned into index)
    columns= "OrderShipRegion", # the column values
    values= "OrderDetailQuantity", # the field(s) to processed in each group
    aggfunc=np.sum, # group operation
)

plt.figsize=(14, 16)
df_pivot.plot(kind='bar' , stacked = True)
plt.title(f"Popularity of {product_name} Among  {region_name} ")
# plt.title("Popularity of Products Among Top five Regions")
plt.xlabel("Months")
plt.ylabel("Total of Products in Thousands")
ticks_y = np.linspace(0,500000, 10 , dtype = int)
# plt.yticks(ticks_y)
plt.legend().set_visible(False)
plt.show()













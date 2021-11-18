import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import squarify

df = pd.read_csv('northwind.csv')
df = df.astype({
    "CategoryCategoryName": str,
    "OrderShipCity": str,
    "OrderShipCountry" :str,
    "OrderShipRegion" :str
})




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




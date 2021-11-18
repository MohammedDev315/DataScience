import matplotlib.pyplot as plt
from pywaffle import Waffle
import numpy as np
import pandas as pd

df = pd.read_csv('northwind.csv')
df = df.astype({
    "CategoryCategoryName": str,
    "OrderShipCity": str,
    "OrderShipCountry" :str,
    "OrderShipRegion" :str
})


group_by_region = df.groupby(['CategoryCategoryName'])[['OrderDetailQuantity']].sum()
labels_group_by_region = group_by_region.index.get_level_values(0).tolist()
value_group_by_region  = group_by_region.reset_index().OrderDetailQuantity.values.tolist()


value_group_by_region_pesentages = []
for x in value_group_by_region:
    result = int((x / sum(value_group_by_region) ) * 100)
    value_group_by_region_pesentages.append(result)

data = {}
for x in np.arange(len(value_group_by_region_pesentages)):
    k = labels_group_by_region[x]
    data[k] = value_group_by_region_pesentages[x]

fig = plt.figure(
    FigureClass=Waffle,
    rows=5,
    values=data,
    colors=["#d12c1f", "#00bfff" ,  "#ff0000", "#ffbf00" , "#40ff00" , "#0040ff" , "#8000ff" , "#a65959" ],
    title={'label': 'Popularity of Products Among Regions', 'loc': 'left'},
    labels=[f"{k} ({v}%)" for k, v in data.items()],
    legend={'loc': 'upper left', 'bbox_to_anchor': (1, 1)},
    starting_location='NW',
    block_arranging_style='snake'
)
fig.set_facecolor('#EEEEEE')
plt.show()




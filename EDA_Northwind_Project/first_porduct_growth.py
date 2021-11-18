import matplotlib.pyplot as plt
import pandas as pd


df = pd.read_csv("northwind2.csv")
df = df.astype({
    "OrderShipRegion" : str
})


def draw_line(regon_name , product):
    mask = (df.loc[ : , "CategoryCategoryName"] == product ) & (df.loc[ : , "OrderShipRegion"] == regon_name )
    regon_selected = df.loc[mask]
    group_by_date = regon_selected.groupby(["YearMonth"])[['OrderDetailQuantity']].sum()
    # result to list
    x_group_by_date = group_by_date.index.get_level_values(0).tolist()
    y_group_by_date  = group_by_date.reset_index().OrderDetailQuantity.values.tolist()
    # plot line
    plt.plot( x_group_by_date , y_group_by_date , label= regon_name )


draw_line( "Western Europe" , "Beverages")
draw_line( "South America" , "Beverages")

plt.legend()
plt.xticks(rotation=90)
plt.show()








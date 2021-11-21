
import pandas as pd


df = pd.read_csv('cereal.csv')

# print(df.sugars.value_counts())
# print(df.loc[: , ["sugars" , "calories"] ].describe())

mask1 = (df.calories > 100)
mask2 = (df.loc[: , "calories" ] > 100)
# print(df[mask1].head(10))
# print(df[mask2].head(10))


# print(df.head(10))

df['vit_25'] = df.vitamins >= 25

# print(df.head(10))


def get_vitamin_level(vitamins2):
    if vitamins2 > 99:
        return 'high'
    elif vitamins2 > 24:
        return 'medium'
    else:
        return 'low'

df['vit_l'] = df.loc[: , "vitamins"].apply(get_vitamin_level)

# print(df.groupby('type')[['calories', 'sugars']].median())

# print(df.info())


result = (df
 .groupby('type')[['mfr','calories', 'sugars', 'rating']]
 .agg({
     'mfr': 'count',
     'calories': 'median',
     'sugars': 'mean',
     'rating': 'mean'
 })
)

# print(result)


df = pd.DataFrame(
    [[123, 'xt23', 20], [123, 'q45', 2], [123, 'a89', 25], [77, 'q45', 3],
     [77, 'a89', 30], [92, 'xt23', 24], [92, 'm33', 63], [92, 'a89', 28]],
    columns=['userid', 'product', 'price'])

print(df['price'].max())
print(df[df.loc[:,"price"] == df['price'].max() ])

print( df.groupby("userid")[["price"]].max() )
print( df.groupby("userid").sum() )



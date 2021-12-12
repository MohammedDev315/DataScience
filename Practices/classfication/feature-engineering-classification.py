#%%
# from collections import OrderedDict
import pandas as pd
import seaborn as sns
sns.set()

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score


df_orders = pd.read_csv('orders_subset.csv')
df_orders.head(3)

df_order_products_prior = pd.read_csv('order_products__prior_subset.csv')
df_order_products_prior.head(3)
df_order_products_train = pd.read_csv('order_products__train_subset.csv')
df_order_products_train.head(3)

df_order_products_train = df_order_products_train.merge(df_orders.drop('eval_set', axis=1), on='order_id')
df_order_products_prior = df_order_products_prior.merge(df_orders.drop('eval_set', axis=1), on='order_id')


df_user_product = (df_order_products_prior.groupby(['product_id','user_id'],as_index=False)
                                          .agg({'order_id':'count'})
                                          .rename(columns={'order_id':'user_product_total_orders'}))

train_ids = df_order_products_train['user_id'].unique()
df_X = df_user_product[df_user_product['user_id'].isin(train_ids)]
train_carts = (df_order_products_train.groupby('user_id',as_index=False)
                                      .agg({'product_id':(lambda x: set(x))})
                                      .rename(columns={'product_id':'latest_cart'}))

df_X = df_X.merge(train_carts, on='user_id')
df_X['in_cart'] = (df_X.apply(lambda row: row['product_id'] in row['latest_cart'], axis=1).astype(int))
target_pcts = df_X.in_cart.value_counts(normalize=True)
print(target_pcts)

target_pcts.plot(kind='bar')

def plot_features(df, sample_size=500):
    sample = (df.drop(['product_id', 'user_id', 'latest_cart'], axis=1)
              .sample(1000, random_state=44))
    sns.pairplot(sample, hue='in_cart', plot_kws=dict(alpha=.3, edgecolor='none'))

plot_features(df_X)


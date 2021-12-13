#%%
# from collections import OrderedDict
import pandas as pd
import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import OrderedDict

sns.set()

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score


df_orders = pd.read_csv('data/orders_subset.csv')
df_orders.head(3)

df_order_products_prior = pd.read_csv('data/order_products__prior_subset.csv')
df_order_products_prior.head(3)
df_order_products_train = pd.read_csv('data/order_products__train_subset.csv')
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

target_pcts = df_X.in_cart.value_counts(normalize=True)
print(target_pcts)

target_pcts.plot(kind='bar')
plt.show()

#%%
def plot_features(df, sample_size=500):
    sample = (df.drop(['product_id', 'user_id', 'latest_cart'], axis=1)
              .sample(1000, random_state=44))
    sns.pairplot(sample, hue='in_cart', plot_kws=dict(alpha=.3, edgecolor='none'))
plot_features(df_X)

#%%
def get_user_split_data(df, test_size=.2, seed=42):
    rs = np.random.RandomState(seed)
    total_users = df['user_id'].unique()
    test_users = rs.choice(total_users,
                           size=int(total_users.shape[0] * test_size),
                           replace=False)
    df_tr = df[~df['user_id'].isin(test_users)]
    df_te = df[df['user_id'].isin(test_users)]
    y_tr, y_te = df_tr['in_cart'], df_te['in_cart']
    X_tr = df_tr.drop(['product_id', 'user_id', 'latest_cart', 'in_cart'], axis=1)
    X_te = df_te.drop(['product_id', 'user_id', 'latest_cart', 'in_cart'], axis=1)
    return X_tr, X_te, y_tr, y_te

X_tr, X_te, y_tr, y_te = get_user_split_data(df_X)
lr = LogisticRegression(solver='lbfgs')
lr.fit(X_tr, y_tr)
print(f1_score(lr.predict(X_te), y_te))
print(lr.coef_)
#%%

prod_features = ['product_total_orders','product_avg_add_to_cart_order']
df_prod_features = (df_order_products_prior.groupby(['product_id'] , as_index=False)
                    .agg(OrderedDict([ ('order_id' , 'nunique' )  , ('add_to_cart_order' , 'mean') ] )))
df_prod_features.columns = ['product_id'] + prod_features
df_prod_features.head()
df_X = df_X.merge(df_prod_features, on='product_id')
print(df_X.head())
#%%
X_tr, X_te, y_tr, y_te = get_user_split_data(df_X)
lr = LogisticRegression()
lr.fit(X_tr, y_tr)
print(f1_score(lr.predict(X_te), y_te))
#%%
user_features = ['user_total_orders','user_avg_cartsize','user_total_products','user_avg_days_since_prior_order']

df_user_features = (df_order_products_prior.groupby(['user_id'],as_index=False)
                                           .agg(OrderedDict(
                                                   [('order_id',['nunique', (lambda x: x.shape[0] / x.nunique())]),
                                                    ('product_id','nunique'),
                                                    ('days_since_prior_order','mean')])))

df_user_features.columns = ['user_id'] + user_features
df_user_features.head()
df_X = df_X.merge(df_user_features, on='user_id')
df_X = df_X.dropna() # note that this is naive NaN handling for simplicity
df_X.head(1)
#%%
X_tr, X_te, y_tr, y_te = get_user_split_data(df_X)
lr = LogisticRegression(max_iter=1000)
lr.fit(X_tr, y_tr)
print(f1_score(lr.predict(X_te), y_te)) #0.1275
#%%

user_prod_features = ['user_product_avg_add_to_cart_order']
df_user_prod_features = (df_order_products_prior.groupby(['product_id','user_id'],as_index=False) \
                                                .agg(OrderedDict(
                                                     [('add_to_cart_order','mean')])))

df_user_prod_features.columns = ['product_id','user_id'] + user_prod_features
df_user_prod_features.head()

df_X = df_X.merge(df_user_prod_features,on=['user_id','product_id'])
df_X['user_product_order_freq'] = df_X['user_product_total_orders'] / df_X['user_total_orders']
df_X.head(1)
#%%
X_tr, X_te, y_tr, y_te = get_user_split_data(df_X)
lr = LogisticRegression(C=1, max_iter=1000)
lr.fit(X_tr, y_tr)
print(f1_score(lr.predict(X_te), y_te))
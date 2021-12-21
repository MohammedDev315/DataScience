#%%
import nltk
import pandas as pd
import numpy as np
import re
import string
from sklearn.feature_extraction.text import CountVectorizer

#%%
data = pd.read_csv('data/coffee.csv')
data.head()
#%%
data = data[data.stars!=3]
mask1 = (data.loc[:,'stars'] != 3 )
data = data[mask1]
print(data.loc[: , 'stars' ].value_counts(normalize=True))
#%%
# Set 4/5 star reviews to positive, the rest to negative
data['sentiment'] = np.where(data['stars'] >= 4, 'positive', 'negative')

#%%
# Include only the sentiment and reviews columns
data = data[['sentiment', 'reviews']]
data.head()
#%%
data.sentiment.value_counts(normalize=True)
#%%
alphanumeric = lambda x: re.sub('\w*\d\w*', ' ', x)
punc_lower = lambda x: re.sub('[%s]' % re.escape(string.punctuation), ' ', x.lower())
data['reviews'] = data.reviews.map(alphanumeric).map(punc_lower)
data.head()
#%%
X = data.reviews
y = data.sentiment
# Split the data into training and test sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
#%%
text = ['Hello my name is james',
'james this is my python notebook',
'james trying to create a big dataset']
coun_vect = CountVectorizer(lowercase=True , stop_words= ['is','to','my' , 'big' ])
#%%
data_fitted = coun_vect.fit(text)
data_transformed = data_fitted.transform(text)
data_to_array = data_transformed.toarray()
print(data_to_array)
df_t2 = pd.DataFrame(data = data_to_array , columns=coun_vect.get_feature_names())
print(df_t2.head())
#%%

count_matrix = coun_vect.fit_transform(text)
print(count_matrix)
#%%
count_array = count_matrix.toarray()
print(count_array)
#%%
print(coun_vect.get_feature_names())
#%%
df22 = pd.DataFrame(data=count_array,columns = coun_vect.get_feature_names())








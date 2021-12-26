#%%
import pandas as pd
import numpy as np
import scipy.sparse as ss
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import datasets
from corextopic import corextopic as ct
from corextopic import vis_topic as vt
import matplotlib.pyplot as plt
import seaborn as sns
#%%
categories = ['alt.atheism', 'comp.graphics', 'rec.sport.baseball',
              'rec.motorcycles', 'sci.space', 'talk.politics.mideast']

ng_train = datasets.fetch_20newsgroups(subset='train', categories=categories,
                                      remove=('headers', 'footers', 'quotes'))

# Throw it into a dataframe
df = pd.DataFrame({'data':ng_train['data'], 'target':ng_train['target']})
df.target = df.target.map({i:v for i, v in enumerate(ng_train['target_names'])})
df.head()
#%%
vectorizer = CountVectorizer(max_features=20000,
                             stop_words='english', token_pattern="\\b[a-z][a-z]+\\b",
                             binary=True)

doc_word = vectorizer.fit_transform(df.data)
words = list(np.asarray(vectorizer.get_feature_names()))
#%%
# I recommend adding docs=df.data to make it easier to check which sentences are in each resulting topic
topic_model = ct.Corex(n_hidden=6, words=words, seed=1)
topic_model.fit(doc_word, words=words, docs=df.data)

#%%
# Print all topics from the CorEx topic model
topics = topic_model.get_topics()
print(topics)
#%%
for n ,topic in enumerate(topics):
    topic_words ,_,_ = zip(*topic)
    print('{}: '.format(n) + ','.join(topic_words))

#%%
# Let's check out topic : graphics
topic_model.get_top_docs(topic=4, n_docs=2)

#%%
predictions = pd.DataFrame(topic_model.predict(doc_word), columns=['topic'+str(i) for i in range(6)])
predictions.head(3)

#%%
topic_model = ct.Corex(n_hidden=6 , words=words , max_iter=200 , verbose=False , seed=1)
topic_model.fit(doc_word, words=words, docs=df.data,
                anchors=[['atheism', 'god', 'religious'],
                         ['graphics'],
                         ['baseball'],
                         ['motorcycle', 'ride'],
                         ['space'],
                         ['politics','armenians', 'jews']], anchor_strength=2)

#%%
topic_model = ct.Corex(n_hidden=6, words=words,
                       max_iter=200, verbose=False, seed=1)

topic_model.fit(doc_word, words=words, docs=df.data,
                anchors=[['nasa', 'politics'], 'nasa'], anchor_strength=10)

#%%
# Print all topics from the CorEx topic model
topics = topic_model.get_topics()
for n,topic in enumerate(topics):
    topic_words,_,_ = zip(*topic)
    print('{}: '.format(n) + ','.join(topic_words))

#%%
topic_model = ct.Corex(n_hidden=8, words=words,
                       max_iter=200, verbose=False, seed=1)

topic_model.fit(doc_word, words=words, docs=df.data,
                anchors=[['truth'], ['truth']], anchor_strength=5)

#%%
# Print all topics from the CorEx topic model
topics = topic_model.get_topics()
for n,topic in enumerate(topics):
    topic_words,_,_ = zip(*topic)
    print('{}: '.format(n) + ','.join(topic_words))



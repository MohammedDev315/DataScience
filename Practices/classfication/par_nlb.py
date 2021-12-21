#%%
import nltk
import pandas as pd
import string
import pandas as pd
import numpy as np
from collections import Counter
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
#%%
df = pd.read_csv('data/amazon_cells_labelled.txt' ,  names=['text' , 'sentiment' ] ,   delimiter="\t")
X_train, X_test, y_train, y_test = train_test_split(df.text, df.sentiment, test_size=0.2, random_state=2018)

#%%
class My_countVectorizer:
    def fit_vocab(self , corpus):
        self.counter = Counter()
        for doc in corpus:
            self.counter.update(doc.split(' ') )
        self.feature_map = { word : i for i , (word , count) in enumerate(self.counter.most_common()) }
    def transform(self , corpus):
        vectors = []
        for doc in corpus:
            vector = np.zeros(len(self.feature_map))
            for word in doc.split(' '):
                if word in self.feature_map:
                    vector[self.feature_map[word]] += 1
            vectors.append(vector)
        word_df = pd.DataFrame(vectors , columns=self.feature_map.keys())
        return word_df
#%%
cv = My_countVectorizer()
cv.fit_vocab(X_train)
X_train = cv.transform(X_train)
X_test = cv.transform(X_test)
print(X_train.head())
#%%
nb = MultinomialNB()
nb.fit(X_train , y_train)
print(nb.score(X_test , y_test))
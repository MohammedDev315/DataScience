#%%
# gensim
from gensim import corpora, models, similarities, matutils
import ssl
ssl._create_default_https_context = ssl._create_unverified_context
# sklearn
from sklearn import datasets
from sklearn.feature_extraction.text import CountVectorizer

# logging for gensim (set to INFO)
import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
categories = ['alt.atheism', 'comp.graphics', 'rec.sport.baseball',
              'rec.motorcycles', 'sci.space', 'talk.politics.mideast']

# Download the training subset of the 20 NG dataset, with headers, footers, quotes removed
# Only keep docs from the 6 categories above
ng_train = datasets.fetch_20newsgroups(subset='train', categories=categories,
                                      remove=('headers', 'footers', 'quotes'))
#%%
print(ng_train.data[0])
# Create a CountVectorizer for parsing/counting words
count_vectorizer = CountVectorizer(ngram_range=(1, 2),
                                   stop_words='english', token_pattern="\\b[a-z][a-z]+\\b")
count_vectorizer.fit(ng_train.data)
#%%
doc_word = count_vectorizer.transform(ng_train.data).transpose()
import pandas as pd
pd.DataFrame(doc_word.toarray(), count_vectorizer.get_feature_names()).head()
#%%
id2word = dict((v, k) for k, v in count_vectorizer.vocabulary_.items())
len(id2word)
#%%
corpus = matutils.Sparse2Corpus(doc_word)
lda = models.LdaModel(corpus=corpus, num_topics=3, id2word=id2word, passes=5)
lda.print_topics()
#%%
# Transform the docs from the word space to the topic space (like "transform" in sklearn)
lda_corpus = lda[corpus]
lda_corpus
# Store the documents' topic vectors in a list so we can take a peak
lda_docs = [doc for doc in lda_corpus]
lda_docs[0:5]
ng_train.data[0]
#%%



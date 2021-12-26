#%%
import spacy
from spacy.tokens import Span
#pip install spacy && python -m spacy download en
nlp = spacy.load('en_core_web_sm')

doc = nlp("Welcome to this test with me, I am very happy to see you")

print("=========")
print(doc)

for x in doc:
    print(x.text)
    print(x.shape_)
    print(x.is_stop)
    print(x.is_alpha)
    print("----------")

#%%
doc2 = nlp(
"""  
When trying to run a project in PyCharm I get the following error. Next May, I have setup the virtualbox , vagrant and all the requirements in the readme file they gave me but when pressing run...
"""
)
for x in doc2:
    print(f"{x.text} -- {x.i} ")
#%%
print(doc2[22:23])
#%%
#Print sentences
for sent in doc2.sents:
    print(sent.text)

#%%
#Important words
def check_doc(doc):
    if doc.ents:
        for ent in doc.ents:
            print(f"{ent.text}  - {ent.label_}  -  {str(spacy.explain(ent.label_))}  ")

check_doc(doc2)
#%%
#Becuase virtualbox Can not be defined in the privous def, we  have to add it
ORG = doc.vocab.strings[u'ORG'] #This  lable come form lable  of spacy ORG:ogrginaization
new_ent = Span(doc2 , 22,23 , label=ORG)
#%%
doc2.ents = list(doc2.ents) + [new_ent]
check_doc(doc2)

#%%
for chunk in doc2.noun_chunks:
    print(chunk)

#%%
# it is between POS and NER
for x in doc2:
    print(f"{x.is_sent_start} == > {x} ")

#%%
#we can add top work by uswing:
nlp.Defaults.stop_words.add("DDD")

#%%
#convevert generator to list
doc_as_list = [sent for sent in doc2.sents]
sentenc_num = 1
print(f"sentence{sentenc_num} sartedAt=> {doc_as_list[sentenc_num].start} endAt=> {doc_as_list[sentenc_num].end} ")

#%%

#######################
### Tokenizer     #####
#######################
#%%
import ssl
ssl._create_default_https_context = ssl._create_unverified_context
from nltk.tokenize import word_tokenize ,sent_tokenize
import nltk
# nltk.download('punkt')
# nltk.download('stopwords')
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
import numpy as np
from numpy.linalg import norm
#%%
ex_word = """  When trying to run a project in PyCharm I get the following error. I have setup the virtualbox, vagrant and all the requirements in the readme file they gave me but when pressing run...
"""

print(word_tokenize(ex_word))
print(sent_tokenize(ex_word))

#%%
stop_work = set(stopwords.words("english"))

#%%
orgin_text = list(word_tokenize(ex_word))
filterd_text = [w for w in word_tokenize(ex_word) if w not in stop_work]
print(orgin_text)
print(filterd_text)
#%%
cv = CountVectorizer(stop_words="english")
text_as_list = sent_tokenize(ex_word)
X = cv.fit_transform(text_as_list)
data_df =  pd.DataFrame(X.toarray() , columns=cv.get_feature_names())
#%%
cv_ifidf = TfidfVectorizer(stop_words='english')
X = cv_ifidf.fit_transform(sent_tokenize(ex_word))
data_df_ifidf = pd.DataFrame(X.toarray() , columns=cv_ifidf.get_feature_names())

#%%
consin_simlarty = lambda v1 , v2 : np.dot(v1 , v2) / (norm(v1) * norm(v2) )
print(consin_simlarty([1,1,1,0] , [0,1,1,1] ) )
#%%
print( consin_simlarty( np.array(data_df.iloc[0]) ,np.array(data_df.iloc[1])  ) )
print( consin_simlarty( np.array(data_df_ifidf.iloc[0]) ,np.array(data_df_ifidf.iloc[1])  ) )
#%%


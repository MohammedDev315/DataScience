#%%
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.decomposition import NMF
from sklearn.metrics.pairwise import cosine_similarity
#%%
num_of_topic = 20
file_path = "/Volumes/Lexar/DataScience/Disease_Diagnosis_NLP_Project/experiment_files/Data/df_signs_and_diagnosis.csv"
df = pd.read_csv(file_path , names=["Title" , "Des"])
#%%
vectorizer = CountVectorizer(stop_words='english')
doc_word = vectorizer.fit_transform(df["Des"])
print(doc_word.shape)
#%%
pd_vectorizer = pd.DataFrame(doc_word.toarray(), index=df["Des"], columns=vectorizer.get_feature_names())
#%%
lsa = TruncatedSVD(num_of_topic)
doc_topic = lsa.fit_transform(doc_word)
print(lsa.explained_variance_ratio_)
#%%
topic_word = pd.DataFrame(lsa.components_.round(3),
             index = ['component'+str(i) for i in range(num_of_topic)],
             columns = vectorizer.get_feature_names())
#%%
print(topic_word)
#%%
def display_topics(model, feature_names, no_top_words, topic_names=None):
    for ix, topic in enumerate(model.components_):
        if not topic_names or not topic_names[ix]:
            print("\nTopic ", ix)
        else:
            print("\nTopic: '",topic_names[ix],"'")
        print(", ".join([feature_names[i]
                        for i in topic.argsort()[:-no_top_words - 1:-1]]))
#%%
display_topics(lsa, vectorizer.get_feature_names(), 10)
#%%
Vt = pd.DataFrame(doc_topic.round(5),
             index = df["Des"],
             columns = ['component'+str(i) for i in range(num_of_topic)])
#%%



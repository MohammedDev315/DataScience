#%%
import spacy
nlp = spacy.load('en_core_web_sm')
review = "I'm so happy I went to this awesome Vegas buffet!"
doc = nlp(review) # Result spaCy Documents
from spacy  import displacy
from collections import Counter


#%%
for token in doc:
    print(f" {token.text} - {token.pos_} - {token.lemma} - {token.is_stop} ")

#%%
for sent in doc.sents:
    print(sent)

#%%
for token in doc:
    print(f" {token.text} -- {token.dep_} ")

#%%
displacy.render(doc , style='dep' , options={'distance' : 80} )

#%%
for token in doc:
    if token.dep_ == "amod" :
        print(f"{token.text}  --> {token.head}")

#%%
print(spacy.explain("amod"))
print(spacy.explain("ROOT"))
print(spacy.explain("prep"))
print(spacy.explain("acomp"))
#%%
for ent in doc.ents:
    print(f" {ent.text} -- {ent.label_} ")

print(spacy.explain("GPE"))

#%%
for ent in doc.ents:
    print(ent.text, ent.label_)
spacy.explain("GPE")
displacy.render(doc, style='ent', jupyter=True)
#%%
document = nlp(
    "One year ago, I visited the Eiffel Tower with Jeff in Paris, France."
    )
displacy.render(document, style='ent', jupyter=True)
spacy.explain("FAC")

#%%
import pandas as pd
import ssl
ssl._create_default_https_context = ssl._create_unverified_context
pd.set_option('max_colwidth', 100)
url = 'http://bit.ly/375FDrO'  #Kaggle dataset
df = pd.read_csv(url, sep='\t')
df.shape
df.columns = ['text', 'rating']
df.head()

#%%
df['spacy_doc'] = list(nlp.pipe(df.text))
print(df.head())
#%%
print(df["spacy_doc"][0])
#%%
positive_reviews = df[df.rating == 1]
negative_reviews = df[df.rating == 0]
#%%
pos_adj = [token.text.lower() for doc in positive_reviews.spacy_doc for token in doc if token.pos_=='ADJ']
neg_adj = [token.text.lower() for doc in negative_reviews.spacy_doc for token in doc if token.pos_=='ADJ']

pos_noun = [token.text.lower() for doc in positive_reviews.spacy_doc for token in doc if token.pos_=='NOUN']
neg_noun = [token.text.lower() for doc in negative_reviews.spacy_doc for token in doc if token.pos_=='NOUN']

#%%
print(Counter(pos_adj).most_common(10))
print(Counter(neg_adj).most_common(10))
print(Counter(pos_noun).most_common(10))
print(Counter(neg_noun).most_common(10))

#%%
from spacy.symbols import amod
from pprint import pprint

def get_amods(noun, ser):
    amod_list = []
    for doc in ser:
        for token in doc:
            if (token.text) == noun:
                for child in token.children:
                    if child.dep == amod:
                        amod_list.append(child.text.lower())
    return sorted(amod_list)




def amods_by_sentiment(noun):
    print(f"Adjectives describing {str.upper(noun)}:\n")

    print("POSITIVE:")
    print(get_amods(noun, positive_reviews.spacy_doc))

    print("\nNEGATIVE:")
    print(get_amods(noun, negative_reviews.spacy_doc))




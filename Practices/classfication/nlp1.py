#%%
import nltk
import pandas as pd
import numpy as np
import re
import string
from sklearn.linear_model import LogisticRegression

data = pd.read_csv('coffee.csv')
data.head()

data.stars.value_counts(normalize=True)


# Remove 3 star reviews
data = data[data.stars!=3]

# Set 4/5 star reviews to positive, the rest to negative
data['sentiment'] = np.where(data['stars'] >= 4, 'positive', 'negative')

# Include only the sentiment and reviews columns
data = data[['sentiment', 'reviews']]


# Note that the dataset has mostly positive reviews
data.sentiment.value_counts(normalize=True)

# Text preprocessing steps - remove numbers, captial letters and punctuation
alphanumeric = lambda x: re.sub('\w*\d\w*', ' ', x)
punc_lower = lambda x: re.sub('[%s]' % re.escape(string.punctuation), ' ', x.lower())


data['reviews'] = data.reviews.map(alphanumeric).map(punc_lower)
print(data.head())
#%%

# Split the data into X and y data sets
X = data.reviews
y = data.sentiment

X=data.reviews


# Split the data into training and test sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# The first document-term matrix has default Count Vectorizer values - counts of unigrams
from sklearn.feature_extraction.text import CountVectorizer

cv1 = CountVectorizer(stop_words='english')

X_train_cv1 = cv1.fit_transform(X_train)
X_test_cv1  = cv1.transform(X_test)

pd.DataFrame(X_train_cv1.toarray(), columns=cv1.get_feature_names()).head()

# The second document-term matrix has both unigrams and bigrams, and indicators instead of counts
cv2 = CountVectorizer(ngram_range=(1,2), binary=True, stop_words='english')

X_train_cv2 = cv2.fit_transform(X_train)
X_test_cv2  = cv2.transform(X_test)

pd.DataFrame(X_train_cv2.toarray(), columns=cv2.get_feature_names()).head()

#%%
lr = LogisticRegression()
lr.fit(X_train_cv1 , y_train)
y_pred_cv1 = lr.predict(X_test_cv1)
#%%
lr.fit(X_train_cv2 , y_train)
y_pred_cv2 = lr.predict(X_test_cv2)
#%%
# Create a function to calculate the error metrics, since we'll be doing this several times
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns


def conf_matrix(actual, predicted):
    cm = confusion_matrix(actual, predicted)
    sns.heatmap(cm, xticklabels=['predicted_negative', 'predicted_positive'],
                yticklabels=['actual_negative', 'actual_positive'], annot=True,
                fmt='d', annot_kws={'fontsize':20}, cmap="YlGnBu")
    plt.show()

    true_neg, false_pos = cm[0]
    false_neg, true_pos = cm[1]

    accuracy = round((true_pos + true_neg) / (true_pos + true_neg + false_pos + false_neg),3)
    precision = round((true_pos) / (true_pos + false_pos),3)
    recall = round((true_pos) / (true_pos + false_neg),3)
    f1 = round(2 * (precision * recall) / (precision + recall),3)

    cm_results = [accuracy, precision, recall, f1]
    return cm_results

#%%
# The heat map for the first logistic regression model
cm1 = conf_matrix(y_test, y_pred_cv1)
# The heat map for the second logistic regression model
cm2 = conf_matrix(y_test, y_pred_cv2)
#%%
# Compile all of the error metrics into a dataframe for comparison
results = pd.DataFrame(list(zip(cm1 , cm2)))
results = results.set_index([['Accuracy', 'Precision', 'Recall', 'F1 Score']])
results.columns = ['LogReg1' , 'LogReg2']
print(results)
#%%
# Fit the first Naive Bayes model
from sklearn.naive_bayes import MultinomialNB
mnb = MultinomialNB()
mnb.fit(X_train_cv1, y_train)
y_pred_cv1_nb = mnb.predict(X_test_cv1)

#%%
mnb = MultinomialNB()
mnb.fit(X_train_cv1 , y_train)
y_pred_cv1_nb = mnb.predict(X_test_cv1)
# Fit the second Naive Bayes model
#%%
from sklearn.naive_bayes import BernoulliNB
bnb = BernoulliNB()
bnb.fit(X_train_cv2, y_train)
y_pred_cv2_nb = bnb.predict(X_test_cv2)
#%%
# Here's the heat map for the first Naive Bayes model
cm3 = conf_matrix(y_test, y_pred_cv1_nb)
# Here's the heat map for the second Naive Bayes model
cm4 = conf_matrix(y_test, y_pred_cv2_nb)
#%%
# Compile all of the error metrics into a dataframe for comparison
results_nb = pd.DataFrame(list(zip(cm3, cm4)))
results_nb = results_nb.set_index([['Accuracy', 'Precision', 'Recall', 'F1 Score']])
results_nb.columns = ['NB1', 'NB2']
results = pd.concat([results, results_nb], axis=1)
#%%
# Create TF-IDF versions of the Count Vectorizers created earlier in the exercise
from sklearn.feature_extraction.text import TfidfVectorizer

tfidf1 = TfidfVectorizer(stop_words='english')
X_train_tfidf1 = tfidf1.fit_transform(X_train)
X_test_tfidf1 = tfidf1.transform(X_test)

tfidf2 = TfidfVectorizer(ngram_range=(1,2) , binary=True , stop_words='english')
X_train_tfidf2 = tfidf2.fit_transform(X_train)
X_test_tfidf2 = tfidf2.transform(X_test)
#%%
lr.fit(X_train_tfidf1, y_train)
y_pred_tfidf1_lr = lr.predict(X_test_tfidf1)
cm5 = conf_matrix(y_test, y_pred_tfidf1_lr)
#%%
lr.fit(X_train_tfidf2, y_train)
y_pred_tfidf2_lr = lr.predict(X_test_tfidf2)
cm6 = conf_matrix(y_test, y_pred_tfidf2_lr)
#%%
mnb.fit(X_train_tfidf1.toarray(), y_train)
y_pred_tfidf1_nb = mnb.predict(X_test_tfidf1)
cm7 = conf_matrix(y_test, y_pred_tfidf1_nb)
#%%
bnb.fit(X_train_tfidf2.toarray(), y_train)
y_pred_tfidf2_nb = bnb.predict(X_test_tfidf2)
cm8 = conf_matrix(y_test, y_pred_tfidf2_nb)
#%%
# Compile all of the error metrics into a dataframe for comparison
results_tf = pd.DataFrame(list(zip(cm5, cm6, cm7, cm8)))
results_tf = results_tf.set_index([['Accuracy', 'Precision', 'Recall', 'F1 Score']])
results_tf.columns = ['LR1-TFIDF', 'LR2-TFIDF', 'NB1-TFIDF', 'NB2-TFIDF']
results_tf

results = pd.concat([results, results_tf], axis=1)
results

class nlp_pipe:
    def __init__(self, vectorizer, cleaning_function, tokenizer, stemmer):
        self.vectorizer = vectorizer
        self.cleaning_function = cleaning_function
        self.tokenizer = tokenizer
        self.stemmer = stemmer
    def fit(self, text_to_fit_on):
        pass
    def transform(self, text_to_clean):
        pass


def print_the_word_bob_three_times():
    for i in range(3):
        print('bob')


class this_is_an_example:
    def __init__(self, function_input):
        self.function_to_run = function_input
    def do_the_thing(self):
        self.function_to_run()

class pre_porcessing:
    def __init__(self , CountVectorizer = None, simple_cleaning_function_i_made, TreebankWordTokenizer =None, PorterStemmer = None ):
        if 



from sklearn.feature_extraction.text import CountVectorizer
from nltk.tokenize import TreebankWordTokenizer
from nltk.stem import PorterStemmer

nlp = nlp_pipe(CountVectorizer(), simple_cleaning_function_i_made, TreebankWordTokenizer(), PorterStemmer())
nlp.fit(train_corpus)
nlp.transform(test_corpus)





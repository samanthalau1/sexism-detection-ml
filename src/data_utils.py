# methods for importing and cleaning data

import pandas as pd
import re
import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk import PorterStemmer, WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from gensim.models import Word2Vec


# load data from files
def load_data():
    train_data = pd.read_csv('EXIST2021_training.tsv', sep="\t")
    test_data = pd.read_csv('EXIST2021_test_labeled.tsv', sep="\t")
    return train_data, test_data


# remove unecessary columns
def clean_cols(train_data, test_data):
    train_clean_data = train_data.drop(columns=['test_case', 'id', 'source', 'task2'])
    test_clean_data = test_data.drop(columns=['test_case', 'id', 'source', 'task2'])
    return train_clean_data, test_clean_data


# makes text lowercase, removes users, and removes punctuation and links
def clean_text(train_clean_data, test_clean_data):

    def remove_users(text, pattern):
        r = re.findall(pattern, text)
        for i in r:
            text = re.sub(i,"",text) 
        return text
    
    train_clean_data['cleantext'] = np.vectorize(remove_users)(train_clean_data['text'], "@[\w]*")
    test_clean_data['cleantext'] = np.vectorize(remove_users)(test_clean_data['text'], "@[\w]*")

    train_clean_data['cleantext'] = train_clean_data['cleantext'].apply(str.lower)
    train_clean_data['cleantext'] = train_clean_data['cleantext'].str.replace(r"(@\[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)|^rt|http.+?", " ", regex = True)

    test_clean_data['cleantext'] = test_clean_data['cleantext'].apply(str.lower)
    test_clean_data['cleantext'] = test_clean_data['cleantext'].str.replace(r"(@\[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)|^rt|http.+?", " ", regex = True)
    return train_clean_data, test_clean_data


# remove stopwords from text
def remove_stop(train_clean_data, test_clean_data): 
    stop = stopwords.words(fileids=('english', 'spanish'))

    train_clean_data['cleantext'] = train_clean_data['cleantext'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop)]))
    test_clean_data['cleantext'] = test_clean_data['cleantext'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop)]))
    return train_clean_data, test_clean_data


# split by language, using only english
def use_eng(train_clean_data, test_clean_data):
    train_engtweets = train_clean_data.loc[train_clean_data['language'] == 'en']
    train_engtweets = train_engtweets.drop(columns = ['language', 'text'])

    test_engtweets = test_clean_data.loc[test_clean_data['language'] == 'en']
    test_engtweets = test_engtweets.drop(columns = ['language', 'text'])
    return train_engtweets, test_engtweets


# stemming and lemmatization
def stem_and_lemm(train_engtweets, test_engtweets):

    train_token_engtweets = train_engtweets['cleantext'].apply(lambda x: x.split())
    test_token_engtweets = test_engtweets['cleantext'].apply(lambda x: x.split())

    ps = PorterStemmer()
    wnl = WordNetLemmatizer()

    train_token_engtweets = train_token_engtweets.apply(lambda x: [wnl.lemmatize(i) if wnl.lemmatize(i).endswith('e') else ps.stem(i) for i in x])
    test_token_engtweets = test_token_engtweets.apply(lambda x: [wnl.lemmatize(i) if wnl.lemmatize(i).endswith('e') else ps.stem(i) for i in x])

    #recombine tokens
    for i in range(len(train_token_engtweets)):
        train_token_engtweets[i] = ' '.join(train_token_engtweets[i])
    
    train_engtweets['cleantext'] = train_token_engtweets

    for i in range(len(test_token_engtweets)):
        test_token_engtweets[i] = ' '.join(test_token_engtweets[i])
    
    test_engtweets['cleantext'] = test_token_engtweets

    return train_engtweets, test_engtweets


# tfidf feature extraction
def tfidf_extract(train_engtweets, test_engtweets):
    vect = TfidfVectorizer().fit(train_engtweets['cleantext'])
    X_train_engtweets = vect.transform(train_engtweets['cleantext'])
    X_test_engtweets = vect.transform(test_engtweets['cleantext'])
    return X_train_engtweets, X_test_engtweets


# word2vec feature extraction
def word2vec_extract(train_engtweets, test_engtweets):
    sentences = [sentence.split() for sentence in train_engtweets['cleantext']]
    w2v_model = Word2Vec(sentences, vector_size=100, window=7, min_count=2, workers=4, sg=1, hs=1, epochs=5, negative=5)

    def vectorize(sentence):
        words = sentence.split()
        words_vecs = [w2v_model.wv[word] for word in words if word in w2v_model.wv]
        if len(words_vecs) == 0:
            return np.zeros(100)
        words_vecs = np.array(words_vecs)
        return words_vecs.mean(axis=0)

    X_train_engtweets = np.array([vectorize(sentence) for sentence in train_engtweets['cleantext']])
    X_test_engtweets = np.array([vectorize(sentence) for sentence in test_engtweets['cleantext']])
    return X_train_engtweets, X_test_engtweets


# creating y for model
def split_y(train_engtweets, test_engtweets):
    y_train_engtweets = train_engtweets['task1']
    y_test_engtweets = test_engtweets['task1']
    return y_train_engtweets, y_test_engtweets
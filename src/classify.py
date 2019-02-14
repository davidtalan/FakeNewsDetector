import pandas as pd
import numpy as np
import pickle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.externals import joblib

def detect():
    df = pd.read_csv("/home/david/2019-ca400-taland2/src/train.csv")
    df.shape
    print(df.head())
    """y = df.label
    df.drop("label", axis1 = 1)
    X_train, X_test , Y_train, Y_test = train_test_split(df['text'],y, test_size = 0.33, random_state = 53)

    vectorizer_count = CountVectorizer(stop_words = 'english')
    train_count = vectorizer_count.fit_transform(X_train)

    test_count = vectorizer_count.transform(X_test)
"""
    #print(df.head())

import pandas as pd
import numpy as np
import pickle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.externals import joblib

def detect():
    train_data = pd.read_csv('/home/david/2019-ca400-taland2/src/dataset/train.csv')
    train_data.shape
    print(train_data.head(3))

    test_data = pd.read_csv('/home/david/2019-ca400-taland2/src/dataset/test.csv')

    vectorizer_count = CountVectorizer(stop_words = 'english')
    train_count = vectorizer_count.fit_transform(train_data)
    test_count = vectorizer_count.transform(test_data)

    tfidf_vectorizer = TfidVectorizer(stop_words = 'english', max_df = 0.7)
    tfidf_train = tfidf_vectorizer.fit_transform(train_data)
    tfidf_test = tfidf_vectorizer.transform(test_data)

    print(tfidf_vectorizer.get_feature_name)
    #print(df.head())
def main():
    detect()

if __name__ == "__main__": main()

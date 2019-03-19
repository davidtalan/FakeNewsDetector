import pandas as pd
import numpy as np
import pickle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.externals import joblib

#//:TODO: Combine the two datasets and clean up the headings
#//:TODO: Write script to do a Google Search of target article and compare results using Document Similarity
#//:TODO: Implement document similarity
def detect():
    train_data = pd.read_csv('/home/david/2019-ca400-taland2/src/dataset/train.csv')
    train_data.shape
    print(train_data.head(4))

    test_data = pd.read_csv('/home/david/2019-ca400-taland2/src/dataset/test.csv')

    vectorizer_count = CountVectorizer(stop_words = 'english')
    train_count = vectorizer_count.fit_transform(train_data)
    test_count = vectorizer_count.transform(test_data)

    tfidf_vectorizer = TfidfVectorizer(stop_words = 'english', max_df = 0.7)
    tfidf_train = tfidf_vectorizer.fit_transform(train_data)
    tfidf_test = tfidf_vectorizer.transform(test_data)

    print(tfidf_vectorizer.get_feature_names()[-10:])
    print(vectorizer_count.get_feature_names()[:10])

    count_df = pd.DataFrame(train_count.A, columns = vectorizer_count.get_feature_names())
    tfidf_df = pd.DataFrame(tfidf_train.A, columns = tfidf_vectorizer.get_feature_names())
    difference = set(count_df.columns) - set(tfidf_df.columns)
    print(count_df.equals(tfidf_df))
    print(tfidf_df.head())

def main():
    detect()

if __name__ == "__main__": main()

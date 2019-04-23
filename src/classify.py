import pandas as pd
import numpy as np
import pickle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.externals import joblib
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn import metrics
from collections import Counter
import timeit
import re
import chardet
import codecs

def extract(link):
    article = open(link,"r")
    article = article.read()
    article = article.lower()
    article = re.sub(r'[^a-zA-Z0-9\s]', ' ', article)
    article = [article]
    return article

def detect():

    article = extract("/home/david/2019-ca400-taland2/src/dataset/test.txt")

    """
    with open('/home/david/fake_or_real_news.csv', 'rb') as f:
        while i in f:
            encoding = chardet.detect(f.readline())
            if econding != 'UTF-8'
             ``
    """
    #using the train dataset as a whole dataset for now
    dftrain = pd.read_csv('/home/david/fake_or_real_news.csv', encoding = "ISO-8859-1")

    #drops rows that have null values
    dftrain = dftrain.dropna()

    #Set column names to variables
    df_x = dftrain['text']
    df_y = dftrain['label']

    #split training data
    x_train, x_test, y_train, y_test = train_test_split(df_x, df_y, test_size=0.33, random_state=53)


    #y_train = dftrain['label']
    #y_test = dftest['label']
    #prints first 4

    #Set up CountVectorizer
    cv = CountVectorizer(stop_words = 'english', max_features = 1000)   

    #fit/transform the dataset
    x_traincv = cv.fit_transform(x_train)
    #article_testcv = cv.transform(article)
    x_testcv = cv.transform(x_test)

    #Set up TfidfVectorizer
    tfv = TfidfVectorizer( stop_words = 'english',max_df = 5, max_features =1000)

    #fit/transform the dataset
    x_traintf = tfv.fit_transform(x_train)
    #article_testtf = tfv.transform(article)
    x_testtf = tfv.transform(x_test)

    #prints out the features for tfidf
    #print(tfv.get_feature_names()[-10:])
    #prints out the features for  cv
    #print(cv.get_feature_names()[-10:])

    #turns it into a data structure
    #cv_count_df = pd.DataFrame(x_traincv.A, columns = cv.get_feature_names())

    #tfv_count_df = pd.DataFrame(x_traintf.A, columns = tfv.get_feature_names())

    #prints the first 4 rows of the dataframe
    #print(cv_count_df.head())
    #print(tfv_count_df.head())

    #initialising the clasifier
    mnb_clf = MultinomialNB()
    #svm = SVC(C = 1.0, kernel= 'linear', degree = 3, gamma = 'auto')
    #pac = PassiveAggressiveClassifier(n_iter = 50)
    #fitting in the dataset
    mnb_clf.fit(x_traincv, y_train)
    #pac.fit(x_traincv, y_train)
    #svm.fit(x_traincv, y_train)

    #prediction
    pred = mnb_clf.predict(x_testtf)
    #pred = mnb_clf.predict(article_testtf)

    joblib.dump(mnb_clf, 'mnb_clf_joblib.pkl')
    joblib.dump(tfv, 'tfv_vec.pkl')
    score = metrics.accuracy_score (y_test, pred)
    print(score)

    """
    if pred == [0]:
        print("This news article is reliable")
    else:
        print("This news article is deemed unreliable")

    n = 100
    class_labels = pac.classes_
    feature_names = tfv.get_feature_names()
    topn_class1 = sorted(zip(pac.coef_[0], feature_names)) [:20]
    topn_class2 = sorted(zip(pac.coef_[0], feature_names)) [-20:]

    for coef, feat in topn_class1:
        print(class_labels[0],'real',coef, feat)

    print()

    for coef, feat in reversed (topn_class2):
        print(class_labels[1],'fake',coef, feat)

    """


def main():
    detect()

if __name__ == "__main__":
    start = timeit.default_timer()
    main()
    stop = timeit.default_timer()
    print('Time: ', stop - start)

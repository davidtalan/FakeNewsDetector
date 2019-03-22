import pandas as pd
import numpy as np
import pickle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.externals import joblib
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics

def detect():

    #using the train dataset as a whole dataset for now
    df = pd.read_csv('/home/david/2019-ca400-taland2/src/dataset/train.csv')
    df['text'] = df.text.fillna('None')
    df = df.set_index("id")
    #y = df.label
    df_x = df['text']
    df_y = df['label']
    x_train, x_test, y_train, y_test = train_test_split(df_x, df_y, test_size=0.33, random_state=53)

    print(x_train.head())
    print(x_test.head())
    print(y_train.head())
    print(y_test.head())

    #x_traintf = tfv.fit_transform(x_train)
    cv = CountVectorizer(stop_words = 'english', max_features = 1000)
    x_traincv = cv.fit_transform(x_train)
    x_testcv = cv.transform(x_test)

    tfv = TfidfVectorizer( stop_words = 'english',max_df = 0.7, max_features = 1000)
    x_traintf = tfv.fit_transform(x_train)
    x_testtf = tfv.transform(x_test)

    print(tfv.get_feature_names()[-10:])
    print(cv.get_feature_names()[-10:])

    cv_count_df = pd.DataFrame(x_traincv.A, columns = cv.get_feature_names())

    tfv_count_df = pd.DataFrame(x_traintf.A, columns = tfv.get_feature_names())

    print(cv_count_df.head())
    print(tfv_count_df.head())
    mnb_clf = MultinomialNB()
    mnb_clf.fit(x_traincv, y_train)
    pred = mnb_clf.predict(x_testcv)
    score = metrics.accuracy_score (y_test, pred)
    print(score)

    """
    x_traintf.toarray()

    x_testtf = tfv.fit(x_test)
    mnb = MultinomialNB()
    y_train = y_train.astype('int')
    mnb.fit(x_traintf,y_train)

    pred = mnb.predict(x_testtf)
    print(pred)
    pre = np.array(pred)

    #df.shape
    #df = df.set_index('id')
    #y = df.label
    #df.drop("label", axis = 1)
    #x_train, x_test, y_train, y_test = train_test_split(df['text'], y   , test_size=0.33, random_state=53)
    print(x_train.head())
    print(x_test.head())
    print(y_train.head())
    print(y_test.head())
    #print(y_train.head(8))
    #test_data = pd.read_csv('/home/david/2019-ca400-taland2/src/dataset/test.csv')



    vectorizer_count = CountVectorizer(stop_words = 'english')
    train_count = vectorizer_count.fit_transform(x_train)
    test_count = vectorizer_count.transform(x_test)
    print(vectorizer_count.get_feature_names()[:10])
    tfidf_vectorizer = TfidfVectorizer(stop_words = 'english', max_df = 0.7)
    tfidf_train = tfidf_vectorizer.fit_transform(x_train)
    tfidf_test = tfidf_vectorizer.transform(x_test)
    print(tfidf_vectorizer.get_feature_names()[-10:])

    count_df = pd.DataFrame(train_count.A, columns = vectorizer_count.get_feature_names())
    print(count_df)
    print(" ")
    tfidf_df = pd.DataFrame(tfidf_train.A, columns = tfidf_vectorizer.get_feature_names())


    clf = MultinomialNB()
    clf.fit(tfidf_train, y_train)
    pred = clf.predict(tfidif_test)
    score = metrics.accuracy_score(y_test,pred)
    print(score)
    difference = set(count_df.columns) - set(tfidf_df.columns)
    print(count_df.equals(tfidf_df))
    print(tfidf_df.head())

    linear_clf = PassiveAggressiveClassifier(n_iter=50)
    linear_clf.fit(tfidf_train,train_data)
    pred = linear_clf.predict(tfidf_test)
    score = metrics.accuracy_score(test_data, pred)
    print("accuracy: %0.3f" %score)
"""

def main():
    detect()

if __name__ == "__main__":
    main()

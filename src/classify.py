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
from collections import Counter

def detect():

    #using the train dataset as a whole dataset for now
    df = pd.read_csv('/home/david/2019-ca400-taland2/src/dataset/train.csv')
    #Fill Null values with 'None'
    df['text'] = df.text.fillna('None')
    #Set column names to variables
    df_x = df['text']
    df_y = df['label']
    #split training data
    x_train, x_test, y_train, y_test = train_test_split(df_x, df_y, test_size=0.33, random_state=53)

    #prints first 4
    print(x_train.head())
    print(x_test.head())
    print(y_train.head())
    print(y_test.head())


    article = r'image copyright afp/getty images image caption president trump did not elaborate on which sanctions he was referring to us president donald trump says he has ordered the withdrawal of recently imposed sanctions against north korea. in a tweet on friday, mr trump referred to "additional large-scale sanctions" '

    article = article.replace('"', '')
    #Set up CountVectorizer
    cv = CountVectorizer(stop_words = 'english', max_features = 1000)

    #fit/transforming the dataset
    x_traincv = cv.fit_transform(x_train)
    #article_testcv = cv.transform(article)
    x_testcv = cv.transform(x_test)

    #Set up TfidfVectorizer
    tfv = TfidfVectorizer( stop_words = 'english',max_df = 0.7, max_features = 1000)

    #fit/transform that dataset
    x_traintf = tfv.fit_transform(x_train)
    #article_testtf = tfv.transform(article)
    x_testtf = tfv.transform(x_test)

    #prints out the features for tfidf
    print(tfv.get_feature_names()[-10:])
    #prints out the features for  cv
    print(cv.get_feature_names()[-10:])

    #turns it into a 2D data structure
    cv_count_df = pd.DataFrame(x_traincv.A, columns = cv.get_feature_names())

    tfv_count_df = pd.DataFrame(x_traintf.A, columns = tfv.get_feature_names())

    #prints the first 4 rows of the dataframe
    print(cv_count_df.head())
    print(tfv_count_df.head())

    #initialising the clasifier
    mnb_clf = MultinomialNB()

    #fitting in the dataset
    mnb_clf.fit(x_traincv, y_train)

    #prediction
    pred = mnb_clf.predict(x_testtf)

    score = metrics.accuracy_score (y_test, pred)
    print(score)
    #print(pred)


def main():
    detect()

if __name__ == "__main__":
    main()

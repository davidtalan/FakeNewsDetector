from flask import Flask,render_template,url_for,request
from flask_bootstrap import Bootstrap
from newspaper import Article
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
from googlesearch import search

app = Flask (__name__)
Bootstrap(app)
#//:TODO: Create a database for previously searched/analysed articles and their results.
#//:TODO:

def extractor(url):
    article = Article(url)
    article.download()
    article.parse()
    article_title = article.title
    article = article.text.lower()
    article = re.sub(r'[^a-zA-Z0-9\s]', ' ', article)
    article = [article]
    return (article, article_title)

def google_search(title):
    query = title
    search_dict = {}
    search_result = []
    search_title = []
    for i in search(query, tld = "com", num = 10, start = 1, stop = 5):
        article = Article(i)
        article.download()
        article.parse()
        title = article.title
        search_result[title] = i
        #search_title.append(article.title)
        #search_result.append(i)
        return search_result
    #return (search_result, search_title)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/handle_data', methods = ['POST'])
def handle_data():
    url = (request.form['article_link'])
    article, article_title = extractor(url)
    dftrain = pd.read_csv('/home/david/2019-ca400-taland2/src/dataset/train.csv')
    #drops rows that have null values
    dftrain = dftrain.dropna()

    #Set column names to variables
    df_x = dftrain['text']
    df_y = dftrain['label']

    #split training data
    x_train, x_test, y_train, y_test = train_test_split(df_x, df_y, test_size=0.33, random_state=53)


    #Set up CountVectorizer
    cv = CountVectorizer(stop_words = 'english', max_features = 1000)

    #fit/transform the dataset
    x_traincv = cv.fit_transform(x_train)
    article_testcv = cv.transform(article)
    #x_testcv = cv.transform(x_test)

    #Set up TfidfVectorizer
    tfv = TfidfVectorizer( stop_words = 'english',max_df = 0.7, max_features =1000)

    #fit/transform the dataset
    x_traintf = tfv.fit_transform(x_train)
    article_testtf = tfv.transform(article)
    #x_testtf = tfv.transform(x_test)



    #initialising the clasifier
    mnb_clf = MultinomialNB()
    #svm = SVC(C = 1.0, kernel= 'linear', degree = 3, gamma = 'auto')
    #pac = PassiveAggressiveClassifier(n_iter = 50)
    #fitting in the dataset
    mnb_clf.fit(x_traincv, y_train)
    #pac.fit(x_traincv, y_train)
    #svm.fit(x_traincv, y_train)

    #prediction
    #pred = mnb_clf.predict(x_testtf)
    pred = mnb_clf.predict(article_testtf)

    #joblib.dump(mnb_clf, 'mnb_clf_joblib.pkl')
    #joblib.dump(tfv, 'tfv_vec.pkl')
    #score = metrics.accuracy_score (y_test, pred)
    #print(score)

    #if pred == [0]:
    title = article_title
    return result(pred, title)


@app.route('/result')
def result(prediction, title):
    article_title = title
    #search_list, search_title = google_search(title)
    search_list = google_search(title)
    if prediction == [0]:
        return render_template('/result.html', variable = "This news article is reliable", title = article_title, list = search_list)
    else:
        return render_template('/result.html', variable = "This news article is deemed unreliable", title = article_title, list = search_list)

if __name__ == '__main__':
    app.run(debug = True)

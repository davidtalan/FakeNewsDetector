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
import re
from googlesearch import search
from sklearn.metrics.pairwise import cosine_similarity

app = Flask (__name__)
Bootstrap(app)
#//:TODO: Create a database for previously searched/analysed articles and their results.
#//:TODO:

def extractor(url):
    article = Article(url)
    try:
        article.download()
        article.parse()
    except:
        pass

    article_title = article.title
    article = article.text.lower()
    article = re.sub(r'[^a-zA-Z0-9\s]', ' ', article)
    article = [article]
    return (article, article_title)

def google_search(title, url):
    search_dict = {}
    #search_result = []
    search_title = []
    search_urls = []
    for i in search(title, tld = "com", num = 10, start = 1, stop = 7):
        if "youtube" not in i and i != url:
            search_urls.append(i)
            article = Article(i)
            try:
                article.download()
                article.parse()
            except:
                pass
            title = article.title
            search_title.append(title)
            #search_result.append(i)

    domains = []
    for i in search_urls:
        s = re.findall(r'\s(?:www.)?(\w+.com)', i)
        domains.append(s)

    return search_urls, search_title, domains
    #return (search_result, search_title)

def similarity(url_list, article):
    article = article
    sim_tfv = TfidfVectorizer(stop_words ="english")
    sim_transform1 = sim_tfv.fit_transform(article)
    cosine = []
    for i in url_list:
        test_article, test_title = extractor(i)
        test_article = [test_article]
        sim_transform2 = sim_tfv.transform(test_article[0])
        score = cosine_similarity(sim_transform1, sim_transform2)
        cosine.append(score)
    return cosine

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

    joblib.dump(mnb_clf, 'mnb_clf_joblib.pkl')
    #joblib.dump(tfv, 'tfv_vec.pkl')
    #score = metrics.accuracy_score (y_test, pred)
    #print(score)

    #if pred == [0]:
    title = article_title
    return result(pred, title, article, url)


@app.route('/result')
def result(prediction, title, article, url):
    article_title = title

    url_list, search_titles, domains = google_search(title, url)

    similarity_score = similarity(url_list, article)

    if prediction == [0]:
        return render_template('/result.html', variable = "This news article is reliable", title = article_title, list = url_list, search_t = search_titles,  domains = domains,sim_score = similarity_score)
    else:
        return render_template('/result.html', variable = "This news article is deemed unreliable", title = article_title, list = url_list, search_t = search_titles,  domains = domains, sim_score = similarity_score)

if __name__ == '__main__':
    app.run(debug = True)

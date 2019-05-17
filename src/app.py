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

def extractor(url):
    article = Article(url)
    try:
        article.download()
        article.parse()
    except:
        pass

    article_title = article.title
    article = article.text.lower()
    article = [article]
    return (article, article_title)

def google_search(title, url):
    search_title = []
    search_urls = []
    for i in search(title, tld = "com", num = 10, start = 1, stop = 7):
        if "youtube" not in i and i not in url:
            search_urls.append(i)
            article = Article(i)
            try:
                article.download()
                article.parse()
            except:
                pass
            title = article.title
            search_title.append(title)
    domains = []
    for i in search_urls:
        s = re.findall(r'\s(?:www.)?(\w+.com)', i)
        domains.append(s)
    return search_urls, search_title, domains

def similarity(url_list, article):
    article = article
    sim_tfv = TfidfVectorizer(stop_words ="english")
    sim_transform1 = sim_tfv.fit_transform(article)
    cosine = []
    cosine2 = []
    for i in url_list:
        test_article, test_title = extractor(i)
        test_article = [test_article]
        sim_transform2 = sim_tfv.transform(test_article[0])
        score = cosine_similarity(sim_transform1, sim_transform2)
        cosine.append(score*100)

    for i in cosine:
        x = str(i).replace('[','').replace(']','')
        cosine2.append(x)
    return cosine2

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/handle_data', methods = ['POST'])
def handle_data():
    url = (request.form['article_link'])
    article, article_title = extractor(url)
    job_vec = joblib.load('tfv.pkl')
    job_mnb = joblib.load('mnb.pkl')
    job_cv = joblib.load('cv.pkl')
    pred = job_mnb.predict(job_cv.transform(article))
    #pred = mnb_clf.predict(article_testtf)
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

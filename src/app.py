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
from sklearn.model_selection import train_test_split
import re
from googlesearch import search
from sklearn.metrics.pairwise import cosine_similarity
from urllib.parse import urlparse
import re

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

def textAreaExtractor(text):

    text = text.lower()
    text = re.sub(r'[^a-zA-Z0-9\s]', ' ', text)
    text = re.sub("(\\r|\r|\n)\\n$", " ", text)
    text = [text]
    return text

def google_search(title, url):
    target = url
    domain = urlparse(target).hostname
    search_title = []
    search_urls = []
    source_sites = []
    for i in search(title, tld = "com", num = 10, start = 1, stop = 6):
        if "youtube" not in i and domain not in i:
            source_sites.append(urlparse(i).hostname)
            search_urls.append(i)
            article = Article(i)
            try:
                article.download()
                article.parse()
            except:
                pass
            title = article.title
            search_title.append(title)

    return search_urls, search_title, source_sites

def similarity(url_list, article):
    article = article
    sim_tfv = TfidfVectorizer(stop_words ="english")
    sim_transform1 = sim_tfv.fit_transform(article)
    cosine = []
    cosineCleaned = []
    cosineAverage = 0
    count = 0
    for i in url_list:
        test_article, test_title = extractor(i)
        test_article = [test_article]
        sim_transform2 = sim_tfv.transform(test_article[0])
        score = cosine_similarity(sim_transform1, sim_transform2)
        cosine.append(score*100)
        print("Article " + str(count) + " similarity calculated")
        count+=1
    for i in cosine:
        x = str(i).replace('[','').replace(']','')
        cosineCleaned.append(x)

    for i in cosine:
        if i !=0:
            cosineAverage = cosineAverage + i
        else:
            count-=1

    averageScore = cosineAverage/count
    averageScore = str(averageScore).replace('[','').replace(']','')
    averageScore = float(averageScore)
    print(averageScore)
    return cosineCleaned, averageScore




def handlelink():
    job_vec = joblib.load('tfv.pkl')
    job_mnb = joblib.load('mnb.pkl')
    job_cv = joblib.load('cv.pkl')
    job_pac = joblib.load('pac.pkl')

    url = (request.form['article_link'])
    article, article_title = extractor(url)
    pred = job_pac.predict(job_vec.transform(article))
    print("Target article has been classified")
    #pred = mnb_clf.predict(article_testtf)

    return pred, article_title, article, url


def handletext():
    job_vec = joblib.load('tfv.pkl')
    job_mnb = joblib.load('mnb.pkl')
    job_cv = joblib.load('cv.pkl')

    text = request.form['article_text']
    textarticle = textAreaExtractor(text)
    pred = job_mnb.predict(job_vec.transform(textarticle))

    return pred, textarticle

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/textResult', methods = ['POST'])
def textResult():

    prediction, article = handletext()

    if prediction == [0]:
        return render_template('/textresult.html', variable="This body of article has been classified as reliable",articletext = article)
    else:
        return render_template('/textresult.html', variable="This body of article has been classified as unreliable", articletext = article)

@app.route('/linkResult', methods = ['POST'])
def linkResult():

    prediction, article_title, article, url = handlelink()
    url_list, search_titles, sitename = google_search(article_title, url)
    similarity_score, avgScore = similarity(url_list, article)

    if prediction == [0] and avgScore < 20:
        return render_template('/linkresult.html', variable = "This news article has been classified as reliable but doesn't have many articles to support this statement.", title = article_title, list = url_list, search_t = search_titles,  average = avgScore,sim_score = similarity_score, site = sitename)

    if prediction == [0] and (avgScore > 20 and avgScore < 50) :
        return render_template('/linkresult.html', variable = "This news article has been classified as reliable and is supported by some articles", title = article_title, list = url_list, search_t = search_titles,  average = avgScore,sim_score = similarity_score, site = sitename)

    if prediction == [0] and avgScore > 50 :
        return render_template('/linkresult.html', variable = "This news article has been classified as reliable and is supported by multiple articles", title = article_title, list = url_list, search_t = search_titles,  average = avgScore,sim_score = similarity_score, site = sitename)

    if prediction == [1] and avgScore < 20:
        return render_template('/linkresult.html', variable = "This news article has been classified as unreliable and doesn't have other articles talking about the same thing.", title = article_title, list = url_list, search_t = search_titles,  average = avgScore, sim_score = similarity_score, site = sitename)

    if prediction == [1] and (avgScore > 20 and avgScore < 50):
        return render_template('/linkresult.html', variable = "This news article has been classified as unreliable but may have some articles that talk about the same thing.", title = article_title, list = url_list, search_t = search_titles,  average = avgScore, sim_score = similarity_score, site = sitename)

    if prediction == [1] and avgScore > 50:
        return render_template('/linkresult.html', variable = "This news article has been classified as unreliable but have multiple articles that say the same thing.", title = article_title, list = url_list, search_t = search_titles,  average = avgScore,sim_score = similarity_score, site = sitename)

if __name__ == '__main__':
    app.run(debug = True)

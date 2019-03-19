from flask import Flask,render_template,url_for,request
from flask_bootstrap import Bootstrap
from newspaper import Article

app = Flask (__name__)
Bootstrap(app)
#//:TODO: Display extracted article
#//:TODO: Create a database for previously searched/analysed articles and their results.
#//:TODO:
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/handle_data', methods = ['POST'])
def handle_data():
    url = (request.form['article_link'])
    article = Article(url)
    article.download()
    article.parse()
    return render_template('/result.html', variable = article.text.lower())

@app.route('/result')
def result():
    return render_template('result.html')

if __name__ == '__main__':
    app.run(debug = True)

from flask import Flask,render_template,url_for,request
from flask_bootstrap import Bootstrap

app = Flask (__name__)
Bootstrap(app)

@app.route('/')
def home():
    return render_template('home.html')


@app.route('/result')
def result():
    return render_template('result.html')

if __name__ == '__main__':
    app.run(debug = True)

from newspaper import Article
url = 'https://www.rte.ie/news/business/2019/0305/1034553-brexit-construction/'
article = Article(url)
article.download()
article.parse()
print(article.text)

from boilerpipe.extract import Extractor

extractor = Extractor(extractor='ArticleExtractor', url="https://www.bbc.com/news/world-africa-47089304")
print (extractor.getText())

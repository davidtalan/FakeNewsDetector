# Blog: Fake News Detector
**David Talan**

## Blog Entry 1 - Beginning of the end
Project idea exploration begins.
Initial ideas:
- Change Management application (similar to the one I used on INTRA)
- Predicting the virality of a tweet. Tweets nowadays go viral pretty quickly and advertisers might flock to the idea of knowing what tweet has the potential to be viewed by millions
- Fake News Detection using link aggregator sites like [Reddit](http://www.reddit.com/) (Donald Trump - enough said)
![donald](https://gitlab.computing.dcu.ie/taland2/2019-ca400-taland2/tree/master/docs/blog/images/donald.jpg)


##Blog Entry 2
Met up with Suzanne Little to pitch her some of my ideas. I noticed she supervised a good bit of Natural Language Processing in last year's 4th year projects, which is the area of a couple of my ideas.

The Twitter virality idea might be too difficult as it can be very subjective.

Leaning towards the idea of a Fake News Detection chrome extension app. The idea is to use comments from link aggregator sites like Reddit or the comment section of a social media network like Facebook, to tell me if an article is fake. Too many times I've read an article heading and didn't click into it and took the title as fact. Sometimes while using Reddit, the comment section disproves the article or calls it misleading, thus giving me the idea of using comments to classify the article.

Suzanne liked the idea as it is a very relevant topic. Big problem is looking for the dataset as comments from a site specifically talking about fake news is very hard to come by, unless I collect it myself.
She redirected me to John McKenna or Yvette Graham whose more specialised in Natural Language Processing.

##Blog Entry 3
Met up with Yvette Graham and discussed the project. She also liked the idea of the Fake News Detector, less so about the Tweet virality one. She had the same worries like Suzanna about the dataset as it is a very specific one. Looks like I might have to switch things up.

She also agreed to be my supervisor so that's one less thing to worry about.

First draft of the proposal to be done by the following Monday.

##Blog Entry 4
Been doing a bit more research about the tools I might need. NLTK for Python seems like a good shout for the NLP aspect of the project. Decided to use Python for the project and maybe Django for the framework. scikit-learn seems to be a very popular machine learning tool; will look into it more.

##Blog Entry 5 -
Proposal form done and dusted.
Meeting with aprroval panel with Paul Clarke and Gareth Jones.
Many questions were asked, "Is it possible?", "what dataset?".
Gareth suggested using a different method, maybe using comparing multiple articles to see how how similar it is - might contribute to proving reliability.

APPROVED though.

##Blog Entry 6 - Spam! Not ham!
Completely ditched the idea of using comments classify fake or real articles. Can't find any type of dataset similar to it.
A bit more research showed people using text classification methods, similar to detecting spam e-mails.
Read some papers/blogs on how some groups did it - TFIDF seems to come up a lot.

##Blog Entry 7
Functional Specification done and submitted.
Gonna take Gareth's advice and include a document similarity aspect to the project - comparing articles extracted from a Google search.

Gonna focus on last few weeks of semester 1 for now - ~~I love Compiler!~~

##Blog Entry 8
Project start after exams. Need to:
- find a way to extract articles
- find a dataset
- what tools to use

##Blog Entry 9
I found an article extractor but it's really giving me a headache. Think this is my 3rd one now I tried to use? Also found a dataset finally - currently trying out Pandas to see how it handles it.

##Blog Entry 10 - Django ~~Un~~ Chained
Decided to use Flask instead. I have a general idea on what the site will look like and I don't think I would need much of what Django offers. A simple web application with a machine learning model on it sounds perfect for a simple, lightweight framework like Flask - we have a winner!

##Blog Entry 11
Started learning how to use sk-learn. Data preprocessing is a bit confusing and getting a lot of errors - need to figure out how to clean up this dataset a bit.

##Blog Entry 12
Finally found a working extractor (Newspaper3k)! And it's so easy use! Still trying to figure out how to work the Machine Learning/Text classification stuff, but I think I'm making some headway.

##Blog Entry 13 - It's alive!
Finally working text classification - gonna do a bunch of testing random text for now, see how fast it is.
Found an API for the the Google search of the article title - basically the idea is, using the Newspaper3k to grab the target article's title, I'll perform a Google search and pull the top few results. Then use a document similarity algorithm to compare them to the original article.

##Blog Entry 14
Text classification is working well.
Making some good progress with scraping the Google search articles. Trying out cosine similarity to compare individual articles for now.
Also created table to display the similar articles with their similarity score - might do an average score, I'll see.

##Blog Entry 15
Similarity score is working properly now.
Some more UI changes to be made - thinking of adding a textbox to classify a block of text to add a bit more functionality.

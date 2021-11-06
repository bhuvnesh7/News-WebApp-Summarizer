# Python script for retrieving news articles from NewsAPI, summarizing them
# using self-implemented summarization function based on TextRank and 
# storing in MongoDB database.

import requests
import json
import time
from newspaper import Article
# from gensim.summarization.summarizer import summarize
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import sent_tokenize,word_tokenize
import networkx as nx

import os
from dotenv import load_dotenv
import pymongo

mongo_link = " " # Your mongo cluster link goes here
client = pymongo.MongoClient(mongo_link)
db = client['news-web']
news_art = db['news-2']


nltk.download('wordnet')
nltk.download('punkt')
nltk.download('stopwords')

vec_path = 'glove/glove.6B.100d.txt' # Glove embeddings file
embeddings_file = open(vec_path, 'r', encoding="utf8")
print('CP1')
embeddings = dict()

for line in embeddings_file:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float64')
    embeddings[word] = coefs

embeddings_file.close()
print('CP2')

def clean(sentence):
    lem = WordNetLemmatizer()
    sentence = sentence.lower()
    sentence = re.sub(r'http\S+',' ',sentence)
    sentence = re.sub(r'[^a-zA-Z]',' ',sentence)
    sentence = sentence.split()
    sentence = [lem.lemmatize(word) for word in sentence if word not in stopwords.words('english')]
    sentence = ' '.join(sentence)
    return sentence

def average_vector(sentence):
    words = sentence.split()
    size = len(words)
    average_vector = np.zeros((size,100))
    unknown_words=[]

    for index, word in enumerate(words):
        try:
            average_vector[index] = embeddings[word].reshape(1,-1)
        except Exception as e:
            unknown_words.append(word)
            average_vector[index] = 0

    if size != 0:
        average_vector = sum(average_vector)/size
    return average_vector,unknown_words

def cosine_similarity(s1, s2):
    v1, _ = average_vector(s1)
    v2, _ = average_vector(s2)
    cos_sim = 0
    try:
        cos_sim = (np.dot(v1,v2)/(np.linalg.norm(v1)*np.linalg.norm(v2)))
    except Exception as e :
        pass
    return cos_sim

def page_rank(M, d = 0.85, iters = 100): #self-implemented pagerank function
  N = M.shape[1]
  ranks = np.full((N,1), 1/N)
  M_hat = (d * M + (1 - d) / N)

  for i in range(iters):
    ranks = M_hat @ ranks
  
  ranks = ranks.flatten()
  return ranks

def textrank_summarise(paragraph, no_of_sentences): # self-implemented summarization function based on TextRank
    
    sentences = sent_tokenize(paragraph) # no. of sentences
    cleaned_sentences=[]
    for sentence in sentences:
        cleaned_sentences.append(clean(sentence))
    similarity_matrix = np.zeros((len(cleaned_sentences),len(cleaned_sentences)))

    for i in range(0,len(cleaned_sentences)):
        for j in range(0,len(cleaned_sentences)):
            if type(cosine_similarity(cleaned_sentences[i],cleaned_sentences[j])) == np.float64 :
                similarity_matrix[i,j] = cosine_similarity(cleaned_sentences[i],cleaned_sentences[j])
    
    # nx_graph = nx.from_numpy_array(similarity_matrix)
    flag = 0
    try:
        # scores = nx.pagerank(nx_graph, max_iter=600)  # VVIMP
        scores = page_rank(similarity_matrix)
    except Exception as e :
        flag=1
        return 'xxx'

    ranked_sentences = sorted(((scores[i],s) for i,s in enumerate(sentences)), reverse=True)

    # summary = ""
    
    # for i in range(no_of_sentences):
    #   summary=summary + ranked_sentences[i][1]
    
    # summary = summary.strip()
    # summary = re.sub(r'\n',' ',summary)

    templis = [y[1] for y in ranked_sentences[:no_of_sentences]]
    
    summary = '\n'.join(templis)
    summary = summary.strip()
    summary = re.sub(r'\n\n','\n',summary)
    
    if flag==0:
        return summary


# Getting articles using the API

api_key = '' # NewsAPI key goes here

biglis = [] #To insert in collection

# Indian news

country = 'in'
cats = ['general','business', 'entertainment', 'sports', 'health', 'science', 'technology']  # international too

cat_dict = {
    'general' : '5dea087938071a083c26abbf' ,
    'business' : '5dea084438071a083c26abb9' ,
    'entertainment' : '5dea086f38071a083c26abbd',
    'sports' : '5dea087538071a083c26abbe' ,
    'health' : '5dea085138071a083c26abba',
    'science' : '5dea086038071a083c26abbb' ,
    'technology' : '5dea086938071a083c26abbc' ,
    'international' : '60695ed2a81093c540951c9f'
}

for cat in cats:
    api_call = ('http://newsapi.org/v2/top-headlines?'
       'country='+country+'&'
       'category='+cat+'&'\
       'apiKey='+api_key)
    response = requests.get(api_call)
    data = response.json() 
    for arti in data['articles']:
        url = arti['url']
        article = Article(url)
        try:
            article.download()
            article.parse()
            title = article.title
            text = article.text
            nos = len(sent_tokenize(text))
            summ = textrank_summarise(text, nos//3) # 33.33% of original text length
            if summ!= '' and summ!='xxx':
                content = {
                    'category_id' : cat_dict[cat],
                    'url' : url,
                    'title' : title,
                    'image' : article.top_image,
                    'content' : summ,
                    'date' : round(time.time()*1000),
                    'views' : 0,
                    'num' : 0,
                    '__v' : 0
                }
                biglis.append(content)
                # print('here')
        except Exception as e :
            # print('there')
            pass


# International news

api_call = ('http://newsapi.org/v2/everything?'
       'domains=apnews.com'+'&'
       'sortBy=popularity' + '&'
       'pageSize=20'+'&'\
       'apiKey='+api_key)
response = requests.get(api_call)
data = response.json() 
for arti in data['articles']:
    cat = 'international'
    url = arti['url']
    article = Article(url)
    try:
        article.download()
        article.parse()
        title = article.title
        text = article.text
        nos = len(sent_tokenize(text))
        summ = textrank_summarise(text, nos//3) # 33.33% of original text length
        if summ!= '' and summ!='xxx':
            content = {
                    'category_id' : cat_dict[cat],
                    'url' : url,
                    'title' : title,
                    'image' : article.top_image,
                    'content' : summ,
                    'date' : round(time.time()*1000),
                    'num' : 0,
                    'views' : 0,
                    'num' : 0,
                    '__v' : 0
                }
            biglis.append(content)
            # print('here')
    except Exception as e :
        # print('there')
        pass

print('CP3')
news_art.insert_many(biglis) #Insert in collection
print('CP4')
#Do not add entry in database if summary is 'xxx' or '' (empty string)

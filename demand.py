# Flask API for on-demand summarization

import requests
import json
import time
from newspaper import Article
from gensim.summarization.summarizer import summarize
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import sent_tokenize,word_tokenize
import networkx as nx


import flask
from flask import request, jsonify

nltk.download('wordnet')
nltk.download('punkt')
nltk.download('stopwords')

vec_path = 'glove/glove.6B.100d.txt' # Glove embeddings file, store in the given path
embeddings_file = open(vec_path, 'r', encoding="utf8")
embeddings = dict()

app = flask.Flask(__name__)
app.config["DEBUG"] = True


for line in embeddings_file:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float64')
    embeddings[word] = coefs

embeddings_file.close()

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
    
    nx_graph = nx.from_numpy_array(similarity_matrix)
    flag = 0
    try:
        scores = nx.pagerank(nx_graph, max_iter=600)  # VVIMP
    except Exception as e :
        flag=1
        return 'xxx'
    

    ranked_sentences = sorted(((scores[i],s) for i,s in enumerate(sentences)), reverse=True)

    # summary = ""
    
    # for i in range(no_of_sentences):
    #   summary=summary + ' ' + ranked_sentences[i][1]
    
    # summary = summary.strip()
    # summary = re.sub(r'\n',' ',summary)

    templis = [y[1] for y in ranked_sentences[:no_of_sentences]]
    
    summary = '\n'.join(templis)
    summary = summary.strip()
    summary = re.sub(r'\n\n','\n',summary)
    
    if flag==0:
        return summary



@app.route('/', methods=['GET'])
def home():
    return '''<h1>News Summary</h1>'''


@app.route('/trank/<path:url>', methods=['GET'])
def trank(url):
    article = Article(url)
    try:
        article.download()
        article.parse()
        title = article.title
        image = article.top_image
        text = article.text
        nos = len(sent_tokenize(text))
        summ = textrank_summarise(text, nos//3) # 33.33% of original text length
        if summ!= '' and summ!='xxx':
            ret = {
                'link' : url,
                'title' : title,
                'image' : image,
                'summary' : summ
            }

            return jsonify(ret)
        else:
            ret = {'fail' : 'Unsuccessful'}
            return jsonify(ret)
    except Exception as e:
        ret = {'fail' : 'Invalid URL'}
        return jsonify(ret)


@app.route('/gsim/<path:url>', methods=['GET'])
def gsim(url):
    article = Article(url)
    try:
        article.download()
        article.parse()
        title = article.title
        image = article.top_image
        text = sent_tokenize(article.text)
        # del text[0:4]
        ftext = ' '.join(text)
        try:
            summ = summarize(ftext, ratio=0.333) # Gensim's summarization function
            if summ!= '' and summ!='xxx':
                # summ = re.sub(r'\n',' ',summ)
                ret = {
                    'link' : url,
                    'title' : title,
                    'image' : image,
                    'summary' : summ
                }

                return jsonify(ret)
                
            else:
                ret = {'fail' : 'Unsuccessful'}
                return jsonify(ret)
        except Exception as e:
            ret = {'fail' : 'Unsuccessful'}
            return jsonify(ret)
    except Exception as e:
        ret = {'fail' : 'Invalid URL'}
        return jsonify(ret)


temp = 'https://apnews.com/article/apple-inc-lifestyle-trials-technology-business-c8acede2ad74d0b996e1b398351d52a3'

app.run(port = 9000)
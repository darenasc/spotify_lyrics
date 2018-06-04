from __future__ import division
import os
import sys
import json
import spotipy
import webbrowser
import spotipy.util as util
from json.decoder import JSONDecoder
import pprint
import requests

import pandas as pd
import urllib2
from bs4 import BeautifulSoup
import re

import nltk
from nltk.tokenize import RegexpTokenizer
from nltk.probability import FreqDist
from nltk.corpus import stopwords
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from wordcloud import WordCloud
import matplotlib.pyplot as plt 

def plot_freq_words(fdist, top_n = 20):
    df = pd.DataFrame(columns=('word', 'freq'))
    i = 0
    for word, frequency in fdist.most_common(21):
        df.loc[i] = (word, frequency)
        i += 1

    title = 'Top %s words in lyrics' % top_n
    df.plot.barh(x='word', y='freq', title=title, figsize=(5,5)).invert_yaxis()
    
    return

def plot_word_cloud(corpus, max_words = 20, width=600, height=400, fig_size=(8,6)):
    try:
        if len(corpus) == 0:
            corpus = 'no words'
        wordcloud = WordCloud(max_words = max_words, width=width, height=height, background_color="black").generate(corpus)
        plt.figure(figsize=fig_size, dpi=80)
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis("off")
        plt.show()
        return
    except:
        pass
    return


def plot_sentiment_analysis(lyrics):
    sid = SentimentIntensityAnalyzer()

    df = pd.DataFrame(columns=('song', 'pos', 'neg', 'neu'))
    df2 = pd.DataFrame(columns=('song', 'pos', 'neg', 'neu'))
    i = 0

    pos = 0
    neg = 0
    neu = 0
    for line in lyrics.splitlines():
        if len(line) > 3:
            ss = sid.polarity_scores(line)

            if ss['compound'] >= 0.05:
                pos+=1
            elif ss['compound'] <= -0.05:
                neg+=1
            else:
                neu+=1

            df.loc[i] = (line, ss['pos'], ss['neg'], ss['neu'])
            if (pos+neg+neu) > 0:
                df2.loc[i] = (line, pos / (pos+neg+neu), neg / (pos+neg+neu), neu / (pos+neg+neu))
            i += 1

    #df.plot.barh(x='song', stacked=True, figsize=(10,15), color=['b','r','y']).invert_yaxis();

    labels = 'Positive', 'Negative', 'Neutral'
    sizes = [pos, neg, neu]
    explode = (0.1, 0, 0)
    colors = ['b','r','y']

    fig1, ax1 = plt.subplots()
    ax1.pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%',
            colors=colors,
            shadow=True, startangle=90)
    ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

    plt.show()
    return

def get_link_from_search(search_term=None):
    def get_song_name():
        artist = raw_input('Artist: ').replace(' ', '+');
        song = raw_input('Song: ').replace(' ', '+');
        return '{}:{}'.format(artist, song)
    
    url_search = 'http://lyrics.wikia.com/wiki/Special:Search?query='
    if search_term is None:
        search_term = get_song_name()
    else:
        search_term = search_term
    site = urllib2.urlopen(url_search + search_term).read()
    soup = BeautifulSoup(site, 'lxml')
    return soup.find("a", class_='result-link').get('href')

def get_lyrics(search_term=None):
    if search_term is None:
        song_url = get_link_from_search()
    else:
        song_url = get_link_from_search(search_term)
    
    try:
        site = urllib2.urlopen(song_url).read()
        soup = BeautifulSoup(site, 'lxml')
        lyric = soup.find_all("div", class_="lyricbox")

        if len(lyric) > 0:
            for element in lyric:
                return re.sub("([a-z])([A-Z])","\g<1> \g<2>", BeautifulSoup(str(lyric[0]).replace('<br/>','\n')).get_text())
    except:
        pass

def get_current_song(token):
    url = "https://api.spotify.com/v1/me/player/currently-playing" 
    headers = {"Accept": "application/json",
               "Content-Type": "application/json",
               "Authorization": "Bearer {}".format(token)}
    resp = requests.get(url, headers=headers).json()
    print 'Band: {}'.format(resp['item']['album']['artists'][0]['name'])
    print 'Song: {}'.format(resp['item']['name'].split(' - ')[0])
    return '{}:{}'.format(resp['item']['album']['artists'][0]['name'].replace(' ', '+'), resp['item']['name'].split(' - ')[0].replace(' ', '+'))

def plot_lyrics(lyrics):
    raw = ''
    for word in lyrics.split():
        raw += word + ' '

    tokenizer = RegexpTokenizer(r'\w+')
    tokens = tokenizer.tokenize(raw.lower())   # tokens without punctuation
    text = nltk.Text(tokens)

    words = [w.lower() for w in tokens]

    filtered_words = [word for word in words if word not in stopwords.words('english') and len(word) > 1 and word not in ['na','la']] # remove the stopwords
    fdist = nltk.FreqDist(filtered_words)

    top_n = 20
    text.dispersion_plot([str(w) for w, f in fdist.most_common(top_n)])
    #plot_freq_words(fdist)
    plot_word_cloud(raw.lower(), max_words=50, fig_size=(10,8))

    plot_sentiment_analysis(lyrics)
    return

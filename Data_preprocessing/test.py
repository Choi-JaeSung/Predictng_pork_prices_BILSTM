# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import os

os.chdir("Predicting_pork_retail_price/Data/Data_etc/PigTimes_news")

words = pd.read_csv("PigTimes_news_words_freq.csv", sep=',')
words = np.array(words)
words = words[:,0]

news_nouns = pd.read_csv("PigTimes_news_nouns_freq.csv", sep=',')
news_nouns = np.array(news_nouns)

stopwords = []

for noun in  news_nouns:
        if noun[0] not in words:
                stopwords.append(noun)

df = pd.DataFrame(stopwords)
df.to_csv("PigTimes_news_stopwords.csv", index=False, header=['Stopword', 'Freq'])
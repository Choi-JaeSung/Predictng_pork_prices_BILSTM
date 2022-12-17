# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import os
from collections import Counter

os.chdir("Predicting_pork_retail_price/Data/Data_etc/PigTimes_news")

news = pd.read_csv("PigTimes_news_nouns_data.csv", sep=',')
news = np.array(news)

stopwords = pd.read_csv("PigTimes_news_stopwords.csv", sep=',')
stopwords = np.array(stopwords)
stopwords = stopwords[:, 0] # 온라인 뉴스의 stopword

news_filterd = []

# 각 뉴스 nouns 중 stop_word에 없는 nouns만 추출
for article in news:
    nouns_filterd = []
    nouns = eval(article[1])
    for noun in nouns:
        if noun not in stopwords:
            nouns_filterd.append(noun)

    article_filterd = [article[0], nouns_filterd]
    news_filterd.append(article_filterd)

df = pd.DataFrame(news_filterd)
df.to_csv("Pigtimes_news_nouns_filterd_data.csv", index=False) # len(news_prep) = 8343

news_nouns_freq = []

# 각 뉴스 nouns_freq 추출
for article in news_filterd:
    nouns = article[1]
    nouns_freq = Counter(nouns)
    news_nouns_freq.append([article[0], nouns_freq.most_common(len(nouns_freq))])

df = pd.DataFrame(news_nouns_freq)
df.to_csv("Pigtimes_news_nouns_filterd_freq_data.csv", index=False) # len(news_nouns_freq) = 8343
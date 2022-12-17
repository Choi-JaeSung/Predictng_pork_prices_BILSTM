import numpy as np
import pandas as pd
import os
from collections import Counter

os.chdir("Predicting_pork_retail_price/Data")

news = pd.read_csv("Data_etc/News_nouns_cnt_prep_data.csv", sep=',')
news = np.array(news)
news = news.tolist()

# BOW
news_freq = []

# 각 뉴스마다 BOW 추출
for article in news:
    nouns = eval(article[1])
    c = Counter(nouns)
    nouns_freq = c.most_common(len(c))
    news_freq.append([article[0], nouns_freq])

df = pd.DataFrame(news_freq)
df.to_csv("News_freq_data.csv", index=False) # len(news_freq) = 2756
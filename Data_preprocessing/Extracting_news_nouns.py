import numpy as np
import pandas as pd
import os
from konlpy.tag import Okt

os.chdir("Predicting_pork_retail_price/Data/Data_etc")

news = pd.read_csv("PigTimes_news/PigTimes_news.csv", sep=',')
news = np.array(news)

okt = Okt()

# 각 뉴스 nouns
news_nouns = []

# 각 뉴스 nouns 추출
for article in news:
    news_nouns.append([article[0], okt.nouns(article[1])])

df = pd.DataFrame(news_nouns)
df.to_csv("PigTimes_news_nouns_data.csv", index=False, header=['Date', 'Nouns']) # len(news_nouns) = 8343

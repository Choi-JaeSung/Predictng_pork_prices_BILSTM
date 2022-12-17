import numpy as np
import pandas as pd
import os
from collections import Counter

os.chdir("Predicting_pork_retail_price/Data/Data_etc/PigTimes_news")

news = pd.read_csv("PigTimes_news_nouns_data.csv", sep=',')
news = np.array(news)

# 모든 뉴스의 nouns
text_all = []

# 모든 뉴스의 nouns 추출
for article in news:
    for noun in eval(article[1]):
        text_all.append(noun)
        
# 모든 뉴스의 BOW 추출
c = Counter(text_all)
nouns_cnt = c.most_common(len(c))

df = pd.DataFrame(nouns_cnt)
df.to_csv("PigTimes_news_nouns_freq_data.csv", index=False, header=['Noun', 'Freq']) # len(news_nouns) = 20473
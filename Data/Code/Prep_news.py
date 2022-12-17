import os
import numpy as np
from numpy.core.fromnumeric import shape
import pandas as pd

# data_path = "C:/Users/skyle/SW_App/Predicting_pork_retail_price/Data"
os.chdir("Predicting_pork_retail_price/Data")


news = pd.read_csv("Data_etc/KBS_news.csv", sep=',')
news = np.array(news)

temp_prices = pd.read_csv("Temp_Price_data.csv", sep=',')
temp_prices = np.array(temp_prices)

#temp_prices 데이터 날짜와 같은 news
news_sorted = np.empty(shape=(0,2), dtype=np.object_)

delete_index = []

for i in  range(0, len(news)):
    if news[i][1] is np.nan:
        i -= 1 * len(delete_index)
        delete_index.append(i)

for i in delete_index:
    news = np.delete(news, i, axis=0)
    
for temp_price in temp_prices:
    temp_price_date = temp_price[0].split('-')
    
    for article in news:
        article_date = article[0].split('.')

        # 같은 날짜 뉴스 추출
        if article_date[0] == temp_price_date[0] and \
            article_date[1] == temp_price_date[1] and \
            article_date[2] == temp_price_date[2]:
                
                news_sorted = np.insert(news_sorted, len(news_sorted), [article], axis=0)

df = pd.DataFrame(news_sorted)
df.to_csv("News_data.csv", index=False) # len(news_sorted) = 2755
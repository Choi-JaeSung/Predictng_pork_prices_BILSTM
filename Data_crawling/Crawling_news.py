from bs4 import BeautifulSoup as bs
from selenium import webdriver as wd
from selenium.webdriver.common.keys import Keys
import requests
import numpy as np
import pandas as pd
from time import sleep

path = "C:/Users/skyle/chromedriver/chromedriver"
driver = wd.Chrome(path)

url = "https://news.kbs.co.kr/search/search.do?query=%EB%8F%BC%EC%A7%80%EA%B3%A0%EA%B8%B0#1"
driver.get(url)

articles = [] # 기사들
flag = False

print("크롤링 시작")

while True:
    if len(articles) == 6400:
        break
    else:
        
        next_btn = driver.find_element_by_xpath("//*[@id='content']/div/div[1]/div[5]/a[2]/span")
        
        page = driver.page_source
        soup = bs(page, "html.parser")
        
        links = soup.find_all("li", {"style" : "display: list-item;"})
        articles_hrefs = [link.find("a")["href"] for link in links]
        
        for article_href in articles_hrefs:
            page_article = requests.get(url=article_href)
            soup_article = bs(page_article.text, "html.parser")
            
            article_date = soup_article.find("em", {"class" : "date"})
            if article_date is None:
                article_date = 0
            else:
                article_date = str(article_date).split(sep=" ")[2]
            
            article = soup_article.find("div", {"class" : "detail-body font-size"})
            if article is None:
                article = "None"
            else:
                article = article.text.strip()
            
            articles.append([article_date, article])
            print(len(articles))
        
        flag = True

        if soup.find("a", {"class": "next disabled"}) is not None:
            continue
        else:
            if flag == True:
                flag = False
                
                driver.execute_script("arguments[0].click();", next_btn)
                driver.get(driver.current_url)
            
print("크롤링 끝~")

driver.close()

articles = np.array(articles)
articles = pd.DataFrame(articles)
articles.to_csv('KBS_news.csv', index=False)
# Pigtimes(양돈타임스)사이트 온라인 뉴스 데이터 크롤링

from bs4 import BeautifulSoup as bs
from selenium import webdriver as wd
import requests
import numpy as np
import pandas as pd

path = "C:/Users/skyle/chromedriver/chromedriver"
driver = wd.Chrome(path)

url = "http://www.pigtimes.co.kr/news/articleList.html?page=1&total=18409&sc_section_code=&sc_sub_section_code=&sc_serial_code=&sc_area=A&sc_level=&sc_article_type=&sc_view_level=&sc_sdate=&sc_edate=&sc_serial_number=&sc_word=%EB%8F%BC%EC%A7%80&sc_word2=&sc_andor=&sc_order_by=E&view_type=sm"
driver.get(url)

search_url = "http://www.pigtimes.co.kr/news/"
articles_url = "http://www.pigtimes.co.kr"

articles = [] # 기사들
flag = False # 현재 페이지 온라인 뉴스 크롤링 완료 여부

print("----------크롤링 시작----------")

while True:
    if len(articles) >= 10565:
        break
    else:        
        current_page = driver.page_source
        soup = bs(current_page, "html.parser")
        
        next_url = soup.find("li", {"class" : "pagination-onenext"})
        next_url = next_url.find("a")["href"] # 다음 페이지 링크
        
        links = soup.find_all("div", {"class" : "list-titles"})
        articles_hrefs = [link.find("a")["href"] for link in links] # 현재 페이지 온라인 뉴스 페이지 링크들
        
        for article_href in articles_hrefs:
            article_url = articles_url + article_href
            
            page_article = requests.get(url=article_url) # 온라인 뉴스 페이지
            soup_article = bs(page_article.text, "html.parser")
            
            article_date = soup_article.find("span", {"class" : "updated"}).text # 온라인 뉴스 발행일자
            if article_date is None:
                article_date = '0'
            else:
                article_date = article_date.split(' ')[0]
            
            article_content = soup_article.find("div", {"id" : "article-view-content-div"}).text # 온라인 뉴스 내용
            if article_content is None:
                article_content = "None"
            else:
                article_content = ' '.join(article_content.split()) # 내용간 긴 공백 제거
            
            articles.append([article_date, article_content])
            print(len(articles))
        
        flag = True

        if flag == True:
            flag = False
            
            url = search_url + next_url
            driver.get(url) # 다음 페이지 이동
            
print("----------크롤링 완료----------")

driver.close()

articles = np.array(articles)
articles = pd.DataFrame(articles)
articles.to_csv('PigTimes_news.csv', index=False) # len(articles) = 8343
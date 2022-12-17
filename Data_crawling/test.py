from bs4 import BeautifulSoup as bs
from selenium import webdriver as wd
import requests
import numpy as np
import pandas as pd
import re

path = "C:/Users/skyle/chromedriver/chromedriver"
driver = wd.Chrome(path)

# url = "http://www.pigtimes.co.kr/news/articleList.html?page=1&total=18409&sc_section_code=&sc_sub_section_code=&sc_serial_code=&sc_area=A&sc_level=&sc_article_type=&sc_view_level=&sc_sdate=&sc_edate=&sc_serial_number=&sc_word=%EB%8F%BC%EC%A7%80&sc_word2=&sc_andor=&sc_order_by=E&view_type=sm"
url = "http://www.pigtimes.co.kr/news/articleView.html?idxno=44772"
driver.get(url)

search_url = "http://www.pigtimes.co.kr/news/"
articles_url = "http://www.pigtimes.co.kr"

articles = [] # 기사들
flag = False

print("크롤링 시작")

page = driver.page_source
soup = bs(page, "html.parser")

# next_url = soup.find("li", {"class" : "pagination-onenext"})
# next_url = next_url.find("a")["href"]
# print(next_url)

# url = search_url + next_url
# print(url)
# driver.get(url)

# links = soup.find_all("div", {"class" : "list-titles"})
# articles_hrefs = [link.find("a")["href"] for link in links]

# article_url = articles_url + articles_hrefs[0]

# page_article = requests.get(url=article_url)
# soup_article = bs(page_article.text, "html.parser")

# article_date = soup_article.find("span", {"class" : "updated"}).text
# article_date = article_date.split(' ')[0]
# print(article_date[0])

# article_content = soup_article.find("div", {"id" : "article-view-content-div"})
# article_content = str(article_content.find_all("p"))
# article_content = re.sub('<.+?>', '', article_content).strip()
article_content = soup.find("div", {"id" : "article-view-content-div"}).text
# article_content = str(article_content.find_all("br"))
article_content = ' '.join(article_content.split())
print(article_content)
from bs4 import BeautifulSoup as bs
from selenium import webdriver as wd
from selenium.webdriver.support.ui import Select
import requests
import numpy as np
import pandas as pd

path = "C:/Users/skyle/chromedriver/chromedriver"
driver = wd.Chrome(path)

url = "https://www.kamis.or.kr/customer/inform/news/news.do"
driver.get(url)

# 제목+내용으로 돼지 검색
search = driver.find_element_by_name("search_keyword")
option = Select(driver.find_element_by_name("search_option"))
search.clear()
option.select_by_visible_text(text="제목+내용")
search.send_keys("돼지")
search.submit()

page = driver.page_source
soup = bs(page, "html.parser")

front_url = "https://www.kamis.or.kr"
contents = [] # 기사들

# 기사 href 가져오기
links = soup.find_all("td", {"class" : "tal"})
print(links)
hrefs = [link.find("a")["href"] for link in links]

# article_url = front_url + hrefs[0]
# page_article = requests.get(url= article_url)
# soup_article = bs(page_article.text, "html.parser")
# content = soup_article.find("td", {"cl" : "row"})
# contents.append(content)

# print(contents)

driver.close()
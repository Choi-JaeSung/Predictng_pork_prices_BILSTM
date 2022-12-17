from bs4 import BeautifulSoup as bs
from selenium import webdriver as wd
from selenium.webdriver.support.ui import Select
import requests
import numpy as np
import pandas as pd

# path = "C:/Users/skyle/chromedriver/chromedriver"
# driver = wd.Chrome(path)

# url = "https://news.kbs.co.kr/news/view.do?ncd=5077803"
# driver.get(url)

# articles = [] # 기사들

# page = driver.page_source
# soup = bs(page, "html.parser")

# date = soup.find("em", {"class" : "date"})
# print(str(date).split(sep=" ")[2])

# article = soup.find("div", {"class" : "detail-body font-size"})
# print(article.text.strip())

path = "C:/Users/skyle/chromedriver/chromedriver"
driver = wd.Chrome(path)

url = "https://news.kbs.co.kr/search/search.do?query=%EB%8F%BC%EC%A7%80#1"
driver.get(url)

# latest_btn = driver.find_element_by_css_selector("#latest > button")
# latest_btn.click()

soup = bs(driver.page_source, "html.parser")
# links = soup.find_all("li", {"style" : "display: list-item;"})
# print(links)

links = soup.find_all("li", {"style" : "display: list-item;"})
articles_hrefs = [link.find("a")["href"] for link in links]
print(articles_hrefs)

driver.find_element_by_link_text("다음으로").click()

driver.get(driver.current_url)
soup2 = bs(driver.page_source, "html.parser")
links2 = soup2.find_all("li", {"style" : "display: list-item;"})
articles_hrefs2 = [link.find("a")["href"] for link in links2]
print(articles_hrefs2)
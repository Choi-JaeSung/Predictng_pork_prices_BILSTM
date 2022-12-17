from bs4 import BeautifulSoup as bs
from selenium import webdriver as wd
from selenium.webdriver.support.ui import Select
import requests
import numpy as np
import pandas as pd

path = "C:/Users/skyle/chromedriver/chromedriver"
driver = wd.Chrome(path)

url = "https://news.kbs.co.kr/search/search.do?query=%EB%8F%BC%EC%A7%80#1"
driver.get(url)

# page = driver.page_source
# soup = bs(page, "html.parser")
# links = soup.find_all("li", {"style" : "display: list-item;"})
# print(links)

search_option = driver.find_element_by_id("btn-search-opt")
search_option.click()
date_option = driver.find_element_by_css_selector("#content > div > div.search-result > div.component > div > dl.select.col1 > dd > ul > li:nth-child(5) > label > span")
date_option.click()

# 기간 2011.01.01 ~ 2020.12.31로 설정
date_from = driver.find_element_by_id("date-select-from")
date_from.clear()
date_from.send_keys("2011.01.01")
date_to = driver.find_element_by_id("date-select-to")
date_to.clear()
date_to.send_keys("2020.12.31")
date_btn = driver.find_element_by_css_selector("#content > div > div.search-result > div.component > div > dl.select.col1 > dd > ul > li:nth-child(5) > button")
date_btn.click()

search_btn = driver.find_element_by_css_selector("#content > div > div.search-result > div.section.input > fieldset > button")
search_btn.click()

url = driver.current_url
driver.get(url)
page = driver.page_source
soup = bs(page, "html.parser")

front_url = "https://news.kbs.co.kr"
contents = [] # 기사들
links = soup.find_all("li", {"style" : "display: list-item;"})
print(links)
# contents_hrefs = [link.find("li") for link in links]



# next_btns = driver.find_element_by_css_selector("#content > div > div.search-result > div.paging.type1 > a.next > span")
# next_btns.click()



# while True:
#     if len(contents) == 10:
#         break;
    
#     # 첫페이지 기사 크롤링
#     if len(contents) == 0:
#         # 기사 href 가져오기
#         links = soup.find_all("td", {"class" : "tal"})
#         hrefs = [link.find("a")["href"] for link in links]
        
#         for i in range(len(hrefs)):
#             article_url = front_url + hrefs[i]
#             page_article = requests.get(url= article_url)
#             soup_article = bs(page_article.text, "html.parser")
#             content = soup_article.find("div", {"class" : "v_content"})
#             contents.append(content)
#     # else:
#     #     # 나머지 페이지 기사 크롤링
#     #     for btn_href in btns_hrefs:
#     #         btn_url = front_url + btn_href
#     #         page_btn = requests.get(url= btn_url)
#     #         soup_btn = bs(page_article.text, "html.parser")
            
#     #         links = soup_btn.find_all("td", {"class" : "tal"})
#     #         hrefs = [link.find("a")["href"] for link in links]
            
#     #         for href in hrefs:
#     #             article_url = front_url + href
#     #             page_article = requests.get(url= article_url)
#     #             soup_article = bs(page_article.text, "html.parser")
#     #             content = soup_article.find("div", {"class" : "v_content"})
#     #             contents.append(content)

# driver.close()

# contents = np.array(contents)
# contents = pd.DataFrame(contents)
# contents.to_csv('KAMIS_news.csv', index=False)
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

while True:
    if len(contents) == 12:
        break;
    
    # btns 가져오고, 불필요한 부분 split
    btns = soup.find_all("div", {"class" : "page_num"})
    btns = str(btns[0]).split('\n')
    btns.pop(0)
    btns.pop(0)
    btns.pop(0)
    btns.pop(-1)
    btns.pop(-1)
    
    # page_btn href 가져오기
    btns_hrefs = []
    
    for btn in btns:
        btn = btn.split(' ')
        btn_href = btn[1].split('"')[1]
        btns_hrefs.append(btn_href)
    
    
    # 첫페이지 기사 크롤링
    if len(contents) == 0:
        # 기사 href 가져오기
        links = soup.find_all("td", {"class" : "tal"})
        hrefs = [link.find("a")["href"] for link in links]
        
        for i in range(len(hrefs)):
            article_url = front_url + hrefs[i]
            page_article = requests.get(url= article_url)
            soup_article = bs(page_article.text, "html.parser")
            content = soup_article.find("div", {"class" : "v_content"})
            contents.append(content)
    # else:
    #     # 나머지 페이지 기사 크롤링
    #     for btn_href in btns_hrefs:
    #         btn_url = front_url + btn_href
    #         page_btn = requests.get(url= btn_url)
    #         soup_btn = bs(page_article.text, "html.parser")
            
    #         links = soup_btn.find_all("td", {"class" : "tal"})
    #         hrefs = [link.find("a")["href"] for link in links]
            
    #         for href in hrefs:
    #             article_url = front_url + href
    #             page_article = requests.get(url= article_url)
    #             soup_article = bs(page_article.text, "html.parser")
    #             content = soup_article.find("div", {"class" : "v_content"})
    #             contents.append(content)

# driver.close()

# contents = np.array(contents)
# contents = pd.DataFrame(contents)
# contents.to_csv('KAMIS_news.csv', index=False)

# 1페이지가 아닌 다음으로 넘어갔을 때도 현재 페이지는 따로 구분
# 마지막을 어떻게 설정할 것인가
# 처음을 어떻게 설정할 것인가
from selenium import webdriver
from selenium.common.exceptions import NoSuchElementException #에러 날 때
from selenium.common.exceptions import StaleElementReferenceException
import time
import pandas as pd
import re

options = webdriver.ChromeOptions()
#options.add_argument('headless') # 브라우저 안 보기
options.add_argument('--no-sandbox') #리눅스에서는 줘야함
options.add_argument('--disable-dev-shm-usage')
options.add_argument('disable-gpu') #리눅스에서는 줘야함
options.add_argument('lang=ko_KR')
driver = webdriver.Chrome('./chromedriver.exe', options= options) #실행 파일 있어야 함
driver.implicitly_wait(10) #페이지 여는데 걸리는 시간


""" 예시  
try :
    driver.get(url)
    title = driver.find_element_by_xpath('//*[@id="section_body"]/ul[1]/li[1]/dl/dt[2]/a').text
    title = re.compile('[^가-힣 | a-z | A-Z]').sub('', title)
    print(title)
except NoSuchFrameException :
    print('NoSuchFrameException')
"""


title_list = []
df_section_title = pd.DataFrame()

print('크롤링 시작')
for k in range(1,100):
    url = f'https://news.naver.com/main/main.nhn?mode=LSD&mid=shm&sid1=102#&date=%2000:00:00&page={k}'

    for j in range(1,5):
        for i in range (1,6) :
            try:
                driver.get(url)
                title = driver.find_element_by_xpath(f'//*[@id="section_body"]/ul[{j}]/li[{i}]/dl/dt[2]/a').text
                title = re.compile('[^가-힣 | a-z | A-Z]').sub('', title)

                title_list.append(title)
            except NoSuchElementException:
                print('NoSuchElementException')
            except StaleElementReferenceException :
                print('StaleElementReferenceException')
            except :
                print('other error')
    df_section_title['category'] = 'Social'
driver.close()

df_section_title = pd.DataFrame(title_list, columns=['title'])
df_section_title.to_csv('C:/work/python/exam/prj01_news_category_classification/crawling_data/1.csv')
print('저장완료')
""""

#################
#### 모든 섹션 긁어오기
category = ['Politics', 'Economic','Social', 'Culture', 'World', 'IT']
page_num = [334, 423, 400, 87, 128, 74]
df_title = pd.DataFrame    #모든 섹션을 집어넣을 dataframe

for l in range(0,6):
    df_section_title = pd.DataFrame()
    for k in range(1,400):
        url = f'https://news.naver.com/main/main.nhn?mode=LSD&mid=shm&sid1=10{l}#&date=%2000:00:00&page={k}'
        title_list = []
        for j in range(1,5):
            for i in range (1,6) :
                try:
                    driver.get(url)
                    title = driver.find_element_by_xpath(f'//*[@id="section_body"]/ul[{j}]/li[{i}]/dl/dt[2]/a').text
                    title = re.compile('[^가-힣 | a-z | A-Z]').sub('', title)

                    title_list.append(title)
                except NoSuchElementException:
                    print('NoSuchElementException') # 사진 없는 기사들 가끔 xpath가 다름
                    
    df_section_title = pd.DataFrame(title_list, columns=['title'])
    df_section_title['category'] = category[l]
    df_title = pd.concat([df_title, df_section_title], axis=0, ignore_index=True) #행으로 이어붙임

driver.close()
df_title.head(30)
df_title.to_csv('../crawling_data/naver_news_titles_20210615.csv')


"""
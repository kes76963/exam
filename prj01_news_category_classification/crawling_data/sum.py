import pandas as pd


df = pd.read_csv('C:/work/python/exam/prj01_news_category_classification/crawling_data/1.csv')
df2 = pd.read_csv('C:/work/python/exam/prj01_news_category_classification/crawling_data/2.csv')
df3 = pd.read_csv('C:/work/python/exam/prj01_news_category_classification/crawling_data/3.csv')
df4 = pd.read_csv('C:/work/python/exam/prj01_news_category_classification/crawling_data/4.csv')

df['category'] = 'Social'

df.rename(columns = {"0":'title'},inplace= True)
df2.rename(columns = {"0":'title'},inplace= True)
df4.rename(columns = {"0":'title'},inplace= True)

df.info()
df2.info()
df3.info()
df4.info()

df = pd.concat([df, df2, df3, df4], axis=0, ignore_index=True)


df = df.drop(['Unnamed: 0'], axis=1)

#print(df.head())
df.to_csv('C:/work/python/exam/prj01_news_category_classification/crawling_data//finish.csv', encoding='utf-8-sig') #인코딩하면 한글 파일로 보이는데, 불러올 때 문제

"""
df = pd.read_csv('C:/work/python/exam/prj01_news_category_classification/crawling_data/naver_news_titles_210616_4.csv')
df = df.drop(['Unnamed: 0'], axis=1)
df.rename(columns = {"0":'title'},inplace= True)
df.to_csv('C:/work/python/exam/prj01_news_category_classification/crawling_data//영익.csv')
"""
import pandas as pd
pd.set_option('display.unicode.east_asian_width', True)
df_headline= pd.read_csv('./crawling_data/naver_news_titles_210616_headline.csv',index_col=0) #0번 컬럼을 인덱스로 써라
df_0 = pd.read_csv('./crawling_data/naver_news_titles_210616_0.csv')
df_1 = pd.read_csv('./crawling_data/naver_news_titles_210616_1.csv',index_col=0)
df_2 = pd.read_csv('./crawling_data/naver_news_titles_210616_2.csv',index_col=0)
df_3 = pd.read_csv('./crawling_data/naver_news_titles_210616_3.csv',index_col=0)
df_4 = pd.read_csv('./crawling_data/naver_news_titles_210616_4.csv',index_col=0)
df_5 = pd.read_csv('./crawling_data/naver_news_titles_210616_5.csv',index_col=0)

'''
print(df_5.columns)
print(df_5.head())
print(df_5.info())
'''

df = pd.concat([df_0, df_1, df_2,df_3,df_4,df_5, df_headline], axis=0, ignore_index=True)
df.to_csv('./crawling_data/naver_news_titles_210616.csv')
print(df.head())
print(df.info())
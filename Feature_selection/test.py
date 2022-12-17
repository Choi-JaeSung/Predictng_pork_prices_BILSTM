from numpy.core.fromnumeric import shape
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
# import gensim
# from gensim import corpora
from multiprocessing import freeze_support
from sklearn.preprocessing import MinMaxScaler



path = str(os.path.expanduser(os.path.join('~', 'Desktop', 'Predicting_pork_retail_price','Data')))
os.chdir(path)

# news train, test 분류

# news = pd.read_csv("Pigtimes_news_nouns_filterd_data.csv", sep=',')
# news = np.array(news)

# train_news = []
# test_news = []

# for article in news:
#     article_date = article[0].split('.')[0]
    
#     if article_date == '2020':
#         test_news.append(article)
#     else:
#         train_news.append(article)

# train_news = np.array(train_news)
# test_news = np.array(test_news)

# train_df = pd.DataFrame(train_news)
# test_df = pd.DataFrame(test_news)

# train_df.to_csv('Pigtimes_train_news_data.csv', index=False, header=['Date', 'Nouns']) # len(train_if) = 7515
# test_df.to_csv('Pigtimes_test_news_data.csv',index=False, header=['Date', 'Nouns']) # len(test_if) = 828


if __name__ == '__main__':
    freeze_support()
    
    # top6_topics plot
    
    # lda = gensim.models.ldamodel.LdaModel.load('model/lda_model_iter700_p80.gensim')
    # topics = lda.print_topics()
    
    # top6_topics = []
    
    # for topic in topics:
    #     if len(top6_topics) < 6:
    #         top6_topics.append(list(topic))

    # top6_topics[0][0] = '아프리카 돼지열병'
    # top6_topics[1][0] = '기타 질병'
    # top6_topics[2][0] = '정부 지원과 가격'
    # top6_topics[3][0] = '구제역'
    # top6_topics[4][0] = '사료'
    # top6_topics[5][0] = '축산물 관련 사업'
    
    # topics_rates = []
    # topics_nouns = []
    
    # for topic in top6_topics:
    #     nouns_rate = topic[1].split('+')
    #     rates = []
    #     nouns = []
    #     for noun_rate in nouns_rate:
    #         rate_noun = noun_rate.split('*')
    #         rates.append(float(rate_noun[0]))
    #         nouns.append(rate_noun[1])
        
    #     topics_rates.append(rates)
    #     topics_nouns.append(nouns)
    
    # plt.rc('font', family='Malgun Gothic')
    # plt.suptitle('Top6 Topics', fontsize=28)
    
    # ax1 = plt.subplot(3, 3, 1)
    # plt.bar(topics_nouns[0], topics_rates[0])
    # plt.ylabel('Word Weight')
    # plt.title(top6_topics[0][0], fontsize=16)
    
    # ax2 = plt.subplot(3, 3, 2)
    # plt.bar(topics_nouns[1], topics_rates[1])
    # plt.ylabel('Word Weight')
    # plt.title(top6_topics[1][0], fontsize=16)
    
    # ax3 = plt.subplot(3, 3, 3)
    # plt.bar(topics_nouns[2], topics_rates[2])
    # plt.ylabel('Word Weight')
    # plt.title(top6_topics[2][0], fontsize=16)
    
    # ax4 = plt.subplot(3, 3, 4)
    # plt.bar(topics_nouns[3], topics_rates[3])
    # plt.ylabel('Word Weight')
    # plt.title(top6_topics[3][0], fontsize=16)
    
    # ax5 = plt.subplot(3, 3, 5)
    # plt.bar(topics_nouns[4], topics_rates[4])
    # plt.ylabel('Word Weight')
    # plt.title(top6_topics[4][0], fontsize=16)
    
    # ax6 = plt.subplot(3, 3, 6)
    # plt.bar(topics_nouns[5], topics_rates[5])
    # plt.ylabel('Word Weight')
    # plt.title(top6_topics[5][0], fontsize=16)
    
    # plt.savefig('Top6 Topics')
    # plt.show()
    
    
    # temp_price train, test 분류
    
    # temp_prices = pd.read_csv('Data_etc/Temp_retail_price/Temp_Price_data.csv', sep=',')
    # temp_prices = np.array(temp_prices)
    
    # train_temp_prices = []
    # test_temp_prices = []
    
    # for temp_price in temp_prices:
    #     date = temp_price[0].split('-')[0]
        
    #     if date == '2020':
    #         test_temp_prices.append(temp_price)
    #     else:
    #         train_temp_prices.append(temp_price)
            
    # df_train_temp_prices = pd.DataFrame(train_temp_prices)
    # df_test_temp_prices = pd.DataFrame(test_temp_prices)
    
    # df_train_temp_prices.to_csv('Train_temp_prices.csv', index=False, header=['Date', 'Temp', 'Price'])
    # df_test_temp_prices.to_csv('Test_temp_prices.csv', index=False, header=['Date', 'Temp', 'Price'])
    
    
    # train_temp_prices 정규화
    
    # train_temp_prices = pd.read_csv('Data_etc/Temp_retail_price/Train_temp_prices.csv', sep=',')
    # train_temp_prices = np.array(train_temp_prices)
    
    # date = train_temp_prices[:, 0]
    # date = np.array(date)
    # date = np.reshape(date, (-1, 1))
    
    # temps = train_temp_prices[:, 1]
    # temps = np.array(temps)
    # temps = np.reshape(temps, (-1, 1))
    
    # prices = train_temp_prices[:, 2]
    # prices = np.array(prices)
    # prices = np.reshape(prices, (-1, 1))
    
    # normalization = MinMaxScaler()
    
    # norm_temps = normalization.fit_transform(temps)
    
    # print(normalization.data_min_) # -14.8
    # print(normalization.data_max_) # 33.7
    
    # # normalization = MinMaxScaler()
    
    # # norm_prices = normalization.fit_transform(prices)
    # # print(normalization.data_min_) # 1057
    # # print(normalization.data_max_) # 2431
    
    # norm_train_temp_prices = np.concatenate((date, norm_temps, prices), axis=1)
    
    # df_train_temp_prices = pd.DataFrame(norm_train_temp_prices)
    # df_train_temp_prices.to_csv('Norm_train_temp_prices.csv', index=False, header=['Date','Temp','Price'])
    
    
    # test_temp_prices 정규화
    
    # test_temp_prices = pd.read_csv('Data_etc/Temp_retail_price/Test_temp_prices.csv', sep=',')
    # test_temp_prices = np.array(test_temp_prices)
    
    # date = test_temp_prices[:, 0]
    # date = np.array(date)
    # date = np.reshape(date, (-1, 1))
    
    # temps = test_temp_prices[:, 1]
    # temps = np.array(temps)
    # temps = np.reshape(temps, (-1, 1))
    
    # prices = test_temp_prices[:, 2]
    # prices = np.array(prices)
    # prices = np.reshape(prices, (-1, 1))
    
    # normalization = MinMaxScaler()
    
    # norm_temps = normalization.fit_transform(temps)
    
    # print(normalization.data_min_) # -14.8
    # print(normalization.data_max_) # 33.7
    
    # normalization = MinMaxScaler()
     
    # # norm_prices = normalization.fit_transform(prices)
    # # print(normalization.data_min_) # 1330
    # # print(normalization.data_max_) # 2413
    
    # norm_test_temp_prices = np.concatenate((date, norm_temps, prices), axis=1)
    
    # df_test_temp_prices = pd.DataFrame(norm_test_temp_prices)
    # df_test_temp_prices.to_csv('Norm_test_temp_prices.csv', index=False, header=['Date','Temp','Price'])
    
    
    # test_news TF_IDF 추출
    
    # news = pd.read_csv('Data_etc/PigTimes_news/Pigtimes_test_news_data.csv', sep=',')
    # news = np.array(news)

    # # news[1](nouns)가 리스트로 변환한 news
    # news_prep = []

    # # news[1](nouns)의 리스트가 str로 되어 있기 때문에 eval()로 리스트로 변환
    # for article in news:
    #     nouns = eval(article[1])
    #     news_prep.append([article[0], nouns])

    # news_prep = np.array(news_prep, dtype=np.object_)

    # news_nouns = news_prep[:, 1] # 해당 뉴스 단어 리스트

    # nouns_dict = corpora.Dictionary(news_nouns) # 뉴스에 나온 모든 단어들을 번호붙임 length: 423

    # corpus = [nouns_dict.doc2bow(noun) for noun in news_nouns] # (해당 번호의 단어, freq): 해당 단어가 해당 뉴스의 freq만큼 등장 length: 828
    
    # lda = gensim.models.ldamodel.LdaModel.load('model/lda_model_iter700_p80.gensim')
    
    # TF_IDF = gensim.models.TfidfModel(corpus)
    # tf_idfs_old = TF_IDF[corpus]
    
    # tf_idfs_new = []
    # index = 0
    
    # for words_weight in tf_idfs_old:
    #     tf_idfs = []
        
    #     for word_weight in words_weight:
    #         tf_idfs.append(word_weight[1])
            
    #     tf_idfs_new.append([news_prep[index][0], tf_idfs])
    #     index += 1
    
    # df = pd.DataFrame(tf_idfs_new)
    # df.to_csv("Test_tf_idf.csv", index=False, header=['Date', 'TF_IDF'])
    
    
    # train_dataset 생성
    
    # trian_tf_idfs = pd.read_csv('Data_etc/PigTimes_news/Train_tf_idf.csv', sep=',')
    # trian_tf_idfs = list(np.array(trian_tf_idfs))
    
    # train_temp_prices = pd.read_csv('Data_etc/Temp_retail_price/Norm_train_temp_prices.csv', sep=',')
    # train_temp_prices = list(np.array(train_temp_prices))
    
    # train_dataset = []
    
    # for tf_idfs in trian_tf_idfs:

    #     tf_idfs_date = tf_idfs[0].split('.')
        
    #     for temp_price in train_temp_prices:
    #         temp_price_date = temp_price[0].split('-')

    #         if tf_idfs_date[0] == temp_price_date[0] and \
    #            tf_idfs_date[1] == temp_price_date[1] and \
    #            tf_idfs_date[2] == temp_price_date[2]:
                   
    #             train_dataset.append([temp_price[0], tf_idfs[1], temp_price[1], temp_price[2]])
    
    # train_dataset = np.array(train_dataset)
    # train_dataset = np.flip(train_dataset, axis=0)
    
    # df_train_dataset = pd.DataFrame(train_dataset)
    # df_train_dataset.to_csv('Train_dataset.csv', index=False, header=['Date', 'TF_IDF', 'Temp', 'Price'])
    
    
    # # test_dataset 생성
    
    # test_tf_idfs = pd.read_csv('Data_etc/PigTimes_news/Test_tf_idf.csv', sep=',')
    # test_tf_idfs = list(np.array(test_tf_idfs))
    
    # test_temp_prices = pd.read_csv('Data_etc/Temp_retail_price/Norm_Test_temp_prices.csv', sep=',')
    # test_temp_prices = list(np.array(test_temp_prices))
    
    # test_dataset = []
    
    # for tf_idfs in test_tf_idfs:

    #     tf_idfs_date = tf_idfs[0].split('.')
        
    #     for temp_price in test_temp_prices:
    #         temp_price_date = temp_price[0].split('-')

    #         if tf_idfs_date[0] == temp_price_date[0] and \
    #            tf_idfs_date[1] == temp_price_date[1] and \
    #            tf_idfs_date[2] == temp_price_date[2]:
                   
    #             test_dataset.append([temp_price[0], tf_idfs[1], temp_price[1], temp_price[2]])
    
    # test_dataset = np.array(test_dataset)
    # test_dataset = np.flip(test_dataset, axis=0)
    
    
    # df_test_dataset = pd.DataFrame(test_dataset)
    # df_test_dataset.to_csv('Test_dataset.csv', index=False, header=['Date', 'TF_IDF', 'Temp', 'Price'])
    
    
    
    # price, temp plot 추출
    
    temp_prices= pd.read_csv('Data_etc/Temp_retail_price/Temp_Price_data.csv', sep=',')
    
    # prices = temp_prices['Price']
    years = [year for year in range(2011, 2021, 1)]
    
    # plt.figure(figsize=(15, 7))
    # plt.plot(prices, label='100g')
    # plt.rc('legend', fontsize=40)
    # plt.legend(loc='lower right')
    
    # # plt.title('Pork Price (2011 ~ 2020)',fontsize=25)
    
    # plt.xlabel('Date', fontsize=40, labelpad=10)
    # plt.ylabel('Price(₩)', fontsize=40, labelpad=10)
    
    # plt.xticks(np.arange(0, len(prices), 247), years)
    # plt.tick_params(axis='x', length=0, labelsize=40)
    # plt.tick_params(axis='y', length=0, labelsize=40)
    
    # plt.savefig('Pork Price.png')
    # plt.show()
    
    
    temps = temp_prices['Temp']
    
    # print(min(temps)) # -14.8
    # print(max(temps)) # 33.7
    # print(min(prices)) # 1057
    # print(max(prices)) # 2431
    
    plt.figure(figsize=(15, 7))
    plt.plot(temps)
    
    # plt.title('Temp (2011 ~ 2020)',fontsize=25)
    
    plt.xlabel('Date', fontsize=40, labelpad=10)
    plt.ylabel('Temp(℃)', fontsize=40, labelpad=10)
    
    plt.xticks(np.arange(0, len(temps), 247), years)
    plt.tick_params(axis='x', length=0, labelsize=40)
    plt.tick_params(axis='y', length=0, labelsize=40)
    
    plt.savefig('Temp.png')
    plt.show()
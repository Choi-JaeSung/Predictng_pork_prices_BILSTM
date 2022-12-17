import numpy as np
import pandas as pd
import os
import gensim
from gensim import corpora
from gensim.models.coherencemodel import CoherenceModel
from multiprocessing import freeze_support

path = str(os.path.expanduser(os.path.join('~', 'SW_App', 'Predicting_pork_retail_price', 'Data')))
os.chdir(path)

if __name__ == '__main__':
    freeze_support()

    news = pd.read_csv('Data_etc/PigTimes_news/Pigtimes_train_news_data.csv', sep=',')
    news = np.array(news)

    # news[1](nouns)가 리스트로 변환한 news
    news_prep = []

    # news[1](nouns)의 리스트가 str로 되어 있기 때문에 eval()로 리스트로 변환
    for article in news:
        nouns = eval(article[1])
        news_prep.append([article[0], nouns])

    news_prep = np.array(news_prep, dtype=np.object_)

    news_nouns = news_prep[:, 1] # 해당 뉴스 단어 리스트

    nouns_dict = corpora.Dictionary(news_nouns) # 뉴스에 나온 모든 단어들을 번호붙임 length: 423

    corpus = [nouns_dict.doc2bow(noun) for noun in news_nouns] # (해당 번호의 단어, freq): 해당 단어가 해당 뉴스의 freq만큼 등장 length: 8343
    
    lda = gensim.models.ldamodel.LdaModel.load('model/lda_model_iter700_p80.gensim')
    
    # topics = lda.print_topics() # topics 추출

    # for topic in topics:
    #     print(topic)
    
    for i, topic_list in enumerate(lda[corpus]):
        if i==6:
            break
        # print(i, '번째 문서의 topic 비율은', topic_list)
    
    TF_IDF = gensim.models.TfidfModel(corpus)
    tf_idfs_old = TF_IDF[corpus]
    # print(tf_idfs_old[0])
    # tf_idfs_old = np.array(tf_idfs_old)
    
    tf_idfs_new = []
    index = 0
    
    for words_weight in tf_idfs_old:
        tf_idfs = []
        
        for word_weight in words_weight:
            tf_idfs.append(word_weight[1])
            
        tf_idfs_new.append([news_prep[index][0], tf_idfs])
        index += 1
    
    # print(tf_idfs_new[0])
    
    # tf_idfs_new = [tf_idf for tf_idf in zip(*tf_idfs_new)]
    
    # print(len(tf_idfs_new))
    # print(len(tf_idfs_new[0]))
    
    df = pd.DataFrame(tf_idfs_new)
    df.to_csv("Train_tf_idf.csv", index=False, header=['Date', 'TF_IDF'])
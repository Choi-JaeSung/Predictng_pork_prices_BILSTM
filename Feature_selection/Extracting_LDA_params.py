import numpy as np
import pandas as pd
import os
import gensim
from gensim import corpora
from gensim.models.coherencemodel import CoherenceModel
from multiprocessing import freeze_support
import matplotlib.pyplot as plt

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

    # ldamodel = gensim.models.ldamodel.LdaModel(corpus=corpus, num_topics=NUM_TOPIC, id2word=nouns_dict, passes=30)

    # topics = ldamodel.print_topics()

    # for topic in topics:
    #     print(topic)

    # for i, topic_list in enumerate(ldamodel[corpus]):
    #     if i==6:
    #         break
    #     print(i, '번째 문서의 topic 비율은', topic_list)

    topic_range = [i for i in range(11, 41)]
    coherences=[]
    perplexities=[]

    for i in range(30):
        
        if i == 0:
            NUM_TOPIC = 1
        else:
            NUM_TOPIC += 1
                
        lda = gensim.models.ldamodel.LdaModel(corpus=corpus, num_topics=NUM_TOPIC, id2word=nouns_dict)
        
        cm = CoherenceModel(model=lda, corpus=corpus, texts=news_nouns, coherence='c_v')
        
        coherence = cm.get_coherence()
        coherences.append(coherence)
        
        # perplexities.append(lda.log_perplexity(corpus))

    # ax1 = plt.subplot(2, 1, 1)
    plt.plot(topic_range, coherences)
    plt.ylabel('Coherences')
    plt.xticks(visible=False)
    
    # ax2 = plt.subplot(2, 1, 2, sharex=ax1)
    # plt.plot(topic_range, perplexities)
    # plt.xlabel('Num_topics')
    # plt.ylabel('Perplexities')
    
    plt.savefig('Extracting_Num_topics.png')
    plt.show()
    
    print(min(coherences))
    print(max(coherences))
    
# 결과는 topic 개수가 12개일 때 coherences가 가장 높았다
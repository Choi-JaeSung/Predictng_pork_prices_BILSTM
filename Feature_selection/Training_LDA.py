import numpy as np
import pandas as pd
import os
import gensim
from gensim import corpora
from gensim.models.coherencemodel import CoherenceModel
from multiprocessing import freeze_support
from matplotlib import pyplot as plt

path = str(os.path.expanduser(os.path.join('~', 'SW_App', 'Predicting_pork_retail_price', 'Data')))
os.chdir(path)

if __name__ == '__main__':
    freeze_support()

    news = pd.read_csv('Pigtimes_train_news_data.csv', sep=',')
    news = np.array(news)

    # news[1](nouns)가 리스트로 변환한 news
    news_prep = []

    # news[1](nouns)의 리스트가 str로 되어 있기 때문에 eval()로 리스트로 변환
    for article in news:
        nouns = eval(article[1])
        news_prep.append([article[0], nouns])

    news_prep = np.array(news_prep, dtype=np.object_) # len(news_prep) = 7515

    news_nouns = news_prep[:, 1] # train 뉴스 단어 리스트
    
    nouns_dict = corpora.Dictionary(news_nouns) # 뉴스에 나온 모든 단어들을 번호붙임 length: 423
    nouns_dict.filter_extremes(no_above=0.5)
    
    corpus = [nouns_dict.doc2bow(noun) for noun in news_nouns] # (해당 번호의 단어, freq): 해당 단어가 해당 뉴스의 freq만큼 등장 length: 8343

    NUM_TOPIC = 12

    coherences=[]
    perplexities=[]
                
    lda = gensim.models.ldamodel.LdaModel(corpus=corpus, num_topics=NUM_TOPIC, id2word=nouns_dict, iterations=700, passes=80, eta='auto', alpha='auto')

    topics = lda.print_topics()
    
    for topic in topics:
        print(topic)
    
    
    cm = CoherenceModel(model=lda, corpus=corpus, texts=news_nouns, coherence='c_v')
    coherence = cm.get_coherence()
    print("Coherence: ", coherence)
    coherences.append(coherence)
    print('Perplexity: ', lda.log_perplexity(corpus),'\n\n')
    perplexities.append(lda.log_perplexity(corpus))

    lda.save('lda_model_iter700_p80.gensim')
            
    # 700 80 54944
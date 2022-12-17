import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from multiprocessing import freeze_support
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler


path = str(os.path.expanduser(os.path.join('~', 'Desktop', 'Predicting_pork_retail_price','Data')))
os.chdir(path)

if __name__ == '__main__':
    freeze_support()
    
    test_data = pd.read_csv('Test_dataset.csv', sep=',')
    test_data = np.array(test_data)
    
    # print('입력변수 tf_idf 길이: {0}'.format(len(test_data[:, 1]))) # 822
    # print('입력변수 temp 길이: {0}'.format(len(test_data[:, 2]))) # 822
    # print('출력변수 Price 길이: {0}'.format(len(test_data[:, 3]))) # 822
    
    test_tf_idfs = [eval(tf_idfs) for tf_idfs in test_data[:, 1]] # str로 저장된 list -> list로 casting
    
    # tf_idf 개수 분포 추출
    
    len_tf_idfs = 0
    max_len_tf_idf = 0
    min_len_tf_idf = 100

    test_data = np.delete(test_data, 1, axis=1)
    test_data = np.insert(test_data, 1, test_tf_idfs, axis=1)
    test_data_copy = np.copy(test_data)
    
    for i in range(len(test_data_copy)):
        if len(test_data_copy[i, 1]) == 0:
            test_data = np.delete(test_data, i, axis=0)
    
    for tf_idf in test_data[:, 1]:
        len_tf_idfs += len(tf_idf)
        
        if max_len_tf_idf < len(tf_idf):
            max_len_tf_idf = len(tf_idf)
        
        if min_len_tf_idf > len(tf_idf):
            min_len_tf_idf = len(tf_idf)
    
    X_test = test_data[:, 1]
    
    # print(min_len_tf_idf) # 4
    # print('단어 최대 길이: {0}'.format(max_len_tf_idf)) # 172
    # print('단어 평균 길이: {0}'.format((len_tf_idfs / len(X_test)))) # 26.69099

    # plt.hist([len(x) for x in X_test])
    # plt.title('Length Distribution of Test Data')
    # plt.xlabel('Length of Data')
    # plt.ylabel('Number of Data')
    
    # plt.annotate('max: 172\nmin: 4', (155, 330))
    
    # plt.savefig('Length Distribtion of Test Data.png')
    # plt.show()

    # 뉴스마다 tf_idf(단어) 수가 다르기 때문에 train_data의 max값으로 padding

    tf_idf_pad = [] # 동일한 개수로 맞춘 tf_idf
    
    for tf_idf in X_test:
        if len(tf_idf) < 183:
            tf_idf_pad.append(np.pad(tf_idf, (0, 183 - len(tf_idf)), 'constant', constant_values=0))
        else:
            tf_idf_pad.append(tf_idf)
    
    X_test = np.array(tf_idf_pad)
    X_test = np.insert(X_test, X_test.shape[-1], test_data[:, 2], axis=1)
    # print("X_test shape: {0}".format(X_test.shape)) # (7459, 184)
    X_test = np.reshape(X_test, newshape=(X_test.shape[0],X_test.shape[-1], 1))
    X_test = np.asarray(X_test).astype(np.float32)
    
    Y_test = test_data[:, 3]
    Y_test = np.reshape(Y_test, (-1, 1))
    
    normalization = MinMaxScaler()
    Y_test = normalization.fit_transform(Y_test)
                                          
    Y_test = np.asarray(Y_test).astype(np.float32)
    
    #-----------------------------test------------------------------
    
    
    model = load_model('model/lstm.h5')
    
    loss, rmse, mae, mape = model.evaluate(X_test, Y_test, batch_size=90, verbose=1)
    # print('rmse: {0}'.format(rmse)) # 2.7037 (loss= 7.3102)

    pred = model.predict(X_test, batch_size=90)
    pred = normalization.inverse_transform(pred)
    
    Y_test = normalization.inverse_transform(Y_test)
    
    print('y_predict 1: {0}'.format(pred[0]))
    print('y_act 1: {0}'.format(Y_test[0]))
    print('y_predict 10: {0}'.format(pred[10]))
    print('y_act 10: {0}'.format(Y_test[10]))
    print('y_predict 100: {0}'.format(pred[100]))
    print('y_act 100: {0}'.format(Y_test[100]))
    
    plt.figure(figsize=(12, 9))
    plt.plot(Y_test, label='actual')
    plt.plot(pred, label='prediction')
    plt.rc('legend', fontsize=40)
    plt.legend(loc='lower right')
    # plt.title('Predicting result for Test Data(LSTM)', fontsize=20)
    plt.xlabel('Num of Data', fontsize=40)
    plt.ylabel('Price', fontsize=40)
    plt.tick_params(axis='x', length=0, labelsize=40)
    plt.tick_params(axis='y', length=0, labelsize=40)
    
    plt.savefig('Testing result(LSTM).png')
    plt.show()
    
    """
    y_predict 1: [1685.3269]
    y_act 1: 1684.0
    y_predict 10: [1672.2687]
    y_act 10: 1671.0
    y_predict 100: [1446.1075]
    y_act 100: 1451.0
    """
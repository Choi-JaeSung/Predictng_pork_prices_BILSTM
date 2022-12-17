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
    X_test = np.asarray(X_test).astype(np.float32)
    
    Y_test = test_data[:, 3]
    Y_test_norm = np.reshape(Y_test.copy(), (-1, 1))
    
    # prices 정규화
    
    normalization = MinMaxScaler()
    Y_test_norm = normalization.fit_transform(Y_test_norm)
    # print(normalization.data_min_) # 1057
    # print(normalization.data_max_) # 2431
    Y_test_norm = np.asarray(Y_test_norm).astype(np.float32)
    Y_test = np.asarray(Y_test).astype(np.float32)
    
    #-----------------------------test------------------------------
    
    
    bilstm = load_model('model/bilstm.h5')
    
    loss, rmse, mae, mape = bilstm.evaluate(X_test, Y_test, batch_size=90, verbose=1)
    # print('rmse: {0}'.format(rmse)) # 2.7037 (loss= 7.3102)

    pred_bilstm = bilstm.predict(X_test, batch_size=90)
    print('BILSTM')
    print('RMSE: {0}'.format(rmse))
    print('MAE: {0}'.format(mae))
    print('MAPE: {0}'.format(mape))
    
    rnn = load_model('model/rnn.h5')
    
    loss, rmse, mae, mape = rnn.evaluate(X_test, Y_test_norm, batch_size=90, verbose=1)
    # print('rmse: {0}'.format(rmse)) # 2.7037 (loss= 7.3102)

    pred_rnn = rnn.predict(X_test, batch_size=90)
    pred_rnn = normalization.inverse_transform(pred_rnn)
    print('RNN')
    print('RMSE: {0}'.format(rmse))
    print('MAE: {0}'.format(mae))
    print('MAPE: {0}'.format(mape))
    
    lstm = load_model('model/lstm.h5')
    
    loss, rmse, mae, mape = lstm.evaluate(X_test, Y_test_norm, batch_size=90, verbose=1)
    
    pred_lstm = lstm.predict(X_test, batch_size=90)
    pred_lstm = normalization.inverse_transform(pred_lstm)
    print('LSTM')
    print('RMSE: {0}'.format(rmse))
    print('MAE: {0}'.format(mae))
    print('MAPE: {0}'.format(mape))
    
    Y_test_norm = normalization.inverse_transform(Y_test_norm)
    
    plt.figure(figsize=(12, 9))
    plt.plot(Y_test_norm, label='actual')
    # plt.plot(pred_bilstm, label='BILSTM')
    # plt.plot(pred_lstm, label='LSTM')
    plt.plot(pred_rnn, label='RNN')
    plt.title('Predicting result for Test Data', fontsize=20)
    plt.xlabel('Num of Data', fontsize=14)
    plt.ylabel('Price', fontsize=14)
    plt.legend()
    plt.savefig('Testing result.png')
    plt.show()
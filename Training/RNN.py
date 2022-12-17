import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import tensorflow as tf
from multiprocessing import freeze_support
from tensorflow.keras.layers import Dense, SimpleRNN
from tensorflow.keras.models import Sequential
from tensorflow.keras import optimizers
from sklearn.preprocessing import MinMaxScaler


path = str(os.path.expanduser(os.path.join('~', 'SW_App', 'Predicting_pork_retail_price', 'Data')))
os.chdir(path)

if __name__ == '__main__':
    freeze_support()
    
    
    train_data = pd.read_csv('Train_dataset.csv', sep=',')
    train_data = np.array(train_data)
    
    # print('입력변수 tf_idf 길이: {0}'.format(len(train_data[:, 1]))) # 7460
    # print('입력변수 temp 길이: {0}'.format(len(train_data[:, 2]))) # 7460
    # print('출력변수 Price 길이: {0}'.format(len(train_data[:, 3]))) # 7460
    
    train_tf_idfs = [eval(tf_idfs) for tf_idfs in train_data[:, 1]] # str로 저장된 list -> list로 casting

    # tf_idf 개수 분포 추출
    
    len_tf_idfs = 0
    max_len_tf_idf = 0
    min_len_tf_idf = 100

    train_data = np.delete(train_data, 1, axis=1)
    train_data = np.insert(train_data, 1, train_tf_idfs, axis=1)
    train_data_copy = np.copy(train_data)
    
    for i in range(len(train_data_copy)):
        if len(train_data_copy[i, 1]) == 0:
            train_data = np.delete(train_data, i, axis=0)
    
    for tf_idf in train_data[:, 1]:
        len_tf_idfs += len(tf_idf)
        
        if max_len_tf_idf < len(tf_idf):
            max_len_tf_idf = len(tf_idf)
        
        if min_len_tf_idf > len(tf_idf):
            min_len_tf_idf = len(tf_idf)
            
            
    X_train = train_data[:, 1]
    
    # print(min_len_tf_idf) # 1
    # print('단어 최대 길이: {0}'.format(max_len_tf_idf)) # 183
    # print('단어 평균 길이: {0}'.format((len_tf_idfs / len(X_train)))) # 23.57479

    # plt.hist([len(x) for x in X_train])
    # plt.title('Length Distribution of Data')
    # plt.xlabel('Length of Data')
    # plt.ylabel('Number of Data')
    
    # plt.annotate('max: 183\nmin: 1', (160, 3500))
    
    # plt.savefig('Length Distribtion of Data.png')
    # plt.show()
    
    
    # 뉴스마다 tf_idf(단어) 수가 다르기 때문에 max값으로 padding
    
    tf_idf_pad = [] # 동일한 개수로 맞춘 tf_idf
    
    for tf_idf in X_train:
        if len(tf_idf) < max_len_tf_idf:
            tf_idf_pad.append(np.pad(tf_idf, (0, max_len_tf_idf - len(tf_idf)), 'constant', constant_values=0))
        else:
            tf_idf_pad.append(tf_idf)
    
    X_train = np.array(tf_idf_pad)
    X_train = np.insert(X_train, X_train.shape[-1], train_data[:, 2],axis=1)
    # print("X_train shape: {0}".format(X_train.shape)) # (7459, 184)
    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
    # X_train = np.reshape(X_train, newshape=(60, X_train.shape[0], X_train.shape[-1]))
    # X_train = np.reshape(X_train, newshape=(7459,184, 1))
    X_train = np.asarray(X_train).astype(np.float32)
    
    Y_train = train_data[:, 3]
    Y_train = np.reshape(Y_train, (-1, 1))
    
    # prices 정규화
    
    normalization = MinMaxScaler()
    Y_train = normalization.fit_transform(Y_train)
    # print(normalization.data_min_) # 1057
    # print(normalization.data_max_) # 2431
    
    Y_train = np.asarray(Y_train).astype(np.float32)
    
    
    #-----------------------------train------------------------------
    
    optmizer = optimizers.Adam(learning_rate=0.001)
    
    model = Sequential()
    model.add(SimpleRNN(32, input_shape=(None, X_train.shape[-1]), activation="relu"))
    model.add(Dense(units=1, activation='relu'))
    model.compile(optimizer=optmizer, loss='MeanSquaredError', metrics=['RootMeanSquaredError', 'MeanAbsoluteError', 'MeanAbsolutePercentageError'])
    model.summary()
    
    history = model.fit(X_train, Y_train, epochs=500, batch_size=90, validation_split=0.2)
    
    loss, rmse, mae, mape = model.evaluate(X_train, Y_train, batch_size=90, verbose=1)
    # print('rmse: {0}'.format(rmse))
    
    pred = model.predict(X_train, batch_size=90)
    pred = normalization.inverse_transform(pred)
    
    Y_train = normalization.inverse_transform(Y_train)
    
    print('y_predict 1: {0}'.format(pred[0]))
    print('y_act 1: {0}'.format(Y_train[0]))
    print('y_predict 10: {0}'.format(pred[10]))
    print('y_act 10: {0}'.format(Y_train[10]))
    print('y_predict 100: {0}'.format(pred[100]))
    print('y_act 100: {0}'.format(Y_train[100]))
    
    model.save('model/rnn.h5')
    
    # bilism 16(relu), dense 1(relu) epoch=1000, lr=0.00001 => rmse=2.0992
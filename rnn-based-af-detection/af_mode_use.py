"""
af_model_crossvalidation.py

fitting a simple bidirectional LSTM model to the af data - now including extra
dropout, an extra fully connected layer, and using the keras functional model

the ideas behind the bidirectional lstm model come from: 
    
https://machinelearningmastery.com/
develop-bidirectional-lstm-sequence-classification-python-keras/

this file runs stratified 10-fold crossvalidation
이 파일은 stratified 10 flod 교차검증을 실행한다

author:     alex shenfield
date:       01/04/2018
"""

# file handling functionality
import os

# useful utilities
import time
import pickle

# let's do datascience ...
import numpy as np

# import keras deep learning functionality
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Bidirectional
from tensorflow.keras.layers import GlobalMaxPool1D
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout

from tensorflow.keras.optimizers import SGD
from tensorflow.keras.optimizers import Adam

from tensorflow.keras.callbacks import ModelCheckpoint

# seed = 1337
# np.random.seed(seed)

#
# get the data
# 데이터를 가져옴

# load the npz file
# npz 파일 load(numpy 배열 파일)
data_path = './data/training_data.npz'
af_data   = np.load(data_path)

# extract the training and validation data sets from this data
# 데이터에서 훈련용 데이터와 검증용 데이터 추출
x_data = af_data['x_data']
y_data = af_data['y_data']

# reshape the data fto be in the format that the lstm wants
# lstm이 원하는 형식으로 데이터를 재구성
x_data = x_data.reshape(x_data.shape[0], x_data.shape[1], 1)

end = 2000
x_data = x_data[0:end]
y_data = y_data[0:end]

from tensorflow.keras.models import load_model
model = load_model('af_lstm_weights.30-0.03.hdf5')
# result = model.predict(x_data)


result = model.predict(x_data, batch_size=1024, verbose=0)

print(result)


for i in range(len(result)):
    if result[i] < 0.99:
        result[i] = 0
    else :
        result[i] = 1

'''
j = 0
for i in range(len(result)):
    if result[i] != y_data[i]:
        j = j+1
    print(i,' result: ', result[i], ' y_data: ', y_data[i])
print(j)
'''

f = open('result.txt', 'w')

temp = result.tolist()
result_list = []
for element in temp:
    result_list += element

for i in range(len(result_list)):
    result_list[i] = str(int(result_list[i]))

for i in range(len(result_list)):
    f.write(result_list[i]+'\n')

f.close()

# we're all done!
print('all done!')

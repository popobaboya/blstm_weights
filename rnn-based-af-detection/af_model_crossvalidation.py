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

# fix random seed for reproduciblity
# 랜덤성을 제어하기 위함, 난수의 생성 패턴을 동일하게 관리
seed = 1337
np.random.seed(seed)

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

#
# create and train the model
# 모델 생성 및 훈련

# set the model parameters
# 모델 파라미터 설정
n_timesteps = x_data.shape[1]
mode = 'concat' # concat 은 데이터를 연결하는 함수, 해당 구문의 정확한 의미는 모르겠음, 변수가 사용이 안됨, conact 은 lstm 의 양방향 모드를 의미하는 듯
n_epochs = 80
batch_size = 1


# epoch => 설정 전체 데이터셋을 1회 학습시킨 것이 1epoch
# batch size => 한번의 연산에 들어가는 데이터의 크기, 너무 크면 학습 속도 저하, 메모리 부족 문제 발생할 수 있다.
# 반대로 너무 작으면 적은 데이터로 가중치는 업데이트 하고, 이 업데이트가 자주 발생하므로 훈련이 불안정해진다.

# create a bidirectional lstm model (based around the model in:
# https://www.kaggle.com/jhoward/improved-lstm-baseline-glove-dropout
# )
# 양방향 lstm 모델 생성, x를 생성하고 생성한 x를 새로운 함수에 넣어서 다시 x에 넣고를 반복
inp = Input(shape=(n_timesteps,1,))
x = Bidirectional(LSTM(200,
                       return_sequences=True,
                       dropout=0.1,
                       recurrent_dropout=0.1))(inp)

# return_sequences : last output return 여부
# dropout : 0과 1사이 값, input 의 선형 변환을 위해 삭제할 단위의 비율
# recurrent_dropout : 0과 1 사이 값, recurrent state 의 선형 변환을 위해 삭제할 단위의 비율

x = GlobalMaxPool1D()(x) # 1d 시간 데이터에 대한 global max pooling 작업
x = Dense(50, activation="relu")(x) # dense 는 layer 관련 함수, 50 은 노드 수, relu 는 은닉층으로 학습
x = Dropout(0.1)(x) # dropuout 은 전체 가중치를 사용하는 것이 아닌 일부만 참여시켜 과적합을 방지
x = Dense(1, activation='sigmoid')(x)  # sigmoid 는 이진분류 문제
model = Model(inputs=inp, outputs=x)  # Model 함수는 group layers 를 객체화

# set the optimiser
# optimizer 설정
# 딥러닝에서 모델을 설계할 때 파라미터를 최적화 시켜야 좋은 성능을 보여준다.
# optimizer 는 학습 프로세스에서 파라미터를 갱신시킴으로 파라미터를 최적화 시키는 역할을 한다.
opt = Adam()

# compile the model
# 모델 컴파일
model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['acc'])

# get the initial weights
# 초기 가중치를 가져옴
initial_weights = model.get_weights()

#
# do stratified k-fold crossvalidation on the model
# 모델에 stratified k-flod 교차검증 실행

# progress ...
print('doing cross validation ...')

# set the root directory for results
# 결과를 위한 루트 디렉토리 설정
results_dir = './model/cross_validation_{0}/'.format(
        time.strftime("%Y%m%d_%H%M"))

# import stratified k-fold functions
# k-폴드 라이브러리 import
from sklearn.model_selection import StratifiedKFold

# create the kfold object with 10 splits
# kflod 를 10 개 생성
n_folds = 10
kf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)

# store some data from the crossvalidation process
# 교차검증 프로세스에서 몇개의 데이터를 저장
xval_history = list()
final_accuracy = list()
final_loss = list()

# do cross validation
# 교차검증 실행
fold = 0
for train_index, test_index in kf.split(x_data, y_data):

    # progress ...
    print('evaluating fold {0}'.format(fold))

    # set up a model checkpoint callback (including making the directory where
    # to save our weights)
    # 모델 checkpoint callback 세팅 ( + 가중치 저장 디렉토리 설정)
    directory = results_dir + 'fold_{0}/'.format(fold)
    os.makedirs(directory)
    filename  = 'af_lstm_weights.{epoch:02d}-{val_loss:.2f}.hdf5'
    checkpointer = ModelCheckpoint(filepath=directory+filename,
                                   verbose=0,
                                   save_best_only=True)

# modelcheckpoint 는 모델이 학습하면서 정의한 조건을 만족했을 때 가중치를 중간저장 함
# 중간에 memory overflow 나 crash 가 났을 경우 weight 을 불러와 학습을 이어나갈 수 있기 때문
    # get train and test data
    x_train, y_train = x_data[train_index], y_data[train_index]
    x_test, y_test   = x_data[test_index], y_data[test_index]

    # train the model
    # 모델 학습
    history = model.fit(x_train,
                        y_train,  # 훈련데이터, x : 입력 데이터, y: 라벨 값
                        epochs=n_epochs,  #epoch 횟수
                        batch_size=batch_size,  # 기울기를 업데이트 할 샘플의 갯수
                        verbose=0, # 학습의 진행상황을 보여줄지 여부, 1이면 보여준다
                        validation_data=(x_test, y_test), # 검증 데이터 설정
                        callbacks=[checkpointer])
                        # 훈련 진행 중 적용 될 collback 의 리스트
    # fit 은 모델 학습 함수이다
# fit 의 return 값으로 history 객체를 얻을 수 있다.


    # run the final model on the validation data and save the predictions and
    # the true labels so we can plot a roc curve at a later date
    # 검증데이터로 최종 모델을 실행하고 예측값과 실제 레이블을 저장, 나중에 plot을 만들 수 있도록
    y_predict = model.predict(x_test, batch_size=batch_size, verbose=0)
    np.save(directory + 'test_predictions.npy', y_predict)
    np.save(directory + 'test_labels.npy', y_test)

    # store the training history
    # 학습기록 저장
    xval_history.append(history.history)

    # print the validation result
    # 검증 결과 출력
    final_loss.append(history.history['val_loss'][-1])
    final_accuracy.append(history.history['val_acc'][-1])
    print('validation loss is {0} and accuracy is {1}'.format(final_loss[-1],
          final_accuracy[-1]))

    # reset the model weights
    # 모델의 가중치 초기화
    model.set_weights(initial_weights)

    # next fold ...
    # fold 값을 1 증가시키고 for loop
    fold = fold + 1

#
# tidy up ...
#

# print the final results
# 최종결과 출력
print('overall performance:')
print('{0:.5f}% (+/- {1:.5f}%)'.format(
        np.mean(final_accuracy),
        np.std(final_accuracy))
     )

# pickle the entire cross validation history so we can use it later
# 교차검증 기록 전체를 pickle

with open(results_dir + 'xval_history', 'wb') as file:
    pickle.dump(xval_history, file) # pickle 은 문자열이나 값을 한 번에 전달, write 대신 사용하는 거라고 보면 될 듯

# we're all done!
print('all done!')
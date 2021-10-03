## Detecting Atrial Fibrillation from heart rate data using deep learning

This is the code repository to go along with the paper:

"Automated detection of atrial fibrillation using long short-term memory network with RR interval signals"

https://www.sciencedirect.com/science/article/pii/S0010482518301847

이것은 다음 논문과 함께 사용할 코드 repo입니다.

"RR 간격 신호를 사용한 장기 단기 메모리 네트워크를 사용하여 심방세동 자동 검출"

https://www.sciencedirect.com/science/article/pii/S0010482518301847

#### TLDR

I have trained a bidirectional LSTM model on the atrial fibrillation data sequences extracted from the physionet database to 99.3% accuracy (see Figure 1 below).

TLDR
나는 physionet 데이터베이스에서 추출한 심방세동 데이터 시퀀스에 대한 양방향 LSTM 모델을 99.3% 정확도로 교육했다(아래 그림 1 참조).

<p float="left">
  <img src="./results/initial_af_lstm_training.png" width="400" />
</p>

This is a fantastic result and very promising. My model looks like:
이것은 환상적인 결과이며 매우 유망합니다. 내 모델은 다음과 같습니다.

~~~python
# create a bidirectional lstm model (based around the model in:
# https://www.kaggle.com/jhoward/improved-lstm-baseline-glove-dropout
# )
inp = Input(shape=(n_timesteps,1,))
x = Bidirectional(LSTM(200, 
                       return_sequences=True, 
                       dropout=0.1, recurrent_dropout=0.1))(inp)
x = GlobalMaxPool1D()(x)
x = Dense(50, activation="relu")(x)
x = Dropout(0.1)(x)
x = Dense(1, activation='sigmoid')(x)
model = Model(inputs=inp, outputs=x)

~~~

#### More details ...

I also evaluated the model using both stratified 10-fold cross-validation and blind-fold validation (using completely held out patient data). Stratified 10-fold cross-validation produced the average accuracy and loss plots shown below:

나는 또한 stratified 10 분할 교차 검증과 블라인드(blind-fold) 검증을 모두 사용하여 모델을 평가했습니다(완전하게 유지된 환자 데이터 사용). stratified 10분할 교차 검증을 통해 아래 표시된 평균 정확도 및 손실 그림이 생성되었습니다.

<p>
  <img src="./results/crossvalidation_accuracy.png" width="400" />
  <img src="./results/crossvalidation_loss.png" width="400" />
</p>

And we achieved a mean AUC of 0.9986 from the 10-fold cross-validation process.

Blind-fold validation performed even better (almost certainly due to the smaller held out data set used in the blind-fold validation process).

#### Data?

I have provided the processed data sequences (in csv format) as a zip file in the data directory of this repository.

Data is in 100 beat sequences with a 99 beat overlap.  The first element of every row in the data file specifies how many beats in each sequence were annotated as exhibiting signs of atrial fibrillation.

To recreate the datasets used in the paper just extract the zip file somewhere, fix the directory in the code, and run the data exploration scripts in the data directory.  As an added bonus you should get some nice plots too! :)


그리고 우리는 10분할 교차 검증 과정을 통해 0.9986의 평균 정확도를 달성했습니다.

블라인드 폴드 검증 프로세스에서 사용되는 홀드아웃 데이터 세트가 작기 때문에 블라인드 폴드 검증 성능이 더욱 우수했습니다.

데이터?
처리된 데이터 시퀀스(csv 형식)를 이 리포지토리의 데이터 디렉토리에 zip 파일로 제공했습니다.

데이터는 100개의 비트 시퀀스에 99개의 비트 오버랩이 있습니다. 데이터 파일의 모든 행의 첫 번째 요소는 각 시퀀스에서 심방세동의 징후로 주석을 단 박동 수를 지정합니다.

문서에 사용된 데이터셋을 다시 만들려면 zip 파일을 어딘가에 추출하고 코드의 디렉토리를 수정한 다음 데이터 디렉토리에서 데이터 탐색 스크립트를 실행하면 됩니다. 추가 보너스로 멋진 plots 도 얻을 수 있습니다. :)

#### Citation:

@article{deep_af_detection_2018,
title = "Automated detection of atrial fibrillation using long short-term memory network with RR interval signals",
journal = "Computers in Biology and Medicine",
year = "2018",
issn = "0010-4825",
doi = "https://doi.org/10.1016/j.compbiomed.2018.07.001",
author = "Oliver Faust and Alex Shenfield and Murtadha Kareem and Tan Ru San and Hamido Fujita and U. Rajendra Acharya"
}

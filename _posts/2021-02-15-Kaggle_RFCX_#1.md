---
title: "Kaggle Competition - Rainforest Connection Species Audio Detection #1"
date: 2021-02-15 08:26:28 -0400
categories: Kaggle
---
### Kaggle Competition - Rainforest Connection Species Audio Detection - Revision #01

<br>
<br>
<br>
<br>

### 0. Intro   

<br>
<br>
<p align="center">
  <img src="/assets/RFCX_01/pic_00.png">
</p>
<br>
<br>

* 이번 Post에서는 3달 전에 Kaggle에 올라온 Competition중 하나인, 'Rainforest Connection Species Audio Detection' Competition에 도전해 보도록 하겠습니다.
<br>
<br>

* 이 Competition의 Link는 여기를 참조하세요.
  
  [Rainforest Connection Species Audio Detection](https://www.kaggle.com/c/rfcx-species-audio-detection)

<br>
<br>

* 간단하게 설명하자면, 1분 길이의 Sound File이 제공되고 각 File에 대한 정보도 제공해 줍니다.

  File 정보에는 1분의 Sound File에서 어느 위치부터 어느 위치까지의 소리가 어떤 종류 동물의 소리인지 표시되어 있습니다.

<br>
<br>    
  
* 총 24가지 종류의 새와 개구리 소리가 구분되어 있으며, Sound Data Format은 FLAC File과 TFRecord File, 2가지 Format으로 제공됩니다.


* 앞으로 몇 번의 Post에 걸쳐 다양한 방법으로 이 Competition에 도전해 보도록 하겠습니다.

<br>
<br>
<br>
<br>
<br>
<br>

### 1. Look Through Data

* 우선 제공된 Dataset을 살펴보기로 하겠습니다.

<br>
<br>
<p align="center">
  <img src="/assets/RFCX_01/pic_01.png">
</p>
<br>
<br>

* 전체 Data Size가 57.23 GB이고, 3개의 Folder와 3개의 File들로 구성되어 있네요.

* Test Folder에는 최종적으로 우리가 Predict 해야할 Sound File들이 들어있습니다.
* 전부 몇개나 있는지 볼까요


```python
import librosa
import numpy as np
import csv
from tqdm.notebook import tqdm
import pickle
from skimage.transform import resize
import pandas as pd
import matplotlib.pyplot as plt

import tensorflow as tf

import librosa.display
import colorednoise as cn

from sklearn.model_selection import StratifiedKFold

from tensorflow.keras.models import Sequential
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout,GlobalAveragePooling2D,BatchNormalization,Conv2D,MaxPooling2D
from tensorflow.keras import optimizers
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

import glob
from sklearn.model_selection import train_test_split
import lightgbm as lgb
```


```python
SR = 48000
```


```python
TestFileNames = glob.glob( '../rfcx-species-audio-detection/test/*.flac' )
print( len( TestFileNames ) )
print( TestFileNames[0] )
```

    1992
    ../rfcx-species-audio-detection/test\000316da7.flac
    
    
<br>
<br>
<br>
<br>
<br>
<br>

* 총 1992개의 Sound File이 있고, 우리 Model로 이 소리가 무슨 새 혹은 개구리의 소리인지 맞추는 것이 이 Competition의 목표입니다.

   

* train Folder와 tfrecords Folder에는 모두 Train용 Data가 들어가 있는데, train Folder에는 FLAC Format의 Data File이 있고, tfrecords folder에는 TFRecord Format의 Data File이 들어가 있습니다.


* 익숙한 Format의 File들을 이용하면 됩니다.


* 저는 FLAC Format을 이용하도록 하겠습니다.

   

   

   

* 그리고 Train에 있어서 가장 중요한 정보를 가지고 있는 'train_tp.csv' File에 대해서 알아보도록 하겠습니다.   


```python
with open('../rfcx-species-audio-detection/train_tp.csv') as f:
    reader = csv.reader(f)
    data = list(reader)

print(len(data))
```

    1217
    


```python
train_tp = pd.read_csv("../rfcx-species-audio-detection/train_tp.csv")
train_tp.head()
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>recording_id</th>
      <th>species_id</th>
      <th>songtype_id</th>
      <th>t_min</th>
      <th>f_min</th>
      <th>t_max</th>
      <th>f_max</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>003bec244</td>
      <td>14</td>
      <td>1</td>
      <td>44.5440</td>
      <td>2531.250</td>
      <td>45.1307</td>
      <td>5531.25</td>
    </tr>
    <tr>
      <td>1</td>
      <td>006ab765f</td>
      <td>23</td>
      <td>1</td>
      <td>39.9615</td>
      <td>7235.160</td>
      <td>46.0452</td>
      <td>11283.40</td>
    </tr>
    <tr>
      <td>2</td>
      <td>007f87ba2</td>
      <td>12</td>
      <td>1</td>
      <td>39.1360</td>
      <td>562.500</td>
      <td>42.2720</td>
      <td>3281.25</td>
    </tr>
    <tr>
      <td>3</td>
      <td>0099c367b</td>
      <td>17</td>
      <td>4</td>
      <td>51.4206</td>
      <td>1464.260</td>
      <td>55.1996</td>
      <td>4565.04</td>
    </tr>
    <tr>
      <td>4</td>
      <td>009b760e6</td>
      <td>10</td>
      <td>1</td>
      <td>50.0854</td>
      <td>947.461</td>
      <td>52.5293</td>
      <td>10852.70</td>
    </tr>
  </tbody>
</table>
</div>


<br>
<br>
<br>
<br>

* 이 File에는 제목 한 줄을 제외한 총 1216개의 정보가 포함되어 있으며, 각 Row에는 위와 같이 6개의 Column으로 이루어진 값들이 포함되어 있습니다.   

* 'recording_id'는 FLAC File이름이며, 고유 값입니다. 

  'species_id'이 바로 Target값입니다.
  
  't_min'과 't_max'는 해당 FLAC File에서 'species_id'가 소리를 내는 구간을 나타내며,단위는 sec입니다.

   

   

   

   

   

* Train File 하나를 읽어볼까요?


* Python에서 Audio File을 다룰때는 Librosa라는 Package가 많이 쓰이고 있습니다.

  저도 이 Package를 이용하도록 하겠습니다.


```python
wav , sr = librosa.load("../rfcx-species-audio-detection/train/"+data[2][0]+".flac" , sr=None)
```


```python
import IPython.display as ipd
```


```python
ipd.Audio(wav, rate=sr)
```


* Train FLAC File들의 길이는 모두 1분으로 동일합니다. 다행이네요.   

   

   

   

* t_min과 t_max를 읽어서 우리가 구분해야할 소리가 나오는 부분만 살펴보도록 하겠습니다.   


```python
t_min = float(train_tp.iloc[1]['t_min']) * sr
t_max = float(train_tp.iloc[1]['t_max']) * sr

slice = wav[int(t_min) : int(t_max)]
```


```python
ipd.Audio(slice, rate=sr)
```


* Feature로 사용할 때는 각 FLAC File마다 해당 부분만 떼어서 사용해야 할 것 같습니다.   

   

   

   

   

   

   

   

* t_max와 t_min의 차이, 즉, 나중에 Train에 사용할 구간들의 길이에 대해서 확인해 보겠습니다.


* 이것을 하는 이유는 나중에 얼마만큼씩 잘라서 사용해야 좋을지 확인하기 위해서입니다.


```python
train_tp['dur'] = train_tp['t_max'] - train_tp['t_min']
```


```python
train_tp['dur'].describe()
```




    count    1216.000000
    mean        2.537119
    std         1.903589
    min         0.272000
    25%         1.093300
    50%         1.856000
    75%         3.344000
    max         7.923900
    Name: dur, dtype: float64




```python
plt.hist( train_tp['dur'] , bins = 50)
```




    (array([ 14.,   0., 114.,  90.,  71., 100.,  10.,   0.,  87.,  37.,  88.,
              9.,  53.,  49.,  54.,  19.,  11.,   0., 101.,   0.,  73.,  23.,
              5.,   1.,   0.,  50.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,
             17.,   0.,   0.,   0.,  50.,   0.,   0.,  40.,   0.,   0.,   0.,
              0.,   0.,   0.,   0.,   0.,  50.]),
     array([0.272   , 0.425038, 0.578076, 0.731114, 0.884152, 1.03719 ,
            1.190228, 1.343266, 1.496304, 1.649342, 1.80238 , 1.955418,
            2.108456, 2.261494, 2.414532, 2.56757 , 2.720608, 2.873646,
            3.026684, 3.179722, 3.33276 , 3.485798, 3.638836, 3.791874,
            3.944912, 4.09795 , 4.250988, 4.404026, 4.557064, 4.710102,
            4.86314 , 5.016178, 5.169216, 5.322254, 5.475292, 5.62833 ,
            5.781368, 5.934406, 6.087444, 6.240482, 6.39352 , 6.546558,
            6.699596, 6.852634, 7.005672, 7.15871 , 7.311748, 7.464786,
            7.617824, 7.770862, 7.9239  ]),
     <a list of 50 Patch objects>)




<br>
<br>
<p align="center">
  <img src="/assets/RFCX_01/output_60_1.png">
</p>
<br>
<br>


* 대부분이 4초보다 짧고, 좀 길면 8초까지 있네요.


* Train Data 전체를 Cover하려면 6초 정도가 가장 적당하겠네요.


<br>
<br>
<br>
<br>
<br>
<br>


### 2. Features of Sound & Audio Data

* 자, 이제 Data도 대충 살펴보았고, 이제 어떻게 Train할 건지 생각해 봐야 할 것 같습니다.


* Feature를 어떻게 Extracr할 것인가가 가장 중요할 것 같은데요, Sound Data는 대부분 **MFCC**와 **MEL Spectogram**을 Feature로 많이 사용하는 것 같더라구요.

* MFCC를 포함한 다양한 Sound Data의 Feature Extaction에 관련된 내용은 아래 Blog를 참조해 주시기 바랍니다.

  [Music Feature Extraction in Python](https://towardsdatascience.com/extract-features-of-music-75a3f9bc265d)

* 아래 Blog도 꽤 좋은 내용을 담고 있습니다.

  [음성인식 기초 이해하기](https://newsight.tistory.com/294)
  
  
<br>
<br>
<br>
<br>


### 3. MFCC & MEL Spectogram

* MFCC와 MEL Spec.을 이제 구해보도록 하겠습니다.   

* 이미 MFCC와 MEL Spec.을 구하기 위한 훌륭한 Package가 있습니다. 앞에서도 잠깐 언급했지만, **Librosa**라는 Package가 매우 훌륭합니다.


* 자세한 Doc.은 아래 Link를 참조해 주시기 바랍니다.
  
  #### Librosa
  - [https://librosa.org/doc/latest/index.html](https://librosa.org/doc/latest/index.html)
  - [Audio Feature 관련](https://librosa.org/doc/latest/feature.html)

* Feature extraction 항목에 보시면 Sound로 부터 다양한 Feature를 Extract할 수 있는 다양한 함수들이 준비되어 있습니다.
  - [Feature extraction](https://librosa.org/doc/latest/feature.html)

* 저는 여기서 MEL Spec.(melspectrogram)과 MFCC(mfcc)를 사용하도록 하겠습니다.


<br>
<br>
<br>
<br>

### 4. Feature Extraction

* Librosa의 함수를 이용하여 Feature Extraction을 해 보겠습니다.

   

   

* 전체적인 흐름은 아래와 같이, MEL Spec.과 MFCC의 평균값으로 각 Train Data의 Feature를 만들 것입니다.   

<br>
<br>
<p align="center">
  <img src="/assets/RFCX_01/pic_02.png">
</p>
<br>
<br>
   
  

   

   

* 우선 전체 Data에서 각 영역의 최대 / 최소 Frequency를 알아보도록 하죠. 이 값을 알아야 이후에 MEL Spec.과 MFCC를 구할 수 있습니다.   


<br>
<br>
<br>
<br>

```python
fmin = 24000
fmax = 0

for i in range(1, len(data)):
    if fmin > float(data[i][4]):
        fmin = float(data[i][4])
    if fmax < float(data[i][6]):
        fmax = float(data[i][6])

# Margin을 조금 줍니다.
fmin = int(fmin * 0.9)
fmax = int(fmax * 1.1)

print('Minimum frequency: ' + str(fmin) + ', maximum frequency: ' + str(fmax))
```

    Minimum frequency: 84, maximum frequency: 15056
    


```python
fft = 2048
hop = 512

WINDOWS_SIZE = 6
sr = 48000
length = WINDOWS_SIZE * sr

NUM_CLASS = 24
```

<br>
<br>
<br>

* 이제, MFCC와 MEL Spec.을 Extract해서 File로 저장해 놓도록 하겠습니다.


```python
TRAIN = pd.DataFrame()

print('Starting Feature Extracting...')


for i in tqdm( range(1, len(data))):    

    wav, sr = librosa.load('../rfcx-species-audio-detection/train/' + data[i][0] + '.flac', sr=None)
    
    t_min = float(data[i][3]) * sr
    t_max = float(data[i][5]) * sr
    
    # 시작과 끝의 중점을 중심으로 전체 길이가 6초로 일정하게 만든다.
    center = np.round((t_min + t_max) / 2)
    beginning = center - length / 2
    if beginning < 0:
        beginning = 0
    
    ending = beginning + length
    
    if ending > len(wav):
        ending = len(wav)
        beginning = ending - length
        
    slice = wav[int(beginning):int(ending)]
    
    # MEL Spec. Feature
    mel_spec = librosa.feature.melspectrogram(slice, n_fft=fft, hop_length=hop, sr=sr, fmin=fmin, fmax=fmax, power=1.5)
    melscaled = np.mean(mel_spec.T,axis=0)

    melscaled = list(melscaled.T)
    
    # MFCC Feature
    mfcc = librosa.feature.mfcc(slice , sr=sr , n_mfcc = 40)
    mfccsscaled = np.mean(mfcc.T,axis=0)
    mfccsscaled = list(mfccsscaled.T)
    
    tmp = melscaled + mfccsscaled
    tmp = pd.DataFrame(tmp).T
    
    TRAIN = pd.concat([TRAIN,tmp])
```

    Starting Feature Extracting...
    
<br>
<br>
<br>

* 이제 Feature에 맞는 Label도 따로 뽑아내겠습니다.
* 그래야 나중에 Train할 때 사용할 수 있으니깐요.
<br>
<br>
<br>

```python
Label = []

for i in tqdm( range(1, len(data))):
    Label.append( data[i][1] )
```


```python
Label
```




    ['14',
     '23',
     '12',
     '17',
     '10',
     '8',
     '0',
     '18',
     '15',
     '1',
     ...]

<br>
<br>
<br>
<br>
<br>
<br>

* 원하는 대로 잘 만들어 졌는지 한 번 볼까요?   


```python
TRAIN.head()
```
<br>
<br>
<br>

<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>0</th>
      <th>1</th>
      <th>2</th>
      <th>3</th>
      <th>4</th>
      <th>5</th>
      <th>6</th>
      <th>7</th>
      <th>8</th>
      <th>9</th>
      <th>...</th>
      <th>158</th>
      <th>159</th>
      <th>160</th>
      <th>161</th>
      <th>162</th>
      <th>163</th>
      <th>164</th>
      <th>165</th>
      <th>166</th>
      <th>167</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>0.034562</td>
      <td>0.011758</td>
      <td>0.008450</td>
      <td>0.023155</td>
      <td>0.021589</td>
      <td>0.005748</td>
      <td>0.003271</td>
      <td>0.003472</td>
      <td>0.002780</td>
      <td>0.002468</td>
      <td>...</td>
      <td>-0.916265</td>
      <td>-1.098225</td>
      <td>-2.676772</td>
      <td>-1.896342</td>
      <td>0.270768</td>
      <td>-2.226133</td>
      <td>-0.259369</td>
      <td>0.513921</td>
      <td>0.795100</td>
      <td>-0.277169</td>
    </tr>
    <tr>
      <td>0</td>
      <td>0.326220</td>
      <td>0.302937</td>
      <td>0.145184</td>
      <td>0.073245</td>
      <td>0.043459</td>
      <td>0.021372</td>
      <td>0.013206</td>
      <td>0.017474</td>
      <td>0.020128</td>
      <td>0.018785</td>
      <td>...</td>
      <td>-3.458029</td>
      <td>-2.858149</td>
      <td>-3.494677</td>
      <td>-1.924518</td>
      <td>-2.351852</td>
      <td>-1.965279</td>
      <td>-1.336849</td>
      <td>-2.033947</td>
      <td>-0.016453</td>
      <td>-0.177412</td>
    </tr>
    <tr>
      <td>0</td>
      <td>0.165520</td>
      <td>0.198534</td>
      <td>0.202551</td>
      <td>0.172053</td>
      <td>0.103592</td>
      <td>0.057968</td>
      <td>0.034055</td>
      <td>0.024015</td>
      <td>0.024043</td>
      <td>0.022022</td>
      <td>...</td>
      <td>-7.113028</td>
      <td>-1.479504</td>
      <td>2.198930</td>
      <td>2.616116</td>
      <td>3.768363</td>
      <td>-2.565371</td>
      <td>-8.847826</td>
      <td>-1.028994</td>
      <td>7.390729</td>
      <td>-0.466450</td>
    </tr>
    <tr>
      <td>0</td>
      <td>0.036173</td>
      <td>0.041161</td>
      <td>0.011874</td>
      <td>0.007293</td>
      <td>0.007921</td>
      <td>0.008111</td>
      <td>0.005517</td>
      <td>0.003888</td>
      <td>0.003161</td>
      <td>0.003231</td>
      <td>...</td>
      <td>0.038490</td>
      <td>-4.215221</td>
      <td>0.132852</td>
      <td>0.432737</td>
      <td>-1.416698</td>
      <td>-0.349015</td>
      <td>-2.514560</td>
      <td>-0.969835</td>
      <td>2.091542</td>
      <td>-2.319309</td>
    </tr>
    <tr>
      <td>0</td>
      <td>0.027391</td>
      <td>0.025106</td>
      <td>0.021964</td>
      <td>0.019909</td>
      <td>0.019068</td>
      <td>0.016489</td>
      <td>0.012955</td>
      <td>0.012114</td>
      <td>0.012478</td>
      <td>0.013255</td>
      <td>...</td>
      <td>-1.107634</td>
      <td>0.938184</td>
      <td>1.522288</td>
      <td>-0.468707</td>
      <td>-0.549454</td>
      <td>-1.000291</td>
      <td>2.265465</td>
      <td>0.910263</td>
      <td>-0.981540</td>
      <td>0.002835</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 168 columns</p>
</div>

<br>
<br>
<br>

```python
TRAIN.shape
```
    
    (1216, 168)

<br>
<br>
<br>

```python
TRAIN.to_csv("MFCC_MEL_Spec_Mean_Train_csv" , index=False)
```

* 이제 Train Data와 이에 해당하는 Label 모두 준비되었으니, LightLGB를 준비하도록 하겠습니다.   


<br>
<br>
<br>
<br>
<br>
<br>

### 5. Train

* 저는 LightGBM을 이용하도록 하겠습니다.   

* 제가 LightGBM을 선호하는 이유는 Tree Base라서 Train Data의 특별한 전처리도 필요없고, 대용량의 Tabular Data 처리에 최적화되어 있기 때문입니다.

* Train Data와 Validation Data를 6:4의 비율로 나누도록 하겠습니다.   


```python
X_train,X_test,y_train,y_test = train_test_split( TRAIN , Label , test_size=0.4 , random_state=27)
```
<br>
<br>
<br>

* LightGBM에 사용하기 위해 약간의 처리를 합니다.

```python
#Converting the dataset in proper LGB format
LGB_Train_Data = lgb.Dataset(X_train, label=y_train)

# Val
LGB_Test_Data = lgb.Dataset(X_test, label = y_test) 
```

<br>
<br>
<br>
   

* 가장 중요한 Hyper Parameter Setting을 하도록 하겠습니다.


* 아래 값들은 일단 기본적인 값들로 설정하였고, 나중에 Tunning하면 됩니다.


```python
params={}
params['learning_rate']=0.03
params['num_iterations']=10000
params['boosting_type']='gbdt' #GradientBoostingDecisionTree
params['objective']='multiclass' #Multi-class target feature
params['metric']='multi_logloss' #metric for multi-class
params['num_class']=24 #no.of unique values in the target class not inclusive of the end value
```

<br>
<br>
<br>
<br>
<br>
<br>   

* 자, 이제 모든 준비가 끝났습니다. Train을 시작해 보겠습니다.


```python
#training the model
Model = lgb.train(params , 
                  LGB_Train_Data , 
                  10000,
                  LGB_Test_Data , 
                  verbose_eval=10,
                  early_stopping_rounds = 500
                 )  #training the model on 100 epocs
```

    C:\Users\csyi\AppData\Local\Continuum\anaconda3\lib\site-packages\lightgbm\engine.py:148: UserWarning: Found `num_iterations` in params. Will use it instead of argument
      warnings.warn("Found `{}` in params. Will use it instead of argument".format(alias))
    

    Training until validation scores don't improve for 500 rounds
    [10]	valid_0's multi_logloss: 2.73311
    [20]	valid_0's multi_logloss: 2.50195
    [30]	valid_0's multi_logloss: 2.34626
    [40]	valid_0's multi_logloss: 2.23263
    [50]	valid_0's multi_logloss: 2.15191
    [60]	valid_0's multi_logloss: 2.08371
    [70]	valid_0's multi_logloss: 2.03414
    [80]	valid_0's multi_logloss: 1.99057
    [90]	valid_0's multi_logloss: 1.95019
    [100]	valid_0's multi_logloss: 1.91641
    [110]	valid_0's multi_logloss: 1.88726
    [120]	valid_0's multi_logloss: 1.86712
    [130]	valid_0's multi_logloss: 1.85241
    [140]	valid_0's multi_logloss: 1.84384
    [150]	valid_0's multi_logloss: 1.83451
    [160]	valid_0's multi_logloss: 1.82917
    [170]	valid_0's multi_logloss: 1.82655
    [180]	valid_0's multi_logloss: 1.82385
    [190]	valid_0's multi_logloss: 1.82632
    [200]	valid_0's multi_logloss: 1.8294
    [210]	valid_0's multi_logloss: 1.83588
    [220]	valid_0's multi_logloss: 1.84367
    [230]	valid_0's multi_logloss: 1.84795
    [240]	valid_0's multi_logloss: 1.85654
    [250]	valid_0's multi_logloss: 1.8652
    [260]	valid_0's multi_logloss: 1.87783
    [270]	valid_0's multi_logloss: 1.89209
    [280]	valid_0's multi_logloss: 1.90651
    [290]	valid_0's multi_logloss: 1.91928
    [300]	valid_0's multi_logloss: 1.92866
    [310]	valid_0's multi_logloss: 1.94303
    [320]	valid_0's multi_logloss: 1.95658
    [330]	valid_0's multi_logloss: 1.97317
    [340]	valid_0's multi_logloss: 1.98953
    [350]	valid_0's multi_logloss: 2.0086
    [360]	valid_0's multi_logloss: 2.02363
    [370]	valid_0's multi_logloss: 2.04138
    [380]	valid_0's multi_logloss: 2.05692
    [390]	valid_0's multi_logloss: 2.07291
    [400]	valid_0's multi_logloss: 2.09063
    [410]	valid_0's multi_logloss: 2.10799
    [420]	valid_0's multi_logloss: 2.12491
    [430]	valid_0's multi_logloss: 2.14385
    [440]	valid_0's multi_logloss: 2.16186
    [450]	valid_0's multi_logloss: 2.17973
    [460]	valid_0's multi_logloss: 2.19931
    [470]	valid_0's multi_logloss: 2.21659
    [480]	valid_0's multi_logloss: 2.23474
    [490]	valid_0's multi_logloss: 2.25269
    [500]	valid_0's multi_logloss: 2.2692
    [510]	valid_0's multi_logloss: 2.28067
    [520]	valid_0's multi_logloss: 2.29046
    [530]	valid_0's multi_logloss: 2.30034
    [540]	valid_0's multi_logloss: 2.30904
    [550]	valid_0's multi_logloss: 2.31594
    [560]	valid_0's multi_logloss: 2.32332
    [570]	valid_0's multi_logloss: 2.32974
    [580]	valid_0's multi_logloss: 2.33579
    [590]	valid_0's multi_logloss: 2.34151
    [600]	valid_0's multi_logloss: 2.34632
    [610]	valid_0's multi_logloss: 2.35019
    [620]	valid_0's multi_logloss: 2.35459
    [630]	valid_0's multi_logloss: 2.35936
    [640]	valid_0's multi_logloss: 2.36239
    [650]	valid_0's multi_logloss: 2.36519
    [660]	valid_0's multi_logloss: 2.36738
    [670]	valid_0's multi_logloss: 2.36999
    Early stopping, best iteration is:
    [179]	valid_0's multi_logloss: 1.82373
    

<br>
<br>
<br>
<br>
<br>
<br>

### 6. Predict   

* 학습 완료된 Model이 생겼으니, 이제 마지막으로 주최측에서 제공한 Test File들이 어떤 종류의 소리를 담고 있는지 Predict해서 제출하기만 하면 됩니다.   

* Predict해야 할 File들을 살펴보도록 하겠습니다.


* 'test' Folder안에 있고, 총 1992개의 Sound File이 있습니다.


* 좋습니다. 자, 이제 각각 하나씩 전부 Trained Model에 집어넣고 결과를 모아서 제출하면 됩니다.

* Test File들도 모두 길이가 1분씩입니다. 

* 우리가 Train시킨 Model은 6초 길이의 Sound File에서 MFCC & MEL Spec.을 전처리한 Data를 받아서 출력하는 기능을 합니다.

* 즉, Predict를 하려면, Test File에 대해서 동일한 전처리를 거친 Data를 Model의 입력으로 넣어줘야 합니다.

* 조금 귀찮지만, 그렇게 해줘야 합니다.   

<br>
<br>
<br>

* Test File들의 위치와 개수를 확인해 봅시다.   


```python
TestFileNames = glob.glob( '../rfcx-species-audio-detection/test/*.flac' )
print( len( TestFileNames ) )
print( TestFileNames[0] )
```

    1992
    ../rfcx-species-audio-detection/test\000316da7.flac
    

* 좋습니다. 우리가 기대하던 그대로 입니다.

<br>
<br>
<br>

* Submission File은 24개 각 Class의 확률을 적어주어야 합니다.

* 아래와 같이 Column Name을 구성해 주고, 해당 확률을 적어주면 됩니다.


```python
submission = pd.DataFrame( columns=['s0', 's1', 's2', 's3', 's4', 's5', 's6', 's7', 's8', 's9', 's10',
       's11', 's12', 's13', 's14', 's15', 's16', 's17', 's18', 's19', 's20',
       's21', 's22', 's23'] )
recording_id = []

for filename in tqdm(TestFileNames):
    
    pred = []
    
    id = filename.replace("../rfcx-species-audio-detection/test\\" , "")
    id = id.replace(".flac" , "")
    recording_id.append( id )
    
    wav, sr = librosa.load(filename, sr=None)
    
    # 6초 단위로 끊어서 변환후 전처리를 거쳐 Model에 입력시킵니다.
    segments = len(wav) / length
    segments = int(np.ceil(segments))    
   
    for i in range(0, segments):
        
        # Last segment going from the end
        if (i + 1) * length > len(wav):
            slice = wav[len(wav) - length:len(wav)]
        else:
            slice = wav[i * length:(i + 1) * length]
            
        mel_spec = librosa.feature.melspectrogram(slice, n_fft=fft, hop_length=hop, sr=sr, fmin=fmin, fmax=fmax, power=1.5)    
    
        mel_spec = librosa.feature.melspectrogram(slice, n_fft=fft, hop_length=hop, sr=sr, fmin=fmin, fmax=fmax, power=1.5)    
        melscaled = np.mean(mel_spec.T,axis=0)

        melscaled = list(melscaled.T)

        mfcc = librosa.feature.mfcc(slice , sr=sr , n_mfcc = 40)
        mfccsscaled = np.mean(mfcc.T,axis=0)
        mfccsscaled = list(mfccsscaled.T)

        tmp = melscaled + mfccsscaled
        tmp = np.array(tmp).reshape(1,-1)

        pred.append( Model.predict( tmp ) )
    

    pred = np.array( pred ).reshape(10,24)
    pred = list(np.max( pred , axis=0))    
    
    a_series = pd.Series(pred, index = submission.columns)
    submission = submission.append(a_series, ignore_index=True)
```

<br>
<br>
<br>
<br>
<br>
<br>   

* 완료 되었습니다. 이제 Column 순서 정리하고 File로 저장해서 제출해 보도록 하겠습니다.   


```python
submission.head()
```


<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>s0</th>
      <th>s1</th>
      <th>s2</th>
      <th>s3</th>
      <th>s4</th>
      <th>s5</th>
      <th>s6</th>
      <th>s7</th>
      <th>s8</th>
      <th>s9</th>
      <th>...</th>
      <th>s14</th>
      <th>s15</th>
      <th>s16</th>
      <th>s17</th>
      <th>s18</th>
      <th>s19</th>
      <th>s20</th>
      <th>s21</th>
      <th>s22</th>
      <th>s23</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>0.022533</td>
      <td>0.018128</td>
      <td>0.024431</td>
      <td>0.698671</td>
      <td>0.022352</td>
      <td>0.121563</td>
      <td>0.053104</td>
      <td>0.023634</td>
      <td>0.018857</td>
      <td>0.028449</td>
      <td>...</td>
      <td>0.030755</td>
      <td>0.027548</td>
      <td>0.023639</td>
      <td>0.036652</td>
      <td>0.182670</td>
      <td>0.062465</td>
      <td>0.702695</td>
      <td>0.026496</td>
      <td>0.247210</td>
      <td>0.121671</td>
    </tr>
    <tr>
      <td>1</td>
      <td>0.007562</td>
      <td>0.007608</td>
      <td>0.006683</td>
      <td>0.017434</td>
      <td>0.013696</td>
      <td>0.007921</td>
      <td>0.007938</td>
      <td>0.008398</td>
      <td>0.006666</td>
      <td>0.010170</td>
      <td>...</td>
      <td>0.018974</td>
      <td>0.010512</td>
      <td>0.967143</td>
      <td>0.298409</td>
      <td>0.012959</td>
      <td>0.007101</td>
      <td>0.025084</td>
      <td>0.006906</td>
      <td>0.043390</td>
      <td>0.019229</td>
    </tr>
    <tr>
      <td>2</td>
      <td>0.387472</td>
      <td>0.008364</td>
      <td>0.019871</td>
      <td>0.048506</td>
      <td>0.016856</td>
      <td>0.112532</td>
      <td>0.023723</td>
      <td>0.041035</td>
      <td>0.008675</td>
      <td>0.076355</td>
      <td>...</td>
      <td>0.011186</td>
      <td>0.012448</td>
      <td>0.010874</td>
      <td>0.037880</td>
      <td>0.014036</td>
      <td>0.126050</td>
      <td>0.319106</td>
      <td>0.010441</td>
      <td>0.376539</td>
      <td>0.023874</td>
    </tr>
    <tr>
      <td>3</td>
      <td>0.921778</td>
      <td>0.003868</td>
      <td>0.011993</td>
      <td>0.006110</td>
      <td>0.029487</td>
      <td>0.013983</td>
      <td>0.006285</td>
      <td>0.009679</td>
      <td>0.006292</td>
      <td>0.004410</td>
      <td>...</td>
      <td>0.027170</td>
      <td>0.004521</td>
      <td>0.004625</td>
      <td>0.052609</td>
      <td>0.063325</td>
      <td>0.007683</td>
      <td>0.003120</td>
      <td>0.011065</td>
      <td>0.004546</td>
      <td>0.057416</td>
    </tr>
    <tr>
      <td>4</td>
      <td>0.021822</td>
      <td>0.019401</td>
      <td>0.037472</td>
      <td>0.093516</td>
      <td>0.021219</td>
      <td>0.461508</td>
      <td>0.086660</td>
      <td>0.088066</td>
      <td>0.017920</td>
      <td>0.314157</td>
      <td>...</td>
      <td>0.154907</td>
      <td>0.038912</td>
      <td>0.022421</td>
      <td>0.189408</td>
      <td>0.260326</td>
      <td>0.129512</td>
      <td>0.326650</td>
      <td>0.020201</td>
      <td>0.085881</td>
      <td>0.052635</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 24 columns</p>
</div>

<br>
<br>
<br>

```python
submission['recording_id'] = recording_id
```


```python
submission = submission[ ['recording_id','s0', 's1', 's2', 's3', 's4', 's5', 's6', 's7', 's8', 's9', 's10',
       's11', 's12', 's13', 's14', 's15', 's16', 's17', 's18', 's19', 's20',
       's21', 's22', 's23'] ]
```


```python
submission.to_csv("submission_MEL_Spec_Mean.csv" , index=False)
```
<br>
<br>
<br>   

* 자, 이제 다 됐습니다. 제출해 보도록 하죠.    

<br>
<br>
<p align="center">
  <img src="/assets/RFCX_01/pic_03.ng">
</p>
<br>
<br>
   

* 점수는 **0.572**

  Baseline 점수로 적절한 것 같네요.
  
  이제부터 다른 방법으로 도전해 보도록 하겠습니다.

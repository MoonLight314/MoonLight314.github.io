---
title: "Hand Gesture Detection - Rev #01"
date: 2021-10-29 08:26:28 -0400
categories: Deep Learning
---
# Hand Gesture Detection - Rev #01

<br>
<br>
<br>
<br>

* RGB Camera만을 이용해서 Online Hand Gesture Detection을 해보려고 합니다.   

<br>

* Depth / IR 등의 다양한 Sensor들을 사용할 수는 환경이 되지 않아 우선 RGB Cam.으로 들어오는 손 모양을 이용해서 Gesture Detection Project를 진행하려고 합니다.   

<br>

## 0. Idea   

<br>

* 손을 Detecting하기 위해서 Google에서 제공하는 Mediapipe라는 Solution을 사용하기로 했습니다.   

<br>

* Mediapipe는 손 뿐만 아니라 다양한 목적으로 사용할 수 있으며, CPU만으로도 실시간으로 매우 훌륭한 성능을 보여줍니다.
  
  더 자세한 사항은 아래 Link를 참고하세요.
  
  https://google.github.io/mediapipe/   

<br>

* Mediapipe는 Image에서 미리 학습된 손 모양이 Detect되면 손가락을 포함한 손 전체의 중요 21개의 Point(Land Mark)에 대한 (x,y,z) 좌표를 Return해 줍니다.

<br>

* 미리 준비한 영상에서 이 63개의 Land Mark(21개 Land Mark X (x,y,z) 3개의 좌표)값들을 뽑아내고, 이 값들의 변화(RNN)를 학습하여 Hand Gesture Detection을 할 예정입니다.

<br>

<p align="center">
  <img src="/assets/Hand_Gesture_Detection_Rev_00/pic_00.png">
</p>

<br>
<br>
<br>
<br>

## 1. Prepare Dataset

<br>
<br>

### 1.1. Generating Dataset   

<br>

* 총 4가지의 Hand Gesture를 동영상으로 녹화한 Dataset을 준비했습니다.

<br>

* 1 Finger Click / 2 Finger Left / 2 Finger Right / Hand Shake 의 4가지 종류별로 100여개 정도의 동영상을 직접 촬영하였습니다.

<br>
<br>

### 1.2. Preprocessing Dataset   

<br>

* 만들어진 각각의 동영상은 모두 다른 Frame 길이를 가지고 있습니다.

<br>

* 먼저 OpenCV로 개별 동영상을 Load하여 Frame 단위로 읽어오고, 이 Frame들을 Mediapipe로 Hand Land Mark를 추출합니다.

<br>

* 결과적으로, 개별 동영상에서는 (Frame 수 X 63 )의 Data가 나오게 됩니다.

<br>

* 최종적으로 LSTM에 입력으로 넣어야 하기 때문에 Time Stamp,즉 Frame 수는 모두 같아야 합니다.

<br>

* 그래서 전체 동영상에서 가장 긴 길이의 Frame에 맞춰서 모든 동영상의 Frame을 맞추도록 하겠습니다.

<br>

* 가장 긴 동영상의 Frame 수는 115 Frame이었습니다. 아래 그림과 같이 앞뒤로 Zero Data를 Padding하여 전체 Data의 Frame길이를 115로 맞춥니다.

<br>

<p align="center">
  <img src="/assets/Hand_Gesture_Detection_Rev_00/pic_01.png">
</p>

<br>

* 이렇게 모두 같은 Frame 수로 맞춰진 후에 Hand Land Mark를 Numpy Data File로 저장합니다.   

<br>

* 최종적으로 이 File들을 Train Data로 활용하도록 하겠습니다.

<br>
<br>
<br>
<br>

## 2. Load Module & Prepare Train

<br>
<br>

* Tensorflow & LSTM 관련 Package를 사용합니다.   

<br>

```python
import numpy as np
import pandas as pd
import os
import tensorflow as tf

from sklearn.preprocessing import LabelEncoder 
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Bidirectional
from tensorflow.keras.callbacks import TensorBoard , ModelCheckpoint
```

<br>   

* Max Frame은 115입니다.   


```python
DATA_FRAMES_PER_DATA = 115
```

<br>
<br>

* (115 x 63)으로 미리 만들어진 Data File 정보를 가진 CSV File을 읽어옵니다.

<br>

```python
meta_data = pd.read_csv("Meta_Data_220105.csv")
```


```python
meta_data
```

<br>

<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>file_path</th>
      <th>action</th>
      <th>csv_file_path</th>
      <th>padded_file_name</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>./Train_Data\1_Finger_Click\Video_Test_10 (2).mp4</td>
      <td>1_Finger_Click</td>
      <td>./Train_Data\1_Finger_Click\Video_Test_10 (2).csv</td>
      <td>./Train_Data\1_Finger_Click\Video_Test_10 (2)_...</td>
    </tr>
    <tr>
      <th>1</th>
      <td>./Train_Data\1_Finger_Click\Video_Test_10 (3).mp4</td>
      <td>1_Finger_Click</td>
      <td>./Train_Data\1_Finger_Click\Video_Test_10 (3).csv</td>
      <td>./Train_Data\1_Finger_Click\Video_Test_10 (3)_...</td>
    </tr>
    <tr>
      <th>2</th>
      <td>./Train_Data\1_Finger_Click\Video_Test_10 (4).mp4</td>
      <td>1_Finger_Click</td>
      <td>./Train_Data\1_Finger_Click\Video_Test_10 (4).csv</td>
      <td>./Train_Data\1_Finger_Click\Video_Test_10 (4)_...</td>
    </tr>
    <tr>
      <th>3</th>
      <td>./Train_Data\1_Finger_Click\Video_Test_10.mp4</td>
      <td>1_Finger_Click</td>
      <td>./Train_Data\1_Finger_Click\Video_Test_10.csv</td>
      <td>./Train_Data\1_Finger_Click\Video_Test_10_padd...</td>
    </tr>
    <tr>
      <th>4</th>
      <td>./Train_Data\1_Finger_Click\Video_Test_11 (2).mp4</td>
      <td>1_Finger_Click</td>
      <td>./Train_Data\1_Finger_Click\Video_Test_11 (2).csv</td>
      <td>./Train_Data\1_Finger_Click\Video_Test_11 (2)_...</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>384</th>
      <td>./Train_Data\Shake_Hand\Video_Test_8.mp4</td>
      <td>Shake_Hand</td>
      <td>./Train_Data\Shake_Hand\Video_Test_8.csv</td>
      <td>./Train_Data\Shake_Hand\Video_Test_8_padded.csv</td>
    </tr>
    <tr>
      <th>385</th>
      <td>./Train_Data\Shake_Hand\Video_Test_9 (2).mp4</td>
      <td>Shake_Hand</td>
      <td>./Train_Data\Shake_Hand\Video_Test_9 (2).csv</td>
      <td>./Train_Data\Shake_Hand\Video_Test_9 (2)_padde...</td>
    </tr>
    <tr>
      <th>386</th>
      <td>./Train_Data\Shake_Hand\Video_Test_9 (3).mp4</td>
      <td>Shake_Hand</td>
      <td>./Train_Data\Shake_Hand\Video_Test_9 (3).csv</td>
      <td>./Train_Data\Shake_Hand\Video_Test_9 (3)_padde...</td>
    </tr>
    <tr>
      <th>387</th>
      <td>./Train_Data\Shake_Hand\Video_Test_9 (4).mp4</td>
      <td>Shake_Hand</td>
      <td>./Train_Data\Shake_Hand\Video_Test_9 (4).csv</td>
      <td>./Train_Data\Shake_Hand\Video_Test_9 (4)_padde...</td>
    </tr>
    <tr>
      <th>388</th>
      <td>./Train_Data\Shake_Hand\Video_Test_9.mp4</td>
      <td>Shake_Hand</td>
      <td>./Train_Data\Shake_Hand\Video_Test_9.csv</td>
      <td>./Train_Data\Shake_Hand\Video_Test_9_padded.csv</td>
    </tr>
  </tbody>
</table>
<p>389 rows × 4 columns</p>
</div>


<br>

```python
file_list = meta_data['padded_file_name'].tolist()
action = meta_data['action'].tolist()
```

<br>

```python
print(len(file_list) , len(action))
```

    389 389
    

* 4개의 Gesture Set의 총 개수는 389개입니다.   

<br>

* 389개 Hand Land Mark Feature 정보를 모두 읽어서 하나로 만듭니다.   

<br>

```python
train_data = []
target_action = []

for idx,file in enumerate( file_list ):
    d = pd.read_csv(file)
    d = np.array(d)
    train_data.append(d)
    target_action.append(action[idx])
```

<br>

```python
train_data = np.array(train_data)
```

<br>

* ( 전체 Train File 수  X  Frame 수  X Feature 수 ) = (389, 115, 63)


```python
train_data.shape
```

<br>

    (389, 115, 63)

<br>

* Target 값을 One-Hot으로 Encoding합니다.


```python
np.unique(target_action)
```




    array(['1_Finger_Click', '2_Fingers_Left', '2_Fingers_Right',
           'Shake_Hand'], dtype='<U15')




```python
le = LabelEncoder()
```


```python
le_action = le.fit(target_action)
```


```python
le_action = le.transform(target_action)
```


```python
y = tf.keras.utils.to_categorical(le_action, num_classes=4)
y
```




    array([[1., 0., 0., 0.],
           [1., 0., 0., 0.],
           [1., 0., 0., 0.],
           ...,
           [0., 0., 0., 1.],
           [0., 0., 0., 1.],
           [0., 0., 0., 1.]], dtype=float32)


<br>
   

* One-Hot으로 Encoding후에 어떻게 Mapping되었는지 확인하는 방법은 아래와 같습니다.   


```python
le.classes_
```

<br>


    array(['1_Finger_Click', '2_Fingers_Left', '2_Fingers_Right',
           'Shake_Hand'], dtype='<U15')

<br>

* Train / Test로 나눕니다.

<br>

* stratify Parameter는 설정한 값을 Train / Test Set에 동일한 비율로 나눠지도록 해줍니다.


```python
X_train, X_test, y_train, y_test = train_test_split(train_data, y, test_size=0.25 , stratify = y)
```

<br>

```python
print( X_train.shape )
print( X_test.shape)
```

    (291, 115, 63)
    (98, 115, 63)
    


```python
print( y_train.shape )
print( y_test.shape)
```

    (291, 4)
    (98, 4)
    
<br>
<br>
<br>

## 3. Define Model

<br>

* Model은 구조는 LSTM을 Base로 하는 구조로 만듭니다.

<br>

* 앞에서 하나의 Video File을 ( 115 , 63 ) 크기로 Input Data를 만들어두었고, 이를 그대로 LSTM Input으로 넣습니다.

<br>

* Model은 전체 Frame의 전후 관계를 학습하며, Dense Layer를 거쳐 최종적으로 각 동작으로 Mapping되도록 학습하는 과정을 거칩니다.

<br>
<br>

```python
def Define_Model_Rev_02():
    model = Sequential()
    model.add(Bidirectional(LSTM(128, return_sequences=True, activation='tanh', input_shape=(DATA_FRAMES_PER_DATA,63))))
    model.add(Bidirectional(LSTM(64, return_sequences=True, activation='tanh')))
    model.add(Bidirectional(LSTM(32, return_sequences=False, activation='tanh')))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(4, activation='softmax'))

    return model
```
<br>

* Tensorflow의 Bug인 것 같은데, LSTM의 Activation Function으로 Relu를 사용하면 Warning이 발생하고, Train이 잘 되지 않는 문제가 생기더군요.

<br>

* Relu와 유사한 Tanh을 사용하면 그런 문제는 없어집니다만, 빨리 문제가 수정이 되었으면 좋겠습니다.

<br>


```python
model = Define_Model_Rev_02()
```


```python
model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-3),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

```


```python
model.build( input_shape=(None, DATA_FRAMES_PER_DATA,63) )
```


```python
model.summary()
```

    Model: "sequential"
    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    bidirectional (Bidirectional (None, 115, 256)          196608    
    _________________________________________________________________
    bidirectional_1 (Bidirection (None, 115, 128)          164352    
    _________________________________________________________________
    bidirectional_2 (Bidirection (None, 64)                41216     
    _________________________________________________________________
    dense (Dense)                (None, 64)                4160      
    _________________________________________________________________
    dense_1 (Dense)              (None, 32)                2080      
    _________________________________________________________________
    dense_2 (Dense)              (None, 4)                 132       
    =================================================================
    Total params: 408,548
    Trainable params: 408,548
    Non-trainable params: 0
    _________________________________________________________________
    

<br>
   

* Check Point & Tensorboard 관련 설정을 합니다.   


```python
log_dir = os.path.join('Logs')
CHECKPOINT_PATH = os.path.join('CheckPoints_Hand_Gesture_Ver_00')
tb_callback = TensorBoard(log_dir=log_dir)

cp = ModelCheckpoint(filepath=CHECKPOINT_PATH, 
                     monitor='val_accuracy',                     
                     save_best_only = True,
                     verbose = 1)
```

<br>   

* 순조롭게 Training이 진행이 되는군요.   


```python
model.fit(  X_train, y_train, 
            validation_data=(X_test , y_test),
            epochs=2000, 
            callbacks=[cp , tb_callback]
         )
```

    Epoch 1/2000
    10/10 [==============================] - 7s 181ms/step - loss: 1.3401 - accuracy: 0.4124 - val_loss: 1.2626 - val_accuracy: 0.5000
    
    Epoch 00001: val_accuracy improved from -inf to 0.50000, saving model to CheckPoints_Hand_Gesture_Ver_00
    

    WARNING:absl:Found untraced functions such as lstm_cell_1_layer_call_and_return_conditional_losses, lstm_cell_1_layer_call_fn, lstm_cell_2_layer_call_and_return_conditional_losses, lstm_cell_2_layer_call_fn, lstm_cell_4_layer_call_and_return_conditional_losses while saving (showing 5 of 30). These functions will not be directly callable after loading.
    

    INFO:tensorflow:Assets written to: CheckPoints_Hand_Gesture_Ver_00\assets
    

    INFO:tensorflow:Assets written to: CheckPoints_Hand_Gesture_Ver_00\assets
    

    Epoch 2/2000
    10/10 [==============================] - 0s 42ms/step - loss: 1.1994 - accuracy: 0.4845 - val_loss: 1.2533 - val_accuracy: 0.4082
    
    Epoch 00002: val_accuracy did not improve from 0.50000
    Epoch 3/2000
    10/10 [==============================] - 0s 42ms/step - loss: 1.0992 - accuracy: 0.5258 - val_loss: 1.1289 - val_accuracy: 0.4388
    
    Epoch 00003: val_accuracy did not improve from 0.50000
    Epoch 4/2000
    10/10 [==============================] - 0s 41ms/step - loss: 0.9655 - accuracy: 0.6151 - val_loss: 0.9343 - val_accuracy: 0.5102
    
    Epoch 00004: val_accuracy improved from 0.50000 to 0.51020, saving model to CheckPoints_Hand_Gesture_Ver_00
    

    WARNING:absl:Found untraced functions such as lstm_cell_1_layer_call_and_return_conditional_losses, lstm_cell_1_layer_call_fn, lstm_cell_2_layer_call_and_return_conditional_losses, lstm_cell_2_layer_call_fn, lstm_cell_4_layer_call_and_return_conditional_losses while saving (showing 5 of 30). These functions will not be directly callable after loading.
    

    INFO:tensorflow:Assets written to: CheckPoints_Hand_Gesture_Ver_00\assets
    

    INFO:tensorflow:Assets written to: CheckPoints_Hand_Gesture_Ver_00\assets
    

    Epoch 5/2000
    10/10 [==============================] - 0s 42ms/step - loss: 0.8116 - accuracy: 0.7251 - val_loss: 0.8239 - val_accuracy: 0.6122
    
    Epoch 00005: val_accuracy improved from 0.51020 to 0.61224, saving model to CheckPoints_Hand_Gesture_Ver_00
    

    WARNING:absl:Found untraced functions such as lstm_cell_1_layer_call_and_return_conditional_losses, lstm_cell_1_layer_call_fn, lstm_cell_2_layer_call_and_return_conditional_losses, lstm_cell_2_layer_call_fn, lstm_cell_4_layer_call_and_return_conditional_losses while saving (showing 5 of 30). These functions will not be directly callable after loading.
    

    INFO:tensorflow:Assets written to: CheckPoints_Hand_Gesture_Ver_00\assets
    

    INFO:tensorflow:Assets written to: CheckPoints_Hand_Gesture_Ver_00\assets
    

    Epoch 6/2000
    10/10 [==============================] - 0s 43ms/step - loss: 0.6971 - accuracy: 0.7388 - val_loss: 0.5423 - val_accuracy: 0.8163
    
    Epoch 00006: val_accuracy improved from 0.61224 to 0.81633, saving model to CheckPoints_Hand_Gesture_Ver_00
    

    WARNING:absl:Found untraced functions such as lstm_cell_1_layer_call_and_return_conditional_losses, lstm_cell_1_layer_call_fn, lstm_cell_2_layer_call_and_return_conditional_losses, lstm_cell_2_layer_call_fn, lstm_cell_4_layer_call_and_return_conditional_losses while saving (showing 5 of 30). These functions will not be directly callable after loading.
    

    INFO:tensorflow:Assets written to: CheckPoints_Hand_Gesture_Ver_00\assets
    

    INFO:tensorflow:Assets written to: CheckPoints_Hand_Gesture_Ver_00\assets
    

    Epoch 7/2000
    10/10 [==============================] - 0s 42ms/step - loss: 0.6130 - accuracy: 0.7801 - val_loss: 0.4424 - val_accuracy: 0.8571
    
    Epoch 00007: val_accuracy improved from 0.81633 to 0.85714, saving model to CheckPoints_Hand_Gesture_Ver_00
    

    WARNING:absl:Found untraced functions such as lstm_cell_1_layer_call_and_return_conditional_losses, lstm_cell_1_layer_call_fn, lstm_cell_2_layer_call_and_return_conditional_losses, lstm_cell_2_layer_call_fn, lstm_cell_4_layer_call_and_return_conditional_losses while saving (showing 5 of 30). These functions will not be directly callable after loading.
    

    INFO:tensorflow:Assets written to: CheckPoints_Hand_Gesture_Ver_00\assets
    

    INFO:tensorflow:Assets written to: CheckPoints_Hand_Gesture_Ver_00\assets
    

    Epoch 8/2000
    10/10 [==============================] - 0s 42ms/step - loss: 0.4498 - accuracy: 0.8763 - val_loss: 0.3313 - val_accuracy: 0.8980
    
    Epoch 00008: val_accuracy improved from 0.85714 to 0.89796, saving model to CheckPoints_Hand_Gesture_Ver_00
    

    WARNING:absl:Found untraced functions such as lstm_cell_1_layer_call_and_return_conditional_losses, lstm_cell_1_layer_call_fn, lstm_cell_2_layer_call_and_return_conditional_losses, lstm_cell_2_layer_call_fn, lstm_cell_4_layer_call_and_return_conditional_losses while saving (showing 5 of 30). These functions will not be directly callable after loading.
    

    INFO:tensorflow:Assets written to: CheckPoints_Hand_Gesture_Ver_00\assets
    

    INFO:tensorflow:Assets written to: CheckPoints_Hand_Gesture_Ver_00\assets
    

    Epoch 9/2000
    10/10 [==============================] - 0s 42ms/step - loss: 0.6163 - accuracy: 0.7869 - val_loss: 0.3195 - val_accuracy: 0.9082
    
    Epoch 00009: val_accuracy improved from 0.89796 to 0.90816, saving model to CheckPoints_Hand_Gesture_Ver_00
    

    WARNING:absl:Found untraced functions such as lstm_cell_1_layer_call_and_return_conditional_losses, lstm_cell_1_layer_call_fn, lstm_cell_2_layer_call_and_return_conditional_losses, lstm_cell_2_layer_call_fn, lstm_cell_4_layer_call_and_return_conditional_losses while saving (showing 5 of 30). These functions will not be directly callable after loading.
    

    INFO:tensorflow:Assets written to: CheckPoints_Hand_Gesture_Ver_00\assets
    

    INFO:tensorflow:Assets written to: CheckPoints_Hand_Gesture_Ver_00\assets
    

    Epoch 10/2000
    10/10 [==============================] - 0s 42ms/step - loss: 0.4396 - accuracy: 0.8419 - val_loss: 0.3986 - val_accuracy: 0.8673
    
    Epoch 00010: val_accuracy did not improve from 0.90816
    Epoch 11/2000
    10/10 [==============================] - 0s 42ms/step - loss: 0.4750 - accuracy: 0.8351 - val_loss: 0.4060 - val_accuracy: 0.8571
    
    Epoch 00011: val_accuracy did not improve from 0.90816
    Epoch 12/2000
    10/10 [==============================] - 0s 42ms/step - loss: 0.3731 - accuracy: 0.8763 - val_loss: 0.3412 - val_accuracy: 0.8776
    
    Epoch 00012: val_accuracy did not improve from 0.90816
    Epoch 13/2000
    10/10 [==============================] - 0s 42ms/step - loss: 0.3067 - accuracy: 0.9175 - val_loss: 0.3185 - val_accuracy: 0.8878
    
    Epoch 00013: val_accuracy did not improve from 0.90816
    Epoch 14/2000
    10/10 [==============================] - 0s 42ms/step - loss: 0.3288 - accuracy: 0.8900 - val_loss: 0.2752 - val_accuracy: 0.9082
    
    Epoch 00014: val_accuracy did not improve from 0.90816
    Epoch 15/2000
    10/10 [==============================] - 0s 42ms/step - loss: 0.2713 - accuracy: 0.9038 - val_loss: 0.3092 - val_accuracy: 0.9082
    
    Epoch 00015: val_accuracy did not improve from 0.90816
    Epoch 16/2000
    10/10 [==============================] - 0s 42ms/step - loss: 0.2607 - accuracy: 0.9244 - val_loss: 0.2166 - val_accuracy: 0.9388
    
    Epoch 00016: val_accuracy improved from 0.90816 to 0.93878, saving model to CheckPoints_Hand_Gesture_Ver_00
    

    WARNING:absl:Found untraced functions such as lstm_cell_1_layer_call_and_return_conditional_losses, lstm_cell_1_layer_call_fn, lstm_cell_2_layer_call_and_return_conditional_losses, lstm_cell_2_layer_call_fn, lstm_cell_4_layer_call_and_return_conditional_losses while saving (showing 5 of 30). These functions will not be directly callable after loading.
    

    INFO:tensorflow:Assets written to: CheckPoints_Hand_Gesture_Ver_00\assets
    

    INFO:tensorflow:Assets written to: CheckPoints_Hand_Gesture_Ver_00\assets
    

    Epoch 17/2000
    10/10 [==============================] - 0s 43ms/step - loss: 0.2057 - accuracy: 0.9278 - val_loss: 0.2626 - val_accuracy: 0.9184
    
    Epoch 00017: val_accuracy did not improve from 0.93878
    Epoch 18/2000
    10/10 [==============================] - 0s 42ms/step - loss: 0.2150 - accuracy: 0.9210 - val_loss: 0.2156 - val_accuracy: 0.9184
    
    Epoch 00018: val_accuracy did not improve from 0.93878
    Epoch 19/2000
    10/10 [==============================] - 0s 43ms/step - loss: 0.2423 - accuracy: 0.9278 - val_loss: 0.2134 - val_accuracy: 0.9388
    
    Epoch 00019: val_accuracy did not improve from 0.93878
    Epoch 20/2000
    10/10 [==============================] - 0s 39ms/step - loss: 0.2297 - accuracy: 0.9278 - val_loss: 0.1725 - val_accuracy: 0.9592
    
    Epoch 00020: val_accuracy improved from 0.93878 to 0.95918, saving model to CheckPoints_Hand_Gesture_Ver_00
    

    WARNING:absl:Found untraced functions such as lstm_cell_1_layer_call_and_return_conditional_losses, lstm_cell_1_layer_call_fn, lstm_cell_2_layer_call_and_return_conditional_losses, lstm_cell_2_layer_call_fn, lstm_cell_4_layer_call_and_return_conditional_losses while saving (showing 5 of 30). These functions will not be directly callable after loading.
    

    INFO:tensorflow:Assets written to: CheckPoints_Hand_Gesture_Ver_00\assets
    

    INFO:tensorflow:Assets written to: CheckPoints_Hand_Gesture_Ver_00\assets
    

    Epoch 21/2000
    10/10 [==============================] - 0s 43ms/step - loss: 0.1405 - accuracy: 0.9656 - val_loss: 0.1253 - val_accuracy: 0.9694
    
    Epoch 00021: val_accuracy improved from 0.95918 to 0.96939, saving model to CheckPoints_Hand_Gesture_Ver_00
    

    WARNING:absl:Found untraced functions such as lstm_cell_1_layer_call_and_return_conditional_losses, lstm_cell_1_layer_call_fn, lstm_cell_2_layer_call_and_return_conditional_losses, lstm_cell_2_layer_call_fn, lstm_cell_4_layer_call_and_return_conditional_losses while saving (showing 5 of 30). These functions will not be directly callable after loading.
    

    INFO:tensorflow:Assets written to: CheckPoints_Hand_Gesture_Ver_00\assets
    

    INFO:tensorflow:Assets written to: CheckPoints_Hand_Gesture_Ver_00\assets
    

    Epoch 22/2000
    10/10 [==============================] - 0s 42ms/step - loss: 0.1602 - accuracy: 0.9416 - val_loss: 0.1779 - val_accuracy: 0.9388
    
    Epoch 00022: val_accuracy did not improve from 0.96939
    Epoch 23/2000
    10/10 [==============================] - 0s 44ms/step - loss: 0.1561 - accuracy: 0.9450 - val_loss: 0.0920 - val_accuracy: 0.9796
    
    Epoch 00023: val_accuracy improved from 0.96939 to 0.97959, saving model to CheckPoints_Hand_Gesture_Ver_00
    

    WARNING:absl:Found untraced functions such as lstm_cell_1_layer_call_and_return_conditional_losses, lstm_cell_1_layer_call_fn, lstm_cell_2_layer_call_and_return_conditional_losses, lstm_cell_2_layer_call_fn, lstm_cell_4_layer_call_and_return_conditional_losses while saving (showing 5 of 30). These functions will not be directly callable after loading.
    

    INFO:tensorflow:Assets written to: CheckPoints_Hand_Gesture_Ver_00\assets
    

    INFO:tensorflow:Assets written to: CheckPoints_Hand_Gesture_Ver_00\assets
    

    Epoch 24/2000
    10/10 [==============================] - 0s 42ms/step - loss: 0.1114 - accuracy: 0.9691 - val_loss: 0.1326 - val_accuracy: 0.9490
    
    Epoch 00024: val_accuracy did not improve from 0.97959
    Epoch 25/2000
    10/10 [==============================] - 0s 42ms/step - loss: 0.1194 - accuracy: 0.9553 - val_loss: 0.1855 - val_accuracy: 0.9490
    
    Epoch 00025: val_accuracy did not improve from 0.97959
    Epoch 26/2000
    10/10 [==============================] - 0s 42ms/step - loss: 0.5457 - accuracy: 0.8591 - val_loss: 0.6200 - val_accuracy: 0.7959
    
    Epoch 00026: val_accuracy did not improve from 0.97959
    Epoch 27/2000
    10/10 [==============================] - 0s 40ms/step - loss: 0.3533 - accuracy: 0.9003 - val_loss: 0.6065 - val_accuracy: 0.7449
    
    Epoch 00027: val_accuracy did not improve from 0.97959
    Epoch 28/2000
    10/10 [==============================] - 0s 41ms/step - loss: 0.2984 - accuracy: 0.8832 - val_loss: 0.3315 - val_accuracy: 0.8878
    
    Epoch 00028: val_accuracy did not improve from 0.97959
    Epoch 29/2000
    10/10 [==============================] - 0s 41ms/step - loss: 0.2879 - accuracy: 0.9175 - val_loss: 0.4583 - val_accuracy: 0.8469
    
    Epoch 00029: val_accuracy did not improve from 0.97959
    Epoch 30/2000
    10/10 [==============================] - 0s 42ms/step - loss: 0.2127 - accuracy: 0.9313 - val_loss: 0.1830 - val_accuracy: 0.9490
    
    Epoch 00030: val_accuracy did not improve from 0.97959
    Epoch 31/2000
    10/10 [==============================] - 0s 42ms/step - loss: 0.1398 - accuracy: 0.9622 - val_loss: 0.1254 - val_accuracy: 0.9694
    
    Epoch 00031: val_accuracy did not improve from 0.97959
    Epoch 32/2000
    10/10 [==============================] - 0s 41ms/step - loss: 0.0754 - accuracy: 0.9759 - val_loss: 0.1265 - val_accuracy: 0.9490
    
    Epoch 00032: val_accuracy did not improve from 0.97959
    Epoch 33/2000
    10/10 [==============================] - 0s 42ms/step - loss: 0.1338 - accuracy: 0.9622 - val_loss: 0.1827 - val_accuracy: 0.9388
    
    Epoch 00033: val_accuracy did not improve from 0.97959
    Epoch 34/2000
    10/10 [==============================] - 0s 41ms/step - loss: 0.1562 - accuracy: 0.9588 - val_loss: 0.1647 - val_accuracy: 0.9490
    
    Epoch 00034: val_accuracy did not improve from 0.97959
    Epoch 35/2000
    10/10 [==============================] - 0s 39ms/step - loss: 0.1652 - accuracy: 0.9519 - val_loss: 0.1364 - val_accuracy: 0.9490
    
    Epoch 00035: val_accuracy did not improve from 0.97959
    Epoch 36/2000
    10/10 [==============================] - 0s 42ms/step - loss: 0.1466 - accuracy: 0.9553 - val_loss: 0.2282 - val_accuracy: 0.9388
    
    Epoch 00036: val_accuracy did not improve from 0.97959
    Epoch 37/2000
    10/10 [==============================] - 0s 42ms/step - loss: 0.1387 - accuracy: 0.9725 - val_loss: 0.1778 - val_accuracy: 0.9592
    
    Epoch 00037: val_accuracy did not improve from 0.97959
    Epoch 38/2000
    10/10 [==============================] - 0s 42ms/step - loss: 0.0971 - accuracy: 0.9725 - val_loss: 0.3873 - val_accuracy: 0.9082
    
    Epoch 00038: val_accuracy did not improve from 0.97959
    Epoch 39/2000
    10/10 [==============================] - 0s 44ms/step - loss: 0.1060 - accuracy: 0.9691 - val_loss: 0.0948 - val_accuracy: 0.9796
    
    Epoch 00039: val_accuracy did not improve from 0.97959
    Epoch 40/2000
    10/10 [==============================] - 0s 42ms/step - loss: 0.0561 - accuracy: 0.9828 - val_loss: 0.0627 - val_accuracy: 0.9796
    
    Epoch 00040: val_accuracy did not improve from 0.97959
    Epoch 41/2000
    10/10 [==============================] - 0s 41ms/step - loss: 0.0489 - accuracy: 0.9897 - val_loss: 0.1453 - val_accuracy: 0.9490
    
    Epoch 00041: val_accuracy did not improve from 0.97959
    Epoch 42/2000
    10/10 [==============================] - 0s 42ms/step - loss: 0.0452 - accuracy: 0.9897 - val_loss: 0.0613 - val_accuracy: 0.9796
    
    Epoch 00042: val_accuracy did not improve from 0.97959
    Epoch 43/2000
    10/10 [==============================] - 0s 41ms/step - loss: 0.0339 - accuracy: 0.9897 - val_loss: 0.0316 - val_accuracy: 0.9898
    
    Epoch 00043: val_accuracy improved from 0.97959 to 0.98980, saving model to CheckPoints_Hand_Gesture_Ver_00
    

    WARNING:absl:Found untraced functions such as lstm_cell_1_layer_call_and_return_conditional_losses, lstm_cell_1_layer_call_fn, lstm_cell_2_layer_call_and_return_conditional_losses, lstm_cell_2_layer_call_fn, lstm_cell_4_layer_call_and_return_conditional_losses while saving (showing 5 of 30). These functions will not be directly callable after loading.
    

    INFO:tensorflow:Assets written to: CheckPoints_Hand_Gesture_Ver_00\assets
    

    INFO:tensorflow:Assets written to: CheckPoints_Hand_Gesture_Ver_00\assets
    

    Epoch 44/2000
    10/10 [==============================] - 0s 42ms/step - loss: 0.0228 - accuracy: 0.9966 - val_loss: 0.0231 - val_accuracy: 1.0000
    
    Epoch 00044: val_accuracy improved from 0.98980 to 1.00000, saving model to CheckPoints_Hand_Gesture_Ver_00
    

    WARNING:absl:Found untraced functions such as lstm_cell_1_layer_call_and_return_conditional_losses, lstm_cell_1_layer_call_fn, lstm_cell_2_layer_call_and_return_conditional_losses, lstm_cell_2_layer_call_fn, lstm_cell_4_layer_call_and_return_conditional_losses while saving (showing 5 of 30). These functions will not be directly callable after loading.
    

    INFO:tensorflow:Assets written to: CheckPoints_Hand_Gesture_Ver_00\assets
    

    INFO:tensorflow:Assets written to: CheckPoints_Hand_Gesture_Ver_00\assets
    

    Epoch 45/2000
    10/10 [==============================] - 0s 43ms/step - loss: 0.0120 - accuracy: 1.0000 - val_loss: 0.0179 - val_accuracy: 1.0000
    
    Epoch 00045: val_accuracy did not improve from 1.00000
    Epoch 46/2000
    10/10 [==============================] - 0s 42ms/step - loss: 0.0105 - accuracy: 0.9966 - val_loss: 0.0130 - val_accuracy: 1.0000
    
    Epoch 00046: val_accuracy did not improve from 1.00000
    Epoch 47/2000
    10/10 [==============================] - 0s 41ms/step - loss: 0.0092 - accuracy: 1.0000 - val_loss: 0.0118 - val_accuracy: 1.0000
    
    Epoch 00047: val_accuracy did not improve from 1.00000
    Epoch 48/2000
    10/10 [==============================] - 0s 42ms/step - loss: 0.0062 - accuracy: 1.0000 - val_loss: 0.0086 - val_accuracy: 1.0000
    
    Epoch 00048: val_accuracy did not improve from 1.00000
    Epoch 49/2000
    10/10 [==============================] - 0s 42ms/step - loss: 0.0047 - accuracy: 1.0000 - val_loss: 0.0068 - val_accuracy: 1.0000
    
    Epoch 00049: val_accuracy did not improve from 1.00000
    Epoch 50/2000
    10/10 [==============================] - 0s 41ms/step - loss: 0.0037 - accuracy: 1.0000 - val_loss: 0.0056 - val_accuracy: 1.0000
    
    Epoch 00050: val_accuracy did not improve from 1.00000
    Epoch 51/2000
    10/10 [==============================] - 0s 39ms/step - loss: 0.0031 - accuracy: 1.0000 - val_loss: 0.0045 - val_accuracy: 1.0000
    
    Epoch 00051: val_accuracy did not improve from 1.00000
    Epoch 52/2000
    10/10 [==============================] - 0s 42ms/step - loss: 0.0026 - accuracy: 1.0000 - val_loss: 0.0038 - val_accuracy: 1.0000
    
    Epoch 00052: val_accuracy did not improve from 1.00000
    Epoch 53/2000
    10/10 [==============================] - 0s 43ms/step - loss: 0.0022 - accuracy: 1.0000 - val_loss: 0.0033 - val_accuracy: 1.0000
    
    Epoch 00053: val_accuracy did not improve from 1.00000
    Epoch 54/2000
    10/10 [==============================] - 0s 40ms/step - loss: 0.0019 - accuracy: 1.0000 - val_loss: 0.0029 - val_accuracy: 1.0000
    
    Epoch 00054: val_accuracy did not improve from 1.00000
    Epoch 55/2000
    10/10 [==============================] - 0s 43ms/step - loss: 0.0017 - accuracy: 1.0000 - val_loss: 0.0026 - val_accuracy: 1.0000
    
    Epoch 00055: val_accuracy did not improve from 1.00000
    Epoch 56/2000
    10/10 [==============================] - 0s 45ms/step - loss: 0.0015 - accuracy: 1.0000 - val_loss: 0.0023 - val_accuracy: 1.0000
    
    Epoch 00056: val_accuracy did not improve from 1.00000
    Epoch 57/2000
    10/10 [==============================] - 0s 44ms/step - loss: 0.0014 - accuracy: 1.0000 - val_loss: 0.0020 - val_accuracy: 1.0000
    
    Epoch 00057: val_accuracy did not improve from 1.00000
    Epoch 58/2000
    10/10 [==============================] - 0s 43ms/step - loss: 0.0012 - accuracy: 1.0000 - val_loss: 0.0019 - val_accuracy: 1.0000
    
    Epoch 00058: val_accuracy did not improve from 1.00000
    Epoch 59/2000
    10/10 [==============================] - 0s 40ms/step - loss: 0.0011 - accuracy: 1.0000 - val_loss: 0.0017 - val_accuracy: 1.0000
    
    Epoch 00059: val_accuracy did not improve from 1.00000
    Epoch 60/2000
    10/10 [==============================] - 0s 40ms/step - loss: 0.0010 - accuracy: 1.0000 - val_loss: 0.0016 - val_accuracy: 1.0000
    
    Epoch 00060: val_accuracy did not improve from 1.00000
    Epoch 61/2000
    10/10 [==============================] - 0s 43ms/step - loss: 9.6202e-04 - accuracy: 1.0000 - val_loss: 0.0015 - val_accuracy: 1.0000
    
    Epoch 00061: val_accuracy did not improve from 1.00000
    Epoch 62/2000
    10/10 [==============================] - 0s 41ms/step - loss: 8.9064e-04 - accuracy: 1.0000 - val_loss: 0.0013 - val_accuracy: 1.0000
    
    Epoch 00062: val_accuracy did not improve from 1.00000
    Epoch 63/2000
    10/10 [==============================] - 0s 41ms/step - loss: 8.2291e-04 - accuracy: 1.0000 - val_loss: 0.0013 - val_accuracy: 1.0000
    
    Epoch 00063: val_accuracy did not improve from 1.00000
    Epoch 64/2000
    10/10 [==============================] - 0s 42ms/step - loss: 7.7714e-04 - accuracy: 1.0000 - val_loss: 0.0011 - val_accuracy: 1.0000
    
    Epoch 00064: val_accuracy did not improve from 1.00000
    Epoch 65/2000
    10/10 [==============================] - 0s 43ms/step - loss: 7.1956e-04 - accuracy: 1.0000 - val_loss: 0.0011 - val_accuracy: 1.0000
    
    Epoch 00065: val_accuracy did not improve from 1.00000
    Epoch 66/2000
    10/10 [==============================] - 0s 43ms/step - loss: 6.7171e-04 - accuracy: 1.0000 - val_loss: 9.9315e-04 - val_accuracy: 1.0000
    
    Epoch 00066: val_accuracy did not improve from 1.00000
    Epoch 67/2000
    10/10 [==============================] - 0s 42ms/step - loss: 6.3440e-04 - accuracy: 1.0000 - val_loss: 9.2510e-04 - val_accuracy: 1.0000
    
    Epoch 00067: val_accuracy did not improve from 1.00000
    Epoch 68/2000
    10/10 [==============================] - 0s 42ms/step - loss: 5.9452e-04 - accuracy: 1.0000 - val_loss: 8.8794e-04 - val_accuracy: 1.0000
    
    Epoch 00068: val_accuracy did not improve from 1.00000
    Epoch 69/2000
    10/10 [==============================] - 0s 43ms/step - loss: 5.6366e-04 - accuracy: 1.0000 - val_loss: 8.2888e-04 - val_accuracy: 1.0000
    
    Epoch 00069: val_accuracy did not improve from 1.00000
    Epoch 70/2000
    10/10 [==============================] - 0s 41ms/step - loss: 5.3337e-04 - accuracy: 1.0000 - val_loss: 7.8085e-04 - val_accuracy: 1.0000
    
    Epoch 00070: val_accuracy did not improve from 1.00000
    Epoch 71/2000
    10/10 [==============================] - 0s 43ms/step - loss: 5.0668e-04 - accuracy: 1.0000 - val_loss: 7.4026e-04 - val_accuracy: 1.0000
    
    Epoch 00071: val_accuracy did not improve from 1.00000
    Epoch 72/2000
    10/10 [==============================] - 0s 43ms/step - loss: 4.8384e-04 - accuracy: 1.0000 - val_loss: 7.0405e-04 - val_accuracy: 1.0000
    
    Epoch 00072: val_accuracy did not improve from 1.00000
    Epoch 73/2000
    10/10 [==============================] - 0s 42ms/step - loss: 4.6091e-04 - accuracy: 1.0000 - val_loss: 6.7255e-04 - val_accuracy: 1.0000
    
    Epoch 00073: val_accuracy did not improve from 1.00000
    Epoch 74/2000
    10/10 [==============================] - 0s 45ms/step - loss: 4.4084e-04 - accuracy: 1.0000 - val_loss: 6.4374e-04 - val_accuracy: 1.0000
    
    Epoch 00074: val_accuracy did not improve from 1.00000
    Epoch 75/2000
    10/10 [==============================] - 0s 42ms/step - loss: 4.2211e-04 - accuracy: 1.0000 - val_loss: 6.1681e-04 - val_accuracy: 1.0000
    
    Epoch 00075: val_accuracy did not improve from 1.00000
    Epoch 76/2000
    10/10 [==============================] - 0s 42ms/step - loss: 4.0484e-04 - accuracy: 1.0000 - val_loss: 5.9214e-04 - val_accuracy: 1.0000
    
    Epoch 00076: val_accuracy did not improve from 1.00000
    Epoch 77/2000
    10/10 [==============================] - 0s 42ms/step - loss: 3.8866e-04 - accuracy: 1.0000 - val_loss: 5.6560e-04 - val_accuracy: 1.0000
    
    Epoch 00077: val_accuracy did not improve from 1.00000
    Epoch 78/2000
    10/10 [==============================] - 0s 40ms/step - loss: 3.7336e-04 - accuracy: 1.0000 - val_loss: 5.4555e-04 - val_accuracy: 1.0000
    
    Epoch 00078: val_accuracy did not improve from 1.00000
    Epoch 79/2000
    10/10 [==============================] - 0s 41ms/step - loss: 3.5975e-04 - accuracy: 1.0000 - val_loss: 5.2469e-04 - val_accuracy: 1.0000
    
    Epoch 00079: val_accuracy did not improve from 1.00000
    Epoch 80/2000
    10/10 [==============================] - 0s 43ms/step - loss: 3.4676e-04 - accuracy: 1.0000 - val_loss: 5.0368e-04 - val_accuracy: 1.0000
    
    Epoch 00080: val_accuracy did not improve from 1.00000
    Epoch 81/2000
    10/10 [==============================] - 0s 40ms/step - loss: 3.3333e-04 - accuracy: 1.0000 - val_loss: 4.8990e-04 - val_accuracy: 1.0000
    
    Epoch 00081: val_accuracy did not improve from 1.00000
    Epoch 82/2000
    10/10 [==============================] - 0s 42ms/step - loss: 3.2246e-04 - accuracy: 1.0000 - val_loss: 4.7290e-04 - val_accuracy: 1.0000
    
    Epoch 00082: val_accuracy did not improve from 1.00000
    Epoch 83/2000
    10/10 [==============================] - 0s 42ms/step - loss: 3.1188e-04 - accuracy: 1.0000 - val_loss: 4.5597e-04 - val_accuracy: 1.0000
    
    Epoch 00083: val_accuracy did not improve from 1.00000
    Epoch 84/2000
    10/10 [==============================] - 0s 42ms/step - loss: 3.0094e-04 - accuracy: 1.0000 - val_loss: 4.4185e-04 - val_accuracy: 1.0000
    
    Epoch 00084: val_accuracy did not improve from 1.00000
    Epoch 85/2000
    10/10 [==============================] - 0s 42ms/step - loss: 2.9148e-04 - accuracy: 1.0000 - val_loss: 4.2189e-04 - val_accuracy: 1.0000
    
    Epoch 00085: val_accuracy did not improve from 1.00000
    Epoch 86/2000
    10/10 [==============================] - 0s 43ms/step - loss: 2.8073e-04 - accuracy: 1.0000 - val_loss: 3.7538e-04 - val_accuracy: 1.0000
    
    Epoch 00086: val_accuracy did not improve from 1.00000
    Epoch 87/2000
    10/10 [==============================] - 0s 42ms/step - loss: 2.7194e-04 - accuracy: 1.0000 - val_loss: 3.5864e-04 - val_accuracy: 1.0000
    
    Epoch 00087: val_accuracy did not improve from 1.00000
    Epoch 88/2000
    10/10 [==============================] - 0s 41ms/step - loss: 2.6130e-04 - accuracy: 1.0000 - val_loss: 3.4925e-04 - val_accuracy: 1.0000
    
    Epoch 00088: val_accuracy did not improve from 1.00000
    Epoch 89/2000
    10/10 [==============================] - 0s 43ms/step - loss: 2.5341e-04 - accuracy: 1.0000 - val_loss: 3.3956e-04 - val_accuracy: 1.0000
    
    Epoch 00089: val_accuracy did not improve from 1.00000
    Epoch 90/2000
    10/10 [==============================] - 0s 42ms/step - loss: 2.4571e-04 - accuracy: 1.0000 - val_loss: 3.1883e-04 - val_accuracy: 1.0000
    
    Epoch 00090: val_accuracy did not improve from 1.00000
    Epoch 91/2000
    10/10 [==============================] - 0s 42ms/step - loss: 2.3662e-04 - accuracy: 1.0000 - val_loss: 3.0886e-04 - val_accuracy: 1.0000
    
    Epoch 00091: val_accuracy did not improve from 1.00000
    Epoch 92/2000
    10/10 [==============================] - 0s 41ms/step - loss: 2.3005e-04 - accuracy: 1.0000 - val_loss: 3.0076e-04 - val_accuracy: 1.0000
    
    Epoch 00092: val_accuracy did not improve from 1.00000
    Epoch 93/2000
    10/10 [==============================] - 0s 43ms/step - loss: 2.2287e-04 - accuracy: 1.0000 - val_loss: 2.9805e-04 - val_accuracy: 1.0000
    
    Epoch 00093: val_accuracy did not improve from 1.00000
    Epoch 94/2000
    10/10 [==============================] - 0s 41ms/step - loss: 2.1641e-04 - accuracy: 1.0000 - val_loss: 2.9316e-04 - val_accuracy: 1.0000
    
    Epoch 00094: val_accuracy did not improve from 1.00000
    Epoch 95/2000
    10/10 [==============================] - 0s 43ms/step - loss: 2.1044e-04 - accuracy: 1.0000 - val_loss: 2.8779e-04 - val_accuracy: 1.0000
    
    Epoch 00095: val_accuracy did not improve from 1.00000
    Epoch 96/2000
    10/10 [==============================] - 0s 42ms/step - loss: 2.0531e-04 - accuracy: 1.0000 - val_loss: 2.8114e-04 - val_accuracy: 1.0000
    
    Epoch 00096: val_accuracy did not improve from 1.00000
    Epoch 97/2000
    10/10 [==============================] - 0s 42ms/step - loss: 1.9977e-04 - accuracy: 1.0000 - val_loss: 2.7443e-04 - val_accuracy: 1.0000
    
    Epoch 00097: val_accuracy did not improve from 1.00000
    Epoch 98/2000
    10/10 [==============================] - 0s 41ms/step - loss: 1.9424e-04 - accuracy: 1.0000 - val_loss: 2.6919e-04 - val_accuracy: 1.0000
    
    Epoch 00098: val_accuracy did not improve from 1.00000
    Epoch 99/2000
    10/10 [==============================] - 0s 42ms/step - loss: 1.8926e-04 - accuracy: 1.0000 - val_loss: 2.6381e-04 - val_accuracy: 1.0000
    
    Epoch 00099: val_accuracy did not improve from 1.00000
    Epoch 100/2000
    10/10 [==============================] - 0s 41ms/step - loss: 1.8432e-04 - accuracy: 1.0000 - val_loss: 2.5892e-04 - val_accuracy: 1.0000
    
    Epoch 00100: val_accuracy did not improve from 1.00000
    Epoch 101/2000
    10/10 [==============================] - 0s 42ms/step - loss: 1.7980e-04 - accuracy: 1.0000 - val_loss: 2.5380e-04 - val_accuracy: 1.0000
    
    Epoch 00101: val_accuracy did not improve from 1.00000
    Epoch 102/2000
    10/10 [==============================] - 0s 43ms/step - loss: 1.7530e-04 - accuracy: 1.0000 - val_loss: 2.4898e-04 - val_accuracy: 1.0000
    
    Epoch 00102: val_accuracy did not improve from 1.00000
    Epoch 103/2000
    10/10 [==============================] - 0s 42ms/step - loss: 1.7097e-04 - accuracy: 1.0000 - val_loss: 2.4421e-04 - val_accuracy: 1.0000
    
    Epoch 00103: val_accuracy did not improve from 1.00000
    Epoch 104/2000
    10/10 [==============================] - 0s 42ms/step - loss: 1.6679e-04 - accuracy: 1.0000 - val_loss: 2.3982e-04 - val_accuracy: 1.0000
    
    Epoch 00104: val_accuracy did not improve from 1.00000
    Epoch 105/2000
    10/10 [==============================] - 0s 40ms/step - loss: 1.6286e-04 - accuracy: 1.0000 - val_loss: 2.3554e-04 - val_accuracy: 1.0000
    
    Epoch 00105: val_accuracy did not improve from 1.00000
    Epoch 106/2000
    10/10 [==============================] - 0s 43ms/step - loss: 1.5919e-04 - accuracy: 1.0000 - val_loss: 2.3038e-04 - val_accuracy: 1.0000
    
    Epoch 00106: val_accuracy did not improve from 1.00000
    Epoch 107/2000
    10/10 [==============================] - 0s 41ms/step - loss: 1.5545e-04 - accuracy: 1.0000 - val_loss: 2.2587e-04 - val_accuracy: 1.0000
    
    Epoch 00107: val_accuracy did not improve from 1.00000
    Epoch 108/2000
    10/10 [==============================] - 0s 43ms/step - loss: 1.5186e-04 - accuracy: 1.0000 - val_loss: 2.2157e-04 - val_accuracy: 1.0000
    
    Epoch 00108: val_accuracy did not improve from 1.00000
    Epoch 109/2000
    10/10 [==============================] - 0s 43ms/step - loss: 1.4836e-04 - accuracy: 1.0000 - val_loss: 2.1814e-04 - val_accuracy: 1.0000
    
    Epoch 00109: val_accuracy did not improve from 1.00000
    Epoch 110/2000
    10/10 [==============================] - 0s 43ms/step - loss: 1.4512e-04 - accuracy: 1.0000 - val_loss: 2.1466e-04 - val_accuracy: 1.0000
    
    Epoch 00110: val_accuracy did not improve from 1.00000
    Epoch 111/2000
    10/10 [==============================] - 0s 41ms/step - loss: 1.4186e-04 - accuracy: 1.0000 - val_loss: 2.0782e-04 - val_accuracy: 1.0000
    
    Epoch 00111: val_accuracy did not improve from 1.00000
    Epoch 112/2000
    10/10 [==============================] - 0s 42ms/step - loss: 1.3882e-04 - accuracy: 1.0000 - val_loss: 2.0302e-04 - val_accuracy: 1.0000
    
    Epoch 00112: val_accuracy did not improve from 1.00000
    Epoch 113/2000
    10/10 [==============================] - 0s 44ms/step - loss: 1.3577e-04 - accuracy: 1.0000 - val_loss: 1.9915e-04 - val_accuracy: 1.0000
    
    Epoch 00113: val_accuracy did not improve from 1.00000
    Epoch 114/2000
    10/10 [==============================] - 0s 43ms/step - loss: 1.3316e-04 - accuracy: 1.0000 - val_loss: 1.9585e-04 - val_accuracy: 1.0000
    
    Epoch 00114: val_accuracy did not improve from 1.00000
    Epoch 115/2000
    10/10 [==============================] - 0s 44ms/step - loss: 1.3010e-04 - accuracy: 1.0000 - val_loss: 1.9300e-04 - val_accuracy: 1.0000
    
    Epoch 00115: val_accuracy did not improve from 1.00000
    Epoch 116/2000
    10/10 [==============================] - 0s 43ms/step - loss: 1.2741e-04 - accuracy: 1.0000 - val_loss: 1.9016e-04 - val_accuracy: 1.0000
    
    Epoch 00116: val_accuracy did not improve from 1.00000
    Epoch 117/2000
    10/10 [==============================] - 0s 42ms/step - loss: 1.2485e-04 - accuracy: 1.0000 - val_loss: 1.8784e-04 - val_accuracy: 1.0000
    
    Epoch 00117: val_accuracy did not improve from 1.00000
    Epoch 118/2000
    10/10 [==============================] - 0s 42ms/step - loss: 1.2242e-04 - accuracy: 1.0000 - val_loss: 1.8558e-04 - val_accuracy: 1.0000
    
    Epoch 00118: val_accuracy did not improve from 1.00000
    Epoch 119/2000
    10/10 [==============================] - 0s 41ms/step - loss: 1.2003e-04 - accuracy: 1.0000 - val_loss: 1.8294e-04 - val_accuracy: 1.0000
    
    Epoch 00119: val_accuracy did not improve from 1.00000
    Epoch 120/2000
    10/10 [==============================] - 0s 42ms/step - loss: 1.1733e-04 - accuracy: 1.0000 - val_loss: 1.7990e-04 - val_accuracy: 1.0000
    
    Epoch 00120: val_accuracy did not improve from 1.00000
    Epoch 121/2000
     9/10 [==========================>...] - ETA: 0s - loss: 1.1390e-04 - accuracy: 1.0000

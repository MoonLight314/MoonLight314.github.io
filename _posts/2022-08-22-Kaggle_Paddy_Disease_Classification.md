---
title: "Kaggle : Paddy Disease Classification"
date: 2021-10-29 08:26:28 -0400
categories: Deep Learning
---
# Paddy Disease Classification

<br>
<br>
<br>

[https://www.kaggle.com/competitions/paddy-disease-classification/overview](https://www.kaggle.com/competitions/paddy-disease-classification/overview)

<br>

* 쌀의 잎 모양을 보고 현재 질병을 있는 없는지, 질병이 있으면 어떤 질병인지 판단하는 Competition입니다.

<br>

* 기본적으로 Image Classification으로 진행해 보도록 하겠습니다.

<br>
<br>

## 0. Train

<br>
<br>

```python
import pandas as pd
import os
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GlobalAveragePooling2D , BatchNormalization , Dropout , Dense
from tensorflow.keras.callbacks import TensorBoard , ModelCheckpoint , LearningRateScheduler , EarlyStopping
```


```python
tf.__version__
```




    '2.10.0-dev20220727'




```python
BATCH_SIZE = 64
DROP_OUT_RATE = 0.2
```


```python
train_csv = pd.read_csv("./paddy-disease-classification/train.csv")
```


```python
train_csv.head()
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>image_id</th>
      <th>label</th>
      <th>variety</th>
      <th>age</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>100330.jpg</td>
      <td>bacterial_leaf_blight</td>
      <td>ADT45</td>
      <td>45</td>
    </tr>
    <tr>
      <th>1</th>
      <td>100365.jpg</td>
      <td>bacterial_leaf_blight</td>
      <td>ADT45</td>
      <td>45</td>
    </tr>
    <tr>
      <th>2</th>
      <td>100382.jpg</td>
      <td>bacterial_leaf_blight</td>
      <td>ADT45</td>
      <td>45</td>
    </tr>
    <tr>
      <th>3</th>
      <td>100632.jpg</td>
      <td>bacterial_leaf_blight</td>
      <td>ADT45</td>
      <td>45</td>
    </tr>
    <tr>
      <th>4</th>
      <td>101918.jpg</td>
      <td>bacterial_leaf_blight</td>
      <td>ADT45</td>
      <td>45</td>
    </tr>
  </tbody>
</table>
</div>




```python
train_csv['label'].value_counts()
```




    normal                      1764
    blast                       1738
    hispa                       1594
    dead_heart                  1442
    tungro                      1088
    brown_spot                   965
    downy_mildew                 620
    bacterial_leaf_blight        479
    bacterial_leaf_streak        380
    bacterial_panicle_blight     337
    Name: label, dtype: int64




```python
train_csv['variety'].value_counts()
```




    ADT45             6992
    KarnatakaPonni     988
    Ponni              657
    AtchayaPonni       461
    Zonal              399
    AndraPonni         377
    Onthanel           351
    IR20               114
    RR                  36
    Surya               32
    Name: variety, dtype: int64

<br>
<br>

* 필요한 것은 'image_id'와 'label'뿐입니다.   

<br>

```python
image_id = train_csv['image_id'].tolist()
```

<br>

```python
label = train_csv['label'].tolist()
```

<br>

```python
train_csv['file_path'] = "./paddy-disease-classification/train_images/" + train_csv['label'] + "/" + train_csv['image_id']
train_csv
```
<br>

<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>image_id</th>
      <th>label</th>
      <th>variety</th>
      <th>age</th>
      <th>file_path</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>100330.jpg</td>
      <td>bacterial_leaf_blight</td>
      <td>ADT45</td>
      <td>45</td>
      <td>./paddy-disease-classification/train_images/ba...</td>
    </tr>
    <tr>
      <th>1</th>
      <td>100365.jpg</td>
      <td>bacterial_leaf_blight</td>
      <td>ADT45</td>
      <td>45</td>
      <td>./paddy-disease-classification/train_images/ba...</td>
    </tr>
    <tr>
      <th>2</th>
      <td>100382.jpg</td>
      <td>bacterial_leaf_blight</td>
      <td>ADT45</td>
      <td>45</td>
      <td>./paddy-disease-classification/train_images/ba...</td>
    </tr>
    <tr>
      <th>3</th>
      <td>100632.jpg</td>
      <td>bacterial_leaf_blight</td>
      <td>ADT45</td>
      <td>45</td>
      <td>./paddy-disease-classification/train_images/ba...</td>
    </tr>
    <tr>
      <th>4</th>
      <td>101918.jpg</td>
      <td>bacterial_leaf_blight</td>
      <td>ADT45</td>
      <td>45</td>
      <td>./paddy-disease-classification/train_images/ba...</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>10402</th>
      <td>107607.jpg</td>
      <td>tungro</td>
      <td>Zonal</td>
      <td>55</td>
      <td>./paddy-disease-classification/train_images/tu...</td>
    </tr>
    <tr>
      <th>10403</th>
      <td>107811.jpg</td>
      <td>tungro</td>
      <td>Zonal</td>
      <td>55</td>
      <td>./paddy-disease-classification/train_images/tu...</td>
    </tr>
    <tr>
      <th>10404</th>
      <td>108547.jpg</td>
      <td>tungro</td>
      <td>Zonal</td>
      <td>55</td>
      <td>./paddy-disease-classification/train_images/tu...</td>
    </tr>
    <tr>
      <th>10405</th>
      <td>110245.jpg</td>
      <td>tungro</td>
      <td>Zonal</td>
      <td>55</td>
      <td>./paddy-disease-classification/train_images/tu...</td>
    </tr>
    <tr>
      <th>10406</th>
      <td>110381.jpg</td>
      <td>tungro</td>
      <td>Zonal</td>
      <td>55</td>
      <td>./paddy-disease-classification/train_images/tu...</td>
    </tr>
  </tbody>
</table>
<p>10407 rows × 5 columns</p>
</div>

<br>
<br>

```python
file_path = train_csv['file_path'].tolist()
file_path
```


    ['./paddy-disease-classification/train_images/bacterial_leaf_blight/100330.jpg',
     './paddy-disease-classification/train_images/bacterial_leaf_blight/100365.jpg',
     './paddy-disease-classification/train_images/bacterial_leaf_blight/100382.jpg',
     './paddy-disease-classification/train_images/bacterial_leaf_blight/100632.jpg',
     './paddy-disease-classification/train_images/bacterial_panicle_blight/104162.jpg',
     './paddy-disease-classification/train_images/bacterial_panicle_blight/104300.jpg',
     './paddy-disease-classification/train_images/bacterial_panicle_blight/104460.jpg',
     './paddy-disease-classification/train_images/bacterial_panicle_blight/104684.jpg',
     './paddy-disease-classification/train_images/bacterial_panicle_blight/104815.jpg',
     './paddy-disease-classification/train_images/bacterial_panicle_blight/104853.jpg',
     './paddy-disease-classification/train_images/bacterial_panicle_blight/104912.jpg',
     './paddy-disease-classification/train_images/bacterial_panicle_blight/105065.jpg',
     ...]

<br>

* Target은 정상상태(normal)을 포함하여 총 10가지 입니다.

<br>

```python
le = LabelEncoder()
le.fit(label)
print(le.classes_)
```

    ['bacterial_leaf_blight' 'bacterial_leaf_streak'
     'bacterial_panicle_blight' 'blast' 'brown_spot' 'dead_heart'
     'downy_mildew' 'hispa' 'normal' 'tungro']

<br>

* 총 5개의 Train / Val. Set으로 나누어 5개의 Model을 만든 후, 가장 성능이 좋은 하나의 Model로 Predict합니다.

<br>

```python
from sklearn.model_selection import StratifiedKFold
```


```python
skf = StratifiedKFold(n_splits=5,random_state=27, shuffle=True)
```


```python
skf
```

<br>

    StratifiedKFold(n_splits=5, random_state=27, shuffle=True)

<br>

```python
folds = []

for train_index, val_index in skf.split(file_path, label):
    X_train = [file_path[i] for i in train_index ]
    X_val = [file_path[i] for i in val_index ]
    
    label_train= [label[i] for i in train_index ]
    label_val = [label[i] for i in val_index ]
    
    le_label = le.transform(label_train)
    label_train = tf.keras.utils.to_categorical(le_label , num_classes=10)
    
    le_label = le.transform(label_val)
    label_val = tf.keras.utils.to_categorical(le_label , num_classes=10)
    
    folds.append([X_train,label_train,X_val,label_val])
```


```python
len(folds)
```




    5




```python
len(folds[0][0]) , len(folds[0][1]) , len(folds[0][2]) , len(folds[0][3])
```




    (8325, 8325, 2082, 2082)




```python
folds[0][1][0]
```




    array([1., 0., 0., 0., 0., 0., 0., 0., 0., 0.], dtype=float32)

<br>
<br>

* Data Augmentation을 Train Dataset에만 적용합니다.

<br>

```python
def load_image_train( image_path , label):
    img = tf.io.read_file(image_path)
    img = tf.image.decode_png(img, channels=3)
    
    # Train Dataset에만 Augmentation을 적용
    img = tf.image.random_flip_left_right(img)
    img = tf.image.random_flip_up_down(img)
    img = tf.image.random_crop(img , tf.shape(img))
    
    img = tf.image.random_contrast(img , 0.5 , 1.5)
    img = tf.image.random_saturation(img, 0.75, 1.25)    
    img = tf.image.random_brightness(img, 0.2)
    
    img = tf.image.resize(img, (224, 224))
    img = tf.keras.applications.efficientnet_v2.preprocess_input(img)
    
    return img , label
```


```python
def load_image_val( image_path , label):
    img = tf.io.read_file(image_path)
    img = tf.image.decode_png(img, channels=3)   
    
    img = tf.image.resize(img, (224, 224))
    img = tf.keras.applications.efficientnet_v2.preprocess_input(img)
    
    return img , label
```

<br>

* Callback들을 정의합니다.


```python
initial_learning_rate = 0.01

def lr_exp_decay(epoch, lr):
    k = 0.1
    return initial_learning_rate * np.math.exp(-k*epoch)

lr_scheduler = LearningRateScheduler(lr_exp_decay, verbose=1)
```


```python
log_dir = os.path.join('Logs')
tb_callback = TensorBoard(log_dir=log_dir)
```


```python
es = EarlyStopping(monitor = 'val_accuracy', patience = 3, mode = 'auto')
```

<br>

* EfficientNet으로 Feature Extraction합니다.

* 각 Fold 마다 다른 Model을 Train합니다.


```python
def generate_model():
    
    effnet_v2 = tf.keras.applications.efficientnet_v2.EfficientNetV2B0(
                weights=None,
                include_top=False)
    
    model= Sequential()

    model.add( effnet_v2 )

    model.add( GlobalAveragePooling2D() ) 
    model.add( Dropout( DROP_OUT_RATE ) ) 
    model.add( BatchNormalization() ) 
    model.add( Dense(128, activation='relu') )
    model.add( Dropout( DROP_OUT_RATE ) ) 
    model.add( BatchNormalization() ) 

    model.add( Dense(10, activation='softmax') )
    
    model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-3),
    loss='categorical_crossentropy',
    metrics=['accuracy']
    )
    
    return model
```

<br>

* 각 Fold별로 Train 시작합니다.

<br>

```python
for idx,fold in enumerate(folds):
    
    # Fold 별로 Checkpoint 다르게 설정
    CHECKPOINT_PATH = os.path.join('CheckPoints_' + str(idx))
    
    cp = ModelCheckpoint(filepath=CHECKPOINT_PATH, 
                 monitor='val_accuracy',                     
                 save_best_only = True,
                 verbose = 1)
    
    train_dataset = tf.data.Dataset.from_tensor_slices( (fold[0] , 
                                                         fold[1] )
                                                          )
    
    val_dataset = tf.data.Dataset.from_tensor_slices( (fold[2] , 
                                                       fold[3] )
                                                        )
    
    train_dataset = train_dataset.shuffle(buffer_size=len(fold[0]))\
                                .map( load_image_train, num_parallel_calls=tf.data.experimental.AUTOTUNE)\
                                .repeat()\
                                .batch(BATCH_SIZE)\
                                .prefetch(tf.data.experimental.AUTOTUNE)


    val_dataset = val_dataset.shuffle(buffer_size=len(fold[2]))\
                                .map( load_image_val, num_parallel_calls=tf.data.experimental.AUTOTUNE)\
                                .repeat()\
                                .batch(BATCH_SIZE)\
                                .prefetch(tf.data.experimental.AUTOTUNE)    #
    
    model = generate_model()
    
    hist = model.fit(train_dataset,
                     validation_data=val_dataset,
                     callbacks=[lr_scheduler , cp , tb_callback, es],
                     steps_per_epoch = 2* (len(fold[0]) / BATCH_SIZE),
                     validation_steps = 2* (len(fold[2]) / BATCH_SIZE),                     
                     epochs = 100,
                     verbose = 1 
                    )

```

    
    Epoch 1: LearningRateScheduler setting learning rate to 0.01.
    Epoch 1/100
    261/260 [==============================] - ETA: 0s - loss: 2.3030 - accuracy: 0.1603
    Epoch 1: val_accuracy improved from -inf to 0.16809, saving model to CheckPoints_0
    

    WARNING:absl:Function `_wrapped_model` contains input name(s) efficientnetv2-b0_input with unsupported characters which will be renamed to efficientnetv2_b0_input in the SavedModel.
    WARNING:absl:Found untraced functions such as _jit_compiled_convolution_op, _jit_compiled_convolution_op, _jit_compiled_convolution_op, _jit_compiled_convolution_op, _jit_compiled_convolution_op while saving (showing 5 of 91). These functions will not be directly callable after loading.
    

    INFO:tensorflow:Assets written to: CheckPoints_0\assets
    

    260/260 [==============================] - 149s 573ms/step - loss: 0.2825 - accuracy: 0.9074 - val_loss: 0.3361 - val_accuracy: 0.9067 - lr: 0.0017
    
    Epoch 20: LearningRateScheduler setting learning rate to 0.0014956861922263505.
    Epoch 20/100
    261/260 [==============================] - ETA: 0s - loss: 0.2324 - accuracy: 0.9221
    Epoch 20: val_accuracy did not improve from 0.90672
    260/260 [==============================] - 107s 411ms/step - loss: 0.2324 - accuracy: 0.9221 - val_loss: 0.3662 - val_accuracy: 0.8980 - lr: 0.0015
    
    Epoch 21: LearningRateScheduler setting learning rate to 0.0013533528323661271.
    Epoch 21/100
    261/260 [==============================] - ETA: 0s - loss: 0.2156 - accuracy: 0.9264
    Epoch 21: val_accuracy did not improve from 0.90672
    260/260 [==============================] - 102s 390ms/step - loss: 0.2156 - accuracy: 0.9264 - val_loss: 0.3430 - val_accuracy: 0.9058 - lr: 0.0014
    
    Epoch 22: LearningRateScheduler setting learning rate to 0.001224564282529819.
    Epoch 22/100
    261/260 [==============================] - ETA: 0s - loss: 0.1887 - accuracy: 0.9379
    Epoch 22: val_accuracy did not improve from 0.90672
    260/260 [==============================] - 103s 395ms/step - loss: 0.1887 - accuracy: 0.9379 - val_loss: 0.3635 - val_accuracy: 0.9048 - lr: 0.0012

<br>
<br>

## 1. Choose Best Model

<br>

* 학습을 마친 5개의 Model 중에 성능이 가장 좋은 Model을 선택합니다.

<br>

* 전체 Train Set에서 가장 점수가 높은 Model을 선택합니다.

<br>

```python
import pandas as pd
import os
import numpy as np
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
from keras.models import load_model, save_model

from tqdm import tqdm
import pandas as pd
import os
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GlobalAveragePooling2D , BatchNormalization , Dropout , Dense
from tensorflow.keras.callbacks import TensorBoard , ModelCheckpoint , LearningRateScheduler , EarlyStopping
```


```python
BATCH_SIZE = 32
```


```python
train_csv = pd.read_csv("./paddy-disease-classification/train.csv")
```


```python
image_id = train_csv['image_id'].tolist()
```


```python
label = train_csv['label'].tolist()
```


```python
train_csv['file_path'] = "./paddy-disease-classification/train_images/" + train_csv['label'] + "/" + train_csv['image_id']
train_csv
```


<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>image_id</th>
      <th>label</th>
      <th>variety</th>
      <th>age</th>
      <th>file_path</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>100330.jpg</td>
      <td>bacterial_leaf_blight</td>
      <td>ADT45</td>
      <td>45</td>
      <td>./paddy-disease-classification/train_images/ba...</td>
    </tr>
    <tr>
      <th>1</th>
      <td>100365.jpg</td>
      <td>bacterial_leaf_blight</td>
      <td>ADT45</td>
      <td>45</td>
      <td>./paddy-disease-classification/train_images/ba...</td>
    </tr>
    <tr>
      <th>2</th>
      <td>100382.jpg</td>
      <td>bacterial_leaf_blight</td>
      <td>ADT45</td>
      <td>45</td>
      <td>./paddy-disease-classification/train_images/ba...</td>
    </tr>
    <tr>
      <th>3</th>
      <td>100632.jpg</td>
      <td>bacterial_leaf_blight</td>
      <td>ADT45</td>
      <td>45</td>
      <td>./paddy-disease-classification/train_images/ba...</td>
    </tr>
    <tr>
      <th>4</th>
      <td>101918.jpg</td>
      <td>bacterial_leaf_blight</td>
      <td>ADT45</td>
      <td>45</td>
      <td>./paddy-disease-classification/train_images/ba...</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>10402</th>
      <td>107607.jpg</td>
      <td>tungro</td>
      <td>Zonal</td>
      <td>55</td>
      <td>./paddy-disease-classification/train_images/tu...</td>
    </tr>
    <tr>
      <th>10403</th>
      <td>107811.jpg</td>
      <td>tungro</td>
      <td>Zonal</td>
      <td>55</td>
      <td>./paddy-disease-classification/train_images/tu...</td>
    </tr>
    <tr>
      <th>10404</th>
      <td>108547.jpg</td>
      <td>tungro</td>
      <td>Zonal</td>
      <td>55</td>
      <td>./paddy-disease-classification/train_images/tu...</td>
    </tr>
    <tr>
      <th>10405</th>
      <td>110245.jpg</td>
      <td>tungro</td>
      <td>Zonal</td>
      <td>55</td>
      <td>./paddy-disease-classification/train_images/tu...</td>
    </tr>
    <tr>
      <th>10406</th>
      <td>110381.jpg</td>
      <td>tungro</td>
      <td>Zonal</td>
      <td>55</td>
      <td>./paddy-disease-classification/train_images/tu...</td>
    </tr>
  </tbody>
</table>
<p>10407 rows × 5 columns</p>
</div>




```python
file_path = train_csv['file_path'].tolist()
file_path[:10]
```




    ['./paddy-disease-classification/train_images/bacterial_leaf_blight/100330.jpg',
     './paddy-disease-classification/train_images/bacterial_leaf_blight/100365.jpg',
     './paddy-disease-classification/train_images/bacterial_leaf_blight/100382.jpg',
     './paddy-disease-classification/train_images/bacterial_leaf_blight/100632.jpg',
     './paddy-disease-classification/train_images/bacterial_leaf_blight/101918.jpg',
     './paddy-disease-classification/train_images/bacterial_leaf_blight/102353.jpg',
     './paddy-disease-classification/train_images/bacterial_leaf_blight/102848.jpg',
     './paddy-disease-classification/train_images/bacterial_leaf_blight/103051.jpg',
     './paddy-disease-classification/train_images/bacterial_leaf_blight/103702.jpg',
     './paddy-disease-classification/train_images/bacterial_leaf_blight/103920.jpg']




```python
le = LabelEncoder()
le.fit(label)
print(le.classes_)
```

    ['bacterial_leaf_blight' 'bacterial_leaf_streak'
     'bacterial_panicle_blight' 'blast' 'brown_spot' 'dead_heart'
     'downy_mildew' 'hispa' 'normal' 'tungro']
    


```python
label_train = train_csv['label'].tolist()
```


```python
le_label = le.transform(label_train)
label_train = tf.keras.utils.to_categorical(le_label , num_classes=10)
```


```python
label_train
```




    array([[1., 0., 0., ..., 0., 0., 0.],
           [1., 0., 0., ..., 0., 0., 0.],
           [1., 0., 0., ..., 0., 0., 0.],
           ...,
           [0., 0., 0., ..., 0., 0., 1.],
           [0., 0., 0., ..., 0., 0., 1.],
           [0., 0., 0., ..., 0., 0., 1.]], dtype=float32)




```python
def load_image( image_path , label ):
    img = tf.io.read_file(image_path)
    img = tf.image.decode_png(img, channels=3)   
    
    img = tf.image.resize(img, (224, 224))
    img = tf.keras.applications.efficientnet_v2.preprocess_input(img)
    
    return img , label
```


```python
dataset = tf.data.Dataset.from_tensor_slices( (file_path , label_train) )
   
eval_dataset = dataset.shuffle(buffer_size=len(file_path))\
                            .map( load_image, num_parallel_calls=tf.data.experimental.AUTOTUNE)\
                            .batch(BATCH_SIZE)\
                            .prefetch(tf.data.experimental.AUTOTUNE)
```

<br>

* 5개 Model을 각각 Load하여 전체 Train Set으로 평가합니다.

<br>

```python
models = []
result = []

for idx in range(5):
    model = load_model("CheckPoints_"+str(idx))
    result.append( model.evaluate(eval_dataset) )
```

    326/326 [==============================] - 17s 35ms/step - loss: 0.0657 - accuracy: 0.9845
    326/326 [==============================] - 13s 35ms/step - loss: 0.0597 - accuracy: 0.9842
    326/326 [==============================] - 12s 34ms/step - loss: 0.0532 - accuracy: 0.9866
    326/326 [==============================] - 12s 34ms/step - loss: 0.1092 - accuracy: 0.9690
    326/326 [==============================] - 13s 35ms/step - loss: 0.1551 - accuracy: 0.9529
    


```python
result
```




    [[0.06573611497879028, 0.9845296144485474],
     [0.05972518026828766, 0.9842413663864136],
     [0.05319569632411003, 0.9866436123847961],
     [0.10924600809812546, 0.9689632058143616],
     [0.15512056648731232, 0.9529163241386414]]

<br>

* 가장 점수가 높은 Model을 찾습니다.

<br>

```python
r = np.array(result)
```


```python
r.T[-1:]
```

<br>

    array([[0.98452961, 0.98424137, 0.98664361, 0.96896321, 0.95291632]])




```python
best_model_idx = np.argmax(r.T[-1:])
best_model_idx
```




    2

<br>

* 2번 Model이 가장 성능이 좋습니다. 이 Model로 Submission Data를 만듭시다.

<br>
<br>
<br>

## Inference   

<br>

```python
train_csv = pd.read_csv("./paddy-disease-classification/train.csv")
submission_csv = pd.read_csv("./paddy-disease-classification/sample_submission.csv")
```


```python
label = train_csv['label'].tolist()
```


```python
le = LabelEncoder()
le.fit(label)
print(le.classes_)
```

    ['bacterial_leaf_blight' 'bacterial_leaf_streak'
     'bacterial_panicle_blight' 'blast' 'brown_spot' 'dead_heart'
     'downy_mildew' 'hispa' 'normal' 'tungro']
    


```python
le_label = le.transform(label)
labels = tf.keras.utils.to_categorical(le_label , num_classes=10)
```


```python
submission_csv.head()
```




<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>image_id</th>
      <th>label</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>200001.jpg</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1</th>
      <td>200002.jpg</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2</th>
      <td>200003.jpg</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>3</th>
      <td>200004.jpg</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>4</th>
      <td>200005.jpg</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>




```python
submission_csv['file_path'] = "./paddy-disease-classification/test_images/" + submission_csv['image_id']
submission_csv
```




<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>image_id</th>
      <th>label</th>
      <th>file_path</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>200001.jpg</td>
      <td>NaN</td>
      <td>./paddy-disease-classification/test_images/200...</td>
    </tr>
    <tr>
      <th>1</th>
      <td>200002.jpg</td>
      <td>NaN</td>
      <td>./paddy-disease-classification/test_images/200...</td>
    </tr>
    <tr>
      <th>2</th>
      <td>200003.jpg</td>
      <td>NaN</td>
      <td>./paddy-disease-classification/test_images/200...</td>
    </tr>
    <tr>
      <th>3</th>
      <td>200004.jpg</td>
      <td>NaN</td>
      <td>./paddy-disease-classification/test_images/200...</td>
    </tr>
    <tr>
      <th>4</th>
      <td>200005.jpg</td>
      <td>NaN</td>
      <td>./paddy-disease-classification/test_images/200...</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>3464</th>
      <td>203465.jpg</td>
      <td>NaN</td>
      <td>./paddy-disease-classification/test_images/203...</td>
    </tr>
    <tr>
      <th>3465</th>
      <td>203466.jpg</td>
      <td>NaN</td>
      <td>./paddy-disease-classification/test_images/203...</td>
    </tr>
    <tr>
      <th>3466</th>
      <td>203467.jpg</td>
      <td>NaN</td>
      <td>./paddy-disease-classification/test_images/203...</td>
    </tr>
    <tr>
      <th>3467</th>
      <td>203468.jpg</td>
      <td>NaN</td>
      <td>./paddy-disease-classification/test_images/203...</td>
    </tr>
    <tr>
      <th>3468</th>
      <td>203469.jpg</td>
      <td>NaN</td>
      <td>./paddy-disease-classification/test_images/203...</td>
    </tr>
  </tbody>
</table>
<p>3469 rows × 3 columns</p>
</div>




```python
file_path = submission_csv['file_path'].tolist()
file_path[:10]
```




    ['./paddy-disease-classification/test_images/200001.jpg',
     './paddy-disease-classification/test_images/200002.jpg',
     './paddy-disease-classification/test_images/200003.jpg',
     './paddy-disease-classification/test_images/200004.jpg',
     './paddy-disease-classification/test_images/200005.jpg',
     './paddy-disease-classification/test_images/200006.jpg',
     './paddy-disease-classification/test_images/200007.jpg',
     './paddy-disease-classification/test_images/200008.jpg',
     './paddy-disease-classification/test_images/200009.jpg',
     './paddy-disease-classification/test_images/200010.jpg']




```python
submission_csv.drop(columns=['file_path'],inplace=True)
submission_csv.head()
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>image_id</th>
      <th>label</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>200001.jpg</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1</th>
      <td>200002.jpg</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2</th>
      <td>200003.jpg</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>3</th>
      <td>200004.jpg</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>4</th>
      <td>200005.jpg</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>

<br>

* 찾은 Model을 Load하여 Test Image를 Predict합니다.

<br>

```python
best_model_chkpoint = "CheckPoints_" + str(best_model_idx)
best_model_chkpoint
```




    'CheckPoints_2'




```python
model = load_model( best_model_chkpoint )
```


```python
targets = []

for path in tqdm(file_path):
    
    img = tf.io.read_file(path)
    img = tf.image.decode_png(img, channels=3)   
    
    img = tf.image.resize(img, (224, 224))
    img = tf.keras.applications.efficientnet_v2.preprocess_input(img)
    img = tf.reshape(img , (-1,224,224,3))
    
    pred = model.predict(img)
    targets.append(le.inverse_transform([np.argmax(pred)])[0])
```

    100%|██████████████████████████████████████████████████████████████████████████████| 3469/3469 [03:46<00:00, 15.35it/s]
    


```python
targets[:10]
```




    ['hispa',
     'normal',
     'blast',
     'blast',
     'blast',
     'brown_spot',
     'dead_heart',
     'brown_spot',
     'hispa',
     'normal']




```python
submission_csv['label'] = targets
submission_csv[:10]
```




<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>image_id</th>
      <th>label</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>200001.jpg</td>
      <td>hispa</td>
    </tr>
    <tr>
      <th>1</th>
      <td>200002.jpg</td>
      <td>normal</td>
    </tr>
    <tr>
      <th>2</th>
      <td>200003.jpg</td>
      <td>blast</td>
    </tr>
    <tr>
      <th>3</th>
      <td>200004.jpg</td>
      <td>blast</td>
    </tr>
    <tr>
      <th>4</th>
      <td>200005.jpg</td>
      <td>blast</td>
    </tr>
    <tr>
      <th>5</th>
      <td>200006.jpg</td>
      <td>brown_spot</td>
    </tr>
    <tr>
      <th>6</th>
      <td>200007.jpg</td>
      <td>dead_heart</td>
    </tr>
    <tr>
      <th>7</th>
      <td>200008.jpg</td>
      <td>brown_spot</td>
    </tr>
    <tr>
      <th>8</th>
      <td>200009.jpg</td>
      <td>hispa</td>
    </tr>
    <tr>
      <th>9</th>
      <td>200010.jpg</td>
      <td>normal</td>
    </tr>
  </tbody>
</table>
</div>




```python
submission_csv.to_csv("submission.csv",index=False)
```

<p align="center">
  <img src="/assets/Paddy_Disease_Classification/Score.png">
</p>

<br>
<br>

* 점수는 0.9454가 나왔네요.

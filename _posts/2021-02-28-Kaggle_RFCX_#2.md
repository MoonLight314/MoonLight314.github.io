---
title: "Kaggle Competition - Rainforest Connection Species Audio Detection #2"
date: 2021-02-28 08:26:28 -0400
categories: Kaggle
---
### Kaggle Competition - Rainforest Connection Species Audio Detection - Revision #02

<br>
<br>
<br>
<br>

* 이번 Post에서는 지난 번에 올렸었던 Kaggle - Rainforest Connection Species Audio Detection Compepition 2번째 Post입니다.

<br>
<br>
<br>

### 0. 변경 내용

* 이번에는 지난 Post에서 다루었던 방법과 다른 방법으로 도전해 보고자 합니다.


* 지난번에 사용했던 Feature인 MEL Spectogram과 MFCC 중에서 **MEL Spec.** 만 사용하도록 하겠습니다.


* 지난 번에는 LightGBM을 사용했었지만, 이번에는 MEL Spec.을 이용해서 **CNN**을 적용해 보고자 합니다.


* 이번 Competition에서 제공되는 Dataset은 1216개 이고, 분류해야할 Class는 총 24개입니다.


* Dataset이 Deep Learning을 돌리기엔 너무나 부족한 수입니다.


* 이런 상황에서는 Overfitting이 발생할 확률이 매우 높기 때문에 Data Augmentation을 수행하기로 하겠습니다.


* CNN을 사용하기로 결정했고, 새롭게 CNN Layer 구조를 만들기 보다는 기존에 있는 훌륭한 Model을 가져와서 사용하기로 하겠습니다.


* 저는 이번에 **ResNet50**을 사용하도록 하겠습니다.


* Data Augmentation과 ResNet을 실시간으로 Training에 사용하기엔 Memory 제한때문에 이전에 다루었던 **Custom Generator**를 적용하도록 하겠습니다.

<br>
<br>
<br>
<br>
<br>
<br>

### 1. Augmentation for Sound Data 

* ImageDataGenerator에는 다양한 Augmentation 기법들이 마련되어 있지만, 사실 이 기법들은 모두 Image들에 대한 Augmentation 기법들입니다.


* 지금 우리가 다루는 Dataset은 CNN을 이용하는것은 맞지만, Image가 아닌 Sound Data입니다.


* 그래서, Image에 적용할 수 있는 Augmentation 기법들이 아닌 Sound에 적용가능한 Augmentation 기법들이 필요합니다.


* Sound Data Augmentation에 적용가능한 다양한 기법들이 소개된 좋은 글이 있어 소개드립니다.

  [RFCX: Audio Data Augmentation(Japanese+English)](https://www.kaggle.com/hidehisaarai1213/rfcx-audio-data-augmentation-japanese-english)

* 저도 이번 Dataset에는 위에서 소개한 다양한 기법들을 적용할 예정입니다.   


<br>
<br>
<br>

* Sound Data Augmentation을 위해서 'colorednoise'라는 Package를 사용하도록 하겠습니다.

* 자세한 소개와 사용법은 아래 Link를 참조해 주세요

  [colorednoise Github](https://github.com/felixpatzelt/colorednoise)
  
  [Package ‘colorednoise’](https://cran.r-project.org/web/packages/colorednoise/colorednoise.pdf)

* colorednoise Package를 설치합니다.   


```python
!pip install colorednoise 
```

    Collecting colorednoise
      Downloading colorednoise-1.1.1.tar.gz (5.2 kB)
    Requirement already satisfied: numpy in /opt/conda/lib/python3.7/site-packages (from colorednoise) (1.19.5)
    Building wheels for collected packages: colorednoise
      Building wheel for colorednoise (setup.py) ... [?25ldone
    [?25h  Created wheel for colorednoise: filename=colorednoise-1.1.1-py3-none-any.whl size=3958 sha256=65340b48a6e373c240574bcf7e108ed4683fa59c36e081904a9372abfdf57aef
      Stored in directory: /root/.cache/pip/wheels/1a/ca/93/dfd64286aef6fc206b727bd4cf2d5c17efe34d62b918c8f4a7
    Successfully built colorednoise
    Installing collected packages: colorednoise
    Successfully installed colorednoise-1.1.1
    
    
<br>
<br>
<br>

* 기타 필요한 Package를 모두 Load합니다.   


```python
import librosa
import numpy as np
import csv
from tqdm.notebook import tqdm
import pickle
from skimage.transform import resize
import pandas as pd

import tensorflow as tf

import librosa.display
import colorednoise as cn

from sklearn.model_selection import StratifiedKFold

from tensorflow.keras.models import Sequential
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout,GlobalAveragePooling2D,BatchNormalization
from tensorflow.keras import optimizers
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
```

   

   

   

* 이 문서는 Kaggle Notebook에서 만들어서 아래 항목은 자동으로 들어가는 내용이니 그냥 참고만 해주세요.   


```python
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        #print(os.path.join(dirname, filename))
        pass

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
```


<br>
<br>
<br>
<br>

* 아래 내용들은 Sound Augmentation 관련 내용들을 그대로 가져온 것입니다.   


```python
class AudioTransform:
    def __init__(self, always_apply=False, p=0.5):
        self.always_apply = always_apply
        self.p = p

    def __call__(self, y: np.ndarray):
        if self.always_apply:
            return self.apply(y)
        else:
            if np.random.rand() < self.p:
                return self.apply(y)
            else:
                return y

    def apply(self, y: np.ndarray):
        raise NotImplementedError


class Compose:
    def __init__(self, transforms: list):
        self.transforms = transforms

    def __call__(self, y: np.ndarray):
        for trns in self.transforms:
            y = trns(y)
        return y


class OneOf:
    def __init__(self, transforms: list):
        self.transforms = transforms

    def __call__(self, y: np.ndarray):
        n_trns = len(self.transforms)
        trns_idx = np.random.choice(n_trns)
        trns = self.transforms[trns_idx]
        return trns(y)
```

<br>
<br>

#### 1.1. GaussianNoiseSNR


```python
class GaussianNoiseSNR(AudioTransform):
    def __init__(self, always_apply=False, p=0.5, min_snr=5.0, max_snr=20.0, **kwargs):
        super().__init__(always_apply, p)

        self.min_snr = min_snr
        self.max_snr = max_snr

    def apply(self, y: np.ndarray, **params):
        snr = np.random.uniform(self.min_snr, self.max_snr)
        a_signal = np.sqrt(y ** 2).max()
        a_noise = a_signal / (10 ** (snr / 20))

        white_noise = np.random.randn(len(y))
        a_white = np.sqrt(white_noise ** 2).max()
        augmented = (y + white_noise * 1 / a_white * a_noise).astype(y.dtype)
        return augmented
```

<br>
<br>

#### 1.2.  PinkNoiseSNR


```python
class PinkNoiseSNR(AudioTransform):
    def __init__(self, always_apply=False, p=0.5, min_snr=5.0, max_snr=20.0, **kwargs):
        super().__init__(always_apply, p)

        self.min_snr = min_snr
        self.max_snr = max_snr

    def apply(self, y: np.ndarray, **params):
        snr = np.random.uniform(self.min_snr, self.max_snr)
        a_signal = np.sqrt(y ** 2).max()
        a_noise = a_signal / (10 ** (snr / 20))

        pink_noise = cn.powerlaw_psd_gaussian(1, len(y))
        a_pink = np.sqrt(pink_noise ** 2).max()
        augmented = (y + pink_noise * 1 / a_pink * a_noise).astype(y.dtype)
        return augmented        
```

<br>
<br>

#### 1.3. PitchShift


```python
class PitchShift(AudioTransform):
    def __init__(self, always_apply=False, p=0.5, max_steps=5, sr=32000):
        super().__init__(always_apply, p)

        self.max_steps = max_steps
        self.sr = sr

    def apply(self, y: np.ndarray, **params):
        n_steps = np.random.randint(-self.max_steps, self.max_steps)
        augmented = librosa.effects.pitch_shift(y, sr=self.sr, n_steps=n_steps)
        return augmented        
```

<br>
<br>

#### 1.4. TimeShift


```python
class TimeShift(AudioTransform):
    def __init__(self, always_apply=False, p=0.5, max_shift_second=2, sr=32000, padding_mode="replace"):
        super().__init__(always_apply, p)
    
        assert padding_mode in ["replace", "zero"], "`padding_mode` must be either 'replace' or 'zero'"
        self.max_shift_second = max_shift_second
        self.sr = sr
        self.padding_mode = padding_mode

    def apply(self, y: np.ndarray, **params):
        shift = np.random.randint(-self.sr * self.max_shift_second, self.sr * self.max_shift_second)
        augmented = np.roll(y, shift)
        if self.padding_mode == "zero":
            if shift > 0:
                augmented[:shift] = 0
            else:
                augmented[shift:] = 0
        return augmented
```

<br>
<br>

#### 1.5. TimeShift


```python
class VolumeControl(AudioTransform):
    def __init__(self, always_apply=False, p=0.5, db_limit=10, mode="uniform"):
        super().__init__(always_apply, p)

        assert mode in ["uniform", "fade", "fade", "cosine", "sine"], \
            "`mode` must be one of 'uniform', 'fade', 'cosine', 'sine'"

        self.db_limit= db_limit
        self.mode = mode

    def apply(self, y: np.ndarray, **params):
        db = np.random.uniform(-self.db_limit, self.db_limit)
        if self.mode == "uniform":
            db_translated = 10 ** (db / 20)
        elif self.mode == "fade":
            lin = np.arange(len(y))[::-1] / (len(y) - 1)
            db_translated = 10 ** (db * lin / 20)
        elif self.mode == "cosine":
            cosine = np.cos(np.arange(len(y)) / len(y) * np.pi * 2)
            db_translated = 10 ** (db * cosine / 20)
        else:
            sine = np.sin(np.arange(len(y)) / len(y) * np.pi * 2)
            db_translated = 10 ** (db * sine / 20)
        augmented = y * db_translated
        return augmented        
```


<br>
<br>
<br>
<br>

* 아래 상수들은 이전 Revision과 거의 동일하게 사용하지만, Sound File을 Crop하는 길이를 6초에서 10초로 늘렸습니다.   


```python
SLICE_TIME_WINDOWS = 10

fft = 2048
hop = 512

# Less rounding errors this way
sr = 48000
length = SLICE_TIME_WINDOWS * sr

F_MIN = 84
F_MAX = 15056

NUM_CLASS = 24
```

<br>
<br>
<br>
<br>

* 위에서 정의한 다양한 Sound Augmentation 기법들을 Random하게 적용하도록 해줍니다.   


```python
transform = Compose([
  OneOf([
      GaussianNoiseSNR(always_apply=True, min_snr=5, max_snr=20),
      PinkNoiseSNR(always_apply=True, min_snr=5.0, max_snr=20.0)
  ]),
    PitchShift(always_apply=True, max_steps=5, sr=sr),
    TimeShift(always_apply=True, max_shift_second=4, sr=sr),
    VolumeControl(always_apply=True, mode="sine")
])
```

<br>
<br>
<br>
<br>
<br>
<br>
<br>
<br>

### 2. Custom Generator

* 이전 Post [Custom Generator](https://moonlight314.github.io/deeplearning/Custom_Generator/)에서 다루었던 사항들을 이번에 적용해 보도록 하겠습니다.


* 이번에는 Train에 사용할 Generator와 Validation에서 사용할 Generator를 따로 만들도록 하겠습니다.


* 두 Generator의 차이점은 **Augmentation을 적용하느냐 하지 않느냐의 차이**만 있을 뿐 다른 차이는 없습니다.


* 일반적으로 Validation Dataset에는 Augmentation을 적용하지 않습니다.


* 'train_gen' option이 Train과 Validation을 구분하는 Parameter입니다. 

   


```python
class DataGenerator(tf.keras.utils.Sequence):
    
    'Generates data for Keras'
    def __init__(self, 
                 list_IDs, 
                 labels, 
                 batch_size=32, 
                 dim=(128,563), 
                 n_channels=1,
                 t_min=[],
                 t_max=[],
                 recording_id=[],
                 train=True,
                 n_classes=24, 
                 shuffle=True):
        
        'Initialization'
        self.dim = dim
        self.batch_size = batch_size
        self.labels = labels
        self.list_IDs = list_IDs
        self.n_channels = n_channels
        self.t_min = t_min
        self.t_max = t_max
        self.recording_id = recording_id
        self.n_classes = n_classes
        self.train_gen = train
        self.shuffle = shuffle
        
        self.n = 0
        self.max = self.__len__()

        self.on_epoch_end()
        

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        # Generate data
        X, y = self.__data_generation(list_IDs_temp)

        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)
            
    
    def __next__(self):
        if self.n >= self.max:
           self.n = 0
        result = self.__getitem__(self.n)
        self.n += 1
        return result
    

    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        X = np.empty((self.batch_size, *self.dim, self.n_channels))
        y = np.empty((self.batch_size), dtype=int)

        
        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            # Store sample
            wav, sr = librosa.load('/kaggle/input/rfcx-species-audio-detection/train/' + recording_id[ID] + '.flac', sr=None)

            t_start = self.t_min[ID] * sr
            t_end = self.t_max[ID] * sr
            
            # Positioning sound slice
            center = np.round((t_start + t_end) / 2)
            beginning = center - length / 2
            
            if beginning < 0:
                beginning = 0                
            
            ending = beginning + length
            
            if ending > len(wav):
                ending = len(wav)
                beginning = ending - length
            
            slice = wav[int(beginning):int(ending)]

            if self.train_gen == True:
                aug_slice = transform( slice )
            else:
                aug_slice = slice
                
        
            mel_spec = librosa.feature.melspectrogram(aug_slice, n_fft=fft, hop_length=hop, sr=sr, fmin=F_MIN, fmax=F_MAX, power=2)    
            mel_spec = librosa.core.amplitude_to_db(np.abs(mel_spec))

            mel_spec = mel_spec - np.min(mel_spec)
            mel_spec = mel_spec / np.max(mel_spec)   
            
            stacked = resize(mel_spec , (self.dim[0], self.dim[1], self.n_channels))
            X[i,] = stacked

            
            # Store class
            y[i] = self.labels[ID]
        
        return X, tf.keras.utils.to_categorical(y, num_classes=self.n_classes)
```


<br>
<br>
<br>
<br>

* 이전 방법과 유사하게 진행하도록 하겠습니다.


* train_tp.csv에서 Train Data 관련 Meta 정보를 읽어서 준비합니다.


```python
data = pd.read_csv("/kaggle/input/rfcx-species-audio-detection/train_tp.csv")
data.head()
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
      <th>0</th>
      <td>003bec244</td>
      <td>14</td>
      <td>1</td>
      <td>44.5440</td>
      <td>2531.250</td>
      <td>45.1307</td>
      <td>5531.25</td>
    </tr>
    <tr>
      <th>1</th>
      <td>006ab765f</td>
      <td>23</td>
      <td>1</td>
      <td>39.9615</td>
      <td>7235.160</td>
      <td>46.0452</td>
      <td>11283.40</td>
    </tr>
    <tr>
      <th>2</th>
      <td>007f87ba2</td>
      <td>12</td>
      <td>1</td>
      <td>39.1360</td>
      <td>562.500</td>
      <td>42.2720</td>
      <td>3281.25</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0099c367b</td>
      <td>17</td>
      <td>4</td>
      <td>51.4206</td>
      <td>1464.260</td>
      <td>55.1996</td>
      <td>4565.04</td>
    </tr>
    <tr>
      <th>4</th>
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

* Generator를 생성하기 위해서 필요한 값들을 미리 뽑아놓겠습니다.   


```python
recording_id = data['recording_id'].to_list()
labels = data['species_id'].to_list()
t_min = data['t_min'].to_list()
t_max = data['t_max'].to_list()
list_IDs = [i for i in range(len(recording_id))]
```

<br>
<br>
<br>

* Train Set과 Validation Set을 2:1 비율로 나누겠습니다.


* Label의 균등한 배분을 위해 StratifiedKFold를 사용하도록 하겠습니다.


* Data가 너무 적어서 걱정이네요...


```python
skf = StratifiedKFold(n_splits=3 , random_state=27 , shuffle=True)
```


```python
for train_index, val_index in skf.split(list_IDs , labels):
    print(len(train_index) , len(val_index))
```

    810 406
    811 405
    811 405
    
<br>
<br>
<br>

* StratifiedKFold로 나눈 값들을 Train & Validation Generator에 넣을 값을 각각 분리합니다.


```python
recording_id_Train = [recording_id[k] for k in train_index]
recording_id_Val = [recording_id[k] for k in val_index]

Label_Train = [labels[k] for k in train_index]
Label_Val = [labels[k] for k in val_index]

t_min_Train = [t_min[k] for k in train_index]
t_min_Val = [t_min[k] for k in val_index]

t_max_Train = [t_max[k] for k in train_index]
t_max_Val = [t_max[k] for k in val_index]

list_IDs_Train = [k for k in range(len(train_index))]
list_IDs_Val = [k for k in range(len(val_index))]
```

<br>
<br>
<br>

Train과 Validation Generator의 차이는 Augmentation을 적용하느냐 마느냐의 차이뿐입니다.

```python
# Train Data Generatpr
train_params = {'dim': (128,1876),
          'batch_size': 32,
          'n_classes': 24,
          'n_channels': 1,
          't_min': t_min_Train,
          't_max': t_max_Train,
          'recording_id' : recording_id_Train,
          'train' : True,
          'shuffle': True}

train_generator = DataGenerator(list_IDs = list_IDs_Train, labels = Label_Train, **train_params)
```
<br>
<br>
<br>

```python
# Train Data Generatpr
val_params = {'dim': (128,1876),
          'batch_size': 32,
          'n_classes': 24,
          'n_channels': 1,
          't_min': t_min_Val,
          't_max': t_max_Val,
          'recording_id' : recording_id_Val,
          'train' : False,
          'shuffle': True}

val_generator = DataGenerator(list_IDs = list_IDs_Val, labels = Label_Val, **val_params)
```

<br>
<br>
<br>
<br>
<br>
<br>

### 3. Metric

* 이번 Competition에는 흔히 많이 쓰이는 Metric이 아닌 **LWLRAP(Label Weighted Label Ranking Average Precision)** 이라는 Evaluation Metric을 사용합니다.

<br>
<br>
<p align="center">
  <img src="/assets/RFCX_02/pic_00.png">
</p>
<br>
<br>

* 이를 잘 설명해 놓은 글이 있어서 소개해 드리겠습니다.   

https://www.kaggle.com/pkmahan/understanding-lwlrap

* Scikit-Learn Site에도 관련된 설명을 찾을 수 있었습니다.   

https://scikit-learn.org/stable/modules/model_evaluation.html#label-ranking-average-precision      

<br>
<br>   
<br>
<br>
   

* 저도 위의 글에서 Code를 가져와서 Montoring에 사용하도록 하겠습니다.   

<br>
<br>   
<br>
<br>

```python
def _one_sample_positive_class_precisions(example):
    y_true, y_pred = example
    y_true = tf.reshape(y_true, tf.shape(y_pred))
    retrieved_classes = tf.argsort(y_pred, direction='DESCENDING')
#     shape = tf.shape(retrieved_classes)
    class_rankings = tf.argsort(retrieved_classes)
    retrieved_class_true = tf.gather(y_true, retrieved_classes)
    retrieved_cumulative_hits = tf.math.cumsum(tf.cast(retrieved_class_true, tf.float32))

    idx = tf.where(y_true)[:, 0]
    i = tf.boolean_mask(class_rankings, y_true)
    r = tf.gather(retrieved_cumulative_hits, i)
    c = 1 + tf.cast(i, tf.float32)
    precisions = r / c

    dense = tf.scatter_nd(idx[:, None], precisions, [y_pred.shape[0]])
    return dense

# @tf.function
class LWLRAP(tf.keras.metrics.Metric):
    def __init__(self, num_classes, name='lwlrap'):
        super().__init__(name=name)

        self._precisions = self.add_weight(
            name='per_class_cumulative_precision',
            shape=[num_classes],
            initializer='zeros',
        )

        self._counts = self.add_weight(
            name='per_class_cumulative_count',
            shape=[num_classes],
            initializer='zeros',
        )

    def update_state(self, y_true, y_pred, sample_weight=None):
        precisions = tf.map_fn(
            fn=_one_sample_positive_class_precisions,
            elems=(y_true, y_pred),
            dtype=(tf.float32),
        )

        increments = tf.cast(precisions > 0, tf.float32)
        total_increments = tf.reduce_sum(increments, axis=0)
        total_precisions = tf.reduce_sum(precisions, axis=0)

        self._precisions.assign_add(total_precisions)
        self._counts.assign_add(total_increments)        

    def result(self):
        per_class_lwlrap = self._precisions / tf.maximum(self._counts, 1.0)
        per_class_weight = self._counts / tf.reduce_sum(self._counts)
        overall_lwlrap = tf.reduce_sum(per_class_lwlrap * per_class_weight)
        return overall_lwlrap

    def reset_states(self):
        self._precisions.assign(self._precisions * 0)
        self._counts.assign(self._counts * 0)
```

<br>
<br>
<br>
<br>
<br>
<br>
<br>
<br>

### 4. Model   

* 자, 이제 Train에 사용할 Model을 구성해 보도록 하겠습니다.   


```python
model = Sequential()
```
   
<br>
<br>
<br>
<br>
   

* 처음에 말씀드렸듯이, ResNet50을 사용하도록 하겠습니다.   

* Tensorflow에서는 ResNet50을 제공해 주고 있는데, 아래 Link에서 자세한 내용을 확인할 수 있습니다.

  [ResNet50 in Tensorflow](https://www.tensorflow.org/api_docs/python/tf/keras/applications/ResNet50)
  
  
* ResNet의 종류는 규모(?)에 따라서 다양한데, 가장 만만한 ResNet50을 사용해 보도록 하겠습니다.


* ResNet50은 기본적으로 ImageNet Dataset으로 Training되었습니다. 즉, Feature Extraction을 할 때 ImageNet과 유사한 Image에 대해서는 잘 할지 몰라도, 이번 Competition에서 사용하는 MEL Spec.와 같은 Image에는 맞지 않을 듯 합니다.


* 그래서, 이번에는 Weight를 Load하지 않고, ResNet50의 Structure만 빌려오도록 하겠습니다.


* 그리고, ResNet50의 Input Shape은 (224,224,3)이지만, MEL Spec.은 Shape이 다르기 때문에 Input Shape을 재정의하였습니다.

<br>
<br>
<br>

```python
model.add( ResNet50(include_top = False, 
                    input_shape=(256, 1876, 1),
                    weights = None ))
```

<br>
<br>
<br>

* Extracted Feature를 받아서 분류할 Simple MLP를 추가합니다.   


```python
model.add(GlobalAveragePooling2D()) 

model.add(BatchNormalization())
model.add(Dropout(0.3))
model.add(Dense(1024, activation='relu' , kernel_initializer = tf.keras.initializers.he_normal()))

model.add(BatchNormalization())
model.add(Dropout(0.3))
model.add(Dense(NUM_CLASS, activation = 'softmax' , kernel_initializer = tf.keras.initializers.he_normal()))
```


<br>
<br>
<br>

* 최종적으로 다음과 같은 구조를 가지게 되었습니다.


* ResNet50 전체의 Weight를 전부 Train시키기 때문에 Trainable params 수가 많네요.


```python
model.summary()
```

    Model: "sequential"
    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    resnet50 (Functional)        (None, 8, 59, 2048)       23581440  
    _________________________________________________________________
    global_average_pooling2d (Gl (None, 2048)              0         
    _________________________________________________________________
    batch_normalization (BatchNo (None, 2048)              8192      
    _________________________________________________________________
    dropout (Dropout)            (None, 2048)              0         
    _________________________________________________________________
    dense (Dense)                (None, 1024)              2098176   
    _________________________________________________________________
    batch_normalization_1 (Batch (None, 1024)              4096      
    _________________________________________________________________
    dropout_1 (Dropout)          (None, 1024)              0         
    _________________________________________________________________
    dense_1 (Dense)              (None, 24)                24600     
    =================================================================
    Total params: 25,716,504
    Trainable params: 25,657,240
    Non-trainable params: 59,264
    _________________________________________________________________
    
<br>
<br>
<br>

* Optimizer는 Adam을 사용하도록 하겠습니다.


* Evaluation Metric은 이전에 소개해드린 LWLRAP를 사용하도록 하겠습니다.


```python
sgd = optimizers.SGD(lr = 0.01, decay = 1e-6, momentum = 0.9, nesterov = True)
adam = optimizers.Adam(lr = 0.001)

model.compile(optimizer = adam, 
              loss = 'categorical_crossentropy', 
              metrics = [LWLRAP(num_classes = NUM_CLASS)],
             )
```
<br>
<br>
<br>

* Callback은 Validation Set에서 LWLRAP값이 가장 높은 것을 저장하도록 하겠습니다.


```python
# set model callbacks to save best model
filepath = "Saved-model-best.hdf5"

my_callbacks = [ModelCheckpoint(filepath, 
                                monitor='val_lwlrap', 
                                verbose=1,
                                save_best_only=True, 
                                mode='max')]
```

<br>
<br>
<br>
* 이제 대망의 Train시작입니다 !


* Generator를 사용하지만, Kaggle Notebook Env.는 TF 2.xx를 사용하고 있기 때문에 그냥 .fit으로도 Train이 됩니다.


* 일단 10번만 돌려보도록 하겠습니다.


```python
#history = model.fit_generator(
history = model.fit(
        train_generator,
        epochs = 10,
        validation_data=val_generator,
        callbacks=my_callbacks,
        verbose=1
)
```

    Epoch 1/10
    25/25 [==============================] - 858s 34s/step - loss: 3.8363 - lwlrap: 0.1664 - val_loss: 3.3410 - val_lwlrap: 0.1465
    
    Epoch 00001: val_lwlrap did not improve from 0.15455
    Epoch 2/10
    25/25 [==============================] - 848s 34s/step - loss: 3.8105 - lwlrap: 0.1541 - val_loss: 3.2576 - val_lwlrap: 0.1667
    
    Epoch 00002: val_lwlrap improved from 0.15455 to 0.16667, saving model to Saved-model-best.hdf5
    Epoch 3/10
    25/25 [==============================] - 863s 35s/step - loss: 3.7783 - lwlrap: 0.1677 - val_loss: 3.3092 - val_lwlrap: 0.1631
    
    Epoch 00003: val_lwlrap did not improve from 0.16667
    Epoch 4/10
    25/25 [==============================] - 872s 35s/step - loss: 3.8039 - lwlrap: 0.1689 - val_loss: 3.4925 - val_lwlrap: 0.1941
    
    Epoch 00004: val_lwlrap improved from 0.16667 to 0.19413, saving model to Saved-model-best.hdf5
    Epoch 5/10
    25/25 [==============================] - 869s 35s/step - loss: 3.7036 - lwlrap: 0.1813 - val_loss: 3.3664 - val_lwlrap: 0.1477
    
    Epoch 00005: val_lwlrap did not improve from 0.19413
    Epoch 6/10
    25/25 [==============================] - 862s 35s/step - loss: 3.6492 - lwlrap: 0.1848 - val_loss: 3.2578 - val_lwlrap: 0.1963
    
    Epoch 00006: val_lwlrap improved from 0.19413 to 0.19625, saving model to Saved-model-best.hdf5
    Epoch 7/10
    25/25 [==============================] - 855s 34s/step - loss: 3.7575 - lwlrap: 0.1706 - val_loss: 3.3313 - val_lwlrap: 0.1642
    
    Epoch 00007: val_lwlrap did not improve from 0.19625
    Epoch 8/10
    25/25 [==============================] - 878s 35s/step - loss: 3.6909 - lwlrap: 0.1683 - val_loss: 3.3364 - val_lwlrap: 0.1547
    
    Epoch 00008: val_lwlrap did not improve from 0.19625
    Epoch 9/10
    25/25 [==============================] - 867s 35s/step - loss: 3.6804 - lwlrap: 0.1811 - val_loss: 3.3591 - val_lwlrap: 0.1454
    
    Epoch 00009: val_lwlrap did not improve from 0.19625
    Epoch 10/10
    25/25 [==============================] - 877s 35s/step - loss: 3.6514 - lwlrap: 0.1711 - val_loss: 3.2316 - val_lwlrap: 0.1690
    
    Epoch 00010: val_lwlrap did not improve from 0.19625
    

<br>
<br>
<br>

* 10회 정도만 돌려보았습니다만, 결과적으로 성공적이지는 않는것 같습니다.
<br>

* 우선, 다행스럽게도 Train Set에서의 LWLRAP값과 Validation Set에서의 LWLRAP 값이 모두 비슷하게 나오고 있는 것으로 보아, 가장 걱정했던 Overfitting은 발생하지 않는 것 같습니다.
<br>

* 아마도 Data Augmentation이나 몇몇 장치들이 영향을 준 것 같습니다.
<br>

* 하지만,결론적으로 이 Model은 사용하지 못할 것 같습니다.  
<br>

* 먼저 LWLRAP값이 Training을 진행해도 크게 개선되지 않고 있습니다. 그리고 LWLRAP 값 자체가 너무 낮습니다. ( LWLRAP 값은 1.00이 최고값입니다. )
<br>

* 사실, 이 Model은 17번째이며, 이전에 다양한 방법과 Feature들로 Test해 본 결과, 가장 큰 문제는 적은 양의 Dataset때문에 Overfitting이 가장 큰 문제였습니다.
<br>

* Overfitting은 Sound Augmentation으로 어느 정도 해결이 된 것 같으나, 낮은 LWLRAP값은 여전히 문제입니다.
<br>

* Algorithm이나 Model의 문제라기 보다는 Feature의 문제로 보이며, 이를 개선하기 위해서 좀 더 Study를 해 봐야 할 것 같습니다.
<br>

* ResNet의 Weight의 수가 많아서인지 Local에서의 Train은 Resource를 많이 잡아먹는 경향을 보였고, GPU에서도 Train 시간도 많이 걸렸습니다.

  ( 이 Notebook도 Kaggle에서 제공하는 GPU에서 돌렸습니다. )
<br>  
  
* 대용량 Dataset이 Kaggle에서도 많아지는 경향이어서 최근 Competition에서는 TFRecord Format도 같이 제공해주는 Competition이 많아지고 있습니다.
<br>

* 이는 다분히 Tensorflow & TPU 조합을 사용하라는 배려(혹은 압박)라고 생각합니다.
<br>

* 다음에 시간되면 TFRecord를 다루는 방법에 대해서도 한 번 알아보도록 하겠습니다.
<br>

* 이번 Post는 여기까지 하도록 하겠습니다.  도움이 되셨기를 바랍니다.

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



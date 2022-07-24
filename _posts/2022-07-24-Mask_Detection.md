---
title: "Mask Detection"
date: 2021-10-29 08:26:28 -0400
categories: Deep Learning
---
# Mask Detection

<br>
<br>
<br>

[이 Project의 Github Repository Link](https://github.com/MoonLight314/Mask_Detection)

<br>
<br>
<br>

## 0. Introduction

<br>
<br>

### 0.0. Motivation

<br>

* COVID-19 상황속에서 Deep Learning을 이용하여 RGB Cam.으로 실시간으로 Mask 착용 여부를 확인할 수 있는 Model을 만들어 보겠습니다.

<br>      

### 0.1. Face Detector

* Dataset에서 사람 얼굴 부분만을 추출하기 위해서 Face Detector를 사용하여야 합니다.

* 우선 사람의 얼굴부분만을 빠르게 Detecting할 수 있는 Model을 찾아보았고, 최종적으로 Tensorflow와 호환이 잘되는 OpenCV DNN Face Detector를 사용하기로 했습니다.

* DNN Face Detector in OpenCV [https://learnopencv.com/face-detection-opencv-dlib-and-deep-learning-c-python/]
  - Face Detector는 Input으로 Image를 넣어주면, 해당 Image에서 사람 얼굴이라고 판단되는 영역의 정보와 확신(신뢰)도를 값으로 Return해 줍니다.
  - Pre-Trained Model을 받아서 사용하면 됩니다.
  - 다음 2개의 File을 이용합니다.
    - MODEL_FILE : opencv_face_detector_uint8.pb
    - CONFIG_FILE : opencv_face_detector.pbtxt
  - 속도도 빠르고 성능도 좋아서 Face Detection에 이 Module을 사용하도록 하겠습니다.
  - 자세한 사용 방법은 이후 Code로 살펴보겠습니다.

<br>
<br>

### 0.2. Dataset

<br>

* Train에 사용할 Dataset은 아래 2가지를 사용하도록 하겠습니다.

 - https://github.com/chandrikadeb7/Face-Mask-Detection/tree/master/dataset 
 
 - https://www.kaggle.com/datasets/andrewmvd/face-mask-detection
     - License(CC0: Public Domain)(https://creativecommons.org/publicdomain/zero/1.0/)

<br>

* 순서
  
  1) 2개의 Dataset은 공통적으로 모두 Mask를 쓴 사람들의 사진과 쓰지 않은 사람들의 사진을 가지고 있지만, Mask 착용 여부 / 사람 얼굴 위치등의 정보를 나타내는 방법은 조금 다릅니다.
  
  2) 최종적으로 ResNet으로 분류를 하기 위한 Dataset으로 Preprocessing을 하는 것이 목적이므로 각 Dataset마다 다른 Preprocessing 방법을 적용하도록 하겠습니다.

<br>
<br>
<br>

## 1. Preprocess      

<br>
<br>

### 1.0. 순서
  
  * 2개의 Dataset은 공통적으로 모두 Mask를 쓴 사람들의 사진과 쓰지 않은 사람들의 사진을 가지고 있지만, Mask 착용 여부 / 사람 얼굴 위치등의 정보를 나타내는 방법은 조금 다릅니다.
  
  
  * 최종적으로 ResNet으로 분류를 하기 위한 Dataset으로 Preprocessing을 하는 것이 목적이므로 각 Dataset마다 다른 Preprocessing 방법을 적용하도록 하겠습니다.      

<br>
<br>

### 1.1. 첫번째 Dataset
  * 이 Dataset은 총 4095장의 Image가 있고, Mask쓴 사람의 Image가 2165장, 쓰지 않는 사람의 Image가 1930장으로 구성되어 있다.
  * Image에는 얼굴만 나오는 사진도 있지만, 사람 몸 전체가 나오는 경우도 있기 때문에 앞서 소개한 OpenCV DNN Face Detector를 이용하여 얼굴부분에 대한 정보만 추출하여 Train에 사용하도록 하겠습니다.
  * Label은 Folder Name으로 알 수 있습니다.
  * 최종적으로 File Path / Mask 착용 여부 / 얼굴부분의 좌표 정보를 추출하여 Pandas Dataframe으로 저장하는 것을 목표로 하겠습니다.

<br>
<br>

```python
import numpy as np
import pandas as pd
import os
import glob
import cv2
from tqdm import tqdm
import tensorflow as tf
import xml.etree.ElementTree as et
```

<br>

* OpenCV DNN Face Detector 관련 상수를 정의합니다.

<br>

* **'CONFIDENCE_FACE = 0.9'**
  - Face Detector가 제공해주는 값으로 추출한 얼굴 부분이 어느 정도 신뢰도가 있는지 나타내주는 값입니다.

<br>

* **'MARGIN_RATIO = 0.2'**
  - Face Detector는 딱 얼굴 부분만 추출하기 때문에 상하좌우 조금 더 여유를 주기 위한 값입니다.

<br>

```python
MODEL_FILE = "opencv_face_detector_uint8.pb"
CONFIG_FILE = "opencv_face_detector.pbtxt"
SIZE = 300
CONFIDENCE_FACE = 0.9
MARGIN_RATIO = 0.2
```

<br>

* 먼저 Folder내에 있는 Image File들의 Full Path List를 만듭니다.      

<br>

```python
def save_file_fist():
    dir_path = "./Dataset/Face-Mask-Detection-master"
    data_file_path = []

    for (root, directories, files) in tqdm(os.walk(dir_path)):
        for file in files:
            file_path = os.path.join(root, file)
            data_file_path.append( file_path )


    return data_file_path
```

<br>

* 이후에 이 File List를 가지고 추가 작업을 하기 때문에 이 작업은 가장 먼저 수행되어야 합니다.      

<br>

```python
data_file_path = save_file_fist()
```

    3it [00:00, 142.83it/s]

<br>
<br>

* OpenCV Face Dectector로 각 Image에서 얼굴 부분에 대한 좌표값을 추출하는 부분입니다.      

<br>

```python
def get_face_coor_info(file_path):
    
    pass_file_list = []
    lefts = []
    tops = []
    rights = []
    bottoms = []
    masks = []
    low_confidence_cnt = 0

    net = cv2.dnn.readNetFromTensorflow( MODEL_FILE , CONFIG_FILE )

    for file in tqdm(file_path):

        try:
            img = cv2.imread(file)
            rows, cols, channels = img.shape
            blob = cv2.dnn.blobFromImage(img, 1.0)

            net.setInput(blob)
            detections = net.forward()

            detection = detections[0, 0]    
            i = np.argmax(detection[:,2])

            if i != 0:
                print(file , "Max index is not 0")
                continue

            if detection[i,2] < CONFIDENCE_FACE:
                #print(file , "Low CONFIDENCE_FACE" , detection[i,2])
                low_confidence_cnt += 1
                continue
            

            left = detection[i,3] * cols
            top = detection[i,4] * rows
            right = detection[i,5] * cols            
            bottom = detection[i,6] * rows

            left = int(left - int((right - left) * MARGIN_RATIO))
            top = int(top - int((bottom - top) * MARGIN_RATIO))
            right = int(right + int((right - left) * MARGIN_RATIO))
            bottom = int(bottom + int((bottom - top) * MARGIN_RATIO / 2))

            if left < 0:
                left = 0

            if right > cols:
                right = cols

            if top < 0:
                top = 0

            if bottom > rows:
                bottom = rows

            pass_file_list.append(file)
            lefts.append(left)
            tops.append(top)
            rights.append(right)
            bottoms.append(bottom)

            if "with_mask" in file:
                masks.append("with_mask")
            elif "without_mask" in file:
                masks.append("without_mask")
        
        except:
            print(file , " Error")


    print(len(pass_file_list))
    print("No. of Low Confidence : ",low_confidence_cnt)

    result = pd.DataFrame(list(zip(pass_file_list, masks , lefts , tops , rights , bottoms)), columns=['file_path','mask','xmin','ymin','xmax','ymax'])

    result = result.astype({    'xmin':'int32', 
                                'ymin':'int32',
                                'xmax':'int32', 
                                'ymax':'int32',
                                })

    return result
```

<br>
<br>

```python
meta_data_01 = get_face_coor_info( data_file_path )
```

     64%|█████████████████████████████████████████████████▎                           | 2622/4095 [00:59<00:14, 103.21it/s]

    ./Dataset/Face-Mask-Detection-master\with_mask\0_0__˙_￠ 2020-02-23 132115.png  Error
    ./Dataset/Face-Mask-Detection-master\with_mask\0_0__˙_￠ 2020-02-23 132400.png  Error
    ./Dataset/Face-Mask-Detection-master\with_mask\0_0__˙_￠ 2020-02-24 171804.png  Error
    ./Dataset/Face-Mask-Detection-master\with_mask\0_0__˙_￠ 2020-02-24 172039.png  Error
    ./Dataset/Face-Mask-Detection-master\with_mask\0_0__˙_￠ 2020-02-24 202509.png  Error
    ./Dataset/Face-Mask-Detection-master\with_mask\0_0__˙_￠ 2020-02-24 205216.png  Error
    ./Dataset/Face-Mask-Detection-master\with_mask\0_0__˙_￠ 2020-02-24 215234.png  Error
    ./Dataset/Face-Mask-Detection-master\with_mask\0_0__˙_￠ 2020-02-24 215615.png  Error
    ./Dataset/Face-Mask-Detection-master\with_mask\0_0__˙_￠ 2020-02-24 220536.png  Error
    ./Dataset/Face-Mask-Detection-master\with_mask\0_0__˙_￠ 2020-02-24 222124.png  Error
    ./Dataset/Face-Mask-Detection-master\with_mask\0_0__˙_￠ 2020-02-24 224833.png  Error
    ./Dataset/Face-Mask-Detection-master\with_mask\0_0__˙_￠ 2020-02-24 225329.png  Error
    ./Dataset/Face-Mask-Detection-master\with_mask\0_0__˙_￠ 2020-02-24 225427.png  Error
    ./Dataset/Face-Mask-Detection-master\with_mask\0_0__˙_￠ 2020-02-25 150422.png  Error
    ./Dataset/Face-Mask-Detection-master\with_mask\0_0__˙_￠ 2020-02-25 150847.png  Error
    ./Dataset/Face-Mask-Detection-master\with_mask\0_0__˙_￠ 2020-02-25 150921.png  Error
    ./Dataset/Face-Mask-Detection-master\with_mask\0_0__˙_￠ 2020-02-25 185823.png  Error
    ./Dataset/Face-Mask-Detection-master\with_mask\0_0__˙_￠ 2020-02-25 190026.png  Error
    ./Dataset/Face-Mask-Detection-master\with_mask\0_0_œ￢‘y.png  Error
    

     73%|████████████████████████████████████████████████████████▋                     | 2973/4095 [01:08<00:16, 69.06it/s]

    ./Dataset/Face-Mask-Detection-master\with_mask\1_0__˙_￠ 2020-02-24 202935.png  Error
    ./Dataset/Face-Mask-Detection-master\with_mask\1_0__˙_￠ 2020-02-24 215624.png  Error
    ./Dataset/Face-Mask-Detection-master\with_mask\1_0__˙_￠ 2020-02-24 224914.png  Error
    ./Dataset/Face-Mask-Detection-master\with_mask\1_0__˙_￠ 2020-02-25 151918.png  Error
    

    100%|██████████████████████████████████████████████████████████████████████████████| 4095/4095 [03:49<00:00, 17.88it/s]

    3197
    No. of Low Confidence :  875

<br>
<br>

* 간혹 File Name에 특수 문자가 있는 경우 Open하지 못하는 경우도 있어서 875장은 사용하지 못했습니다.

<br>

* 최종적으로 첫번째 Dataset에서는 총 3197장의 유효한 Data를 얻었습니다.

<br>
<br>

### 1.2. 두번째 Dataset

<br>

   * 이 Dataset은 총 853장의 Image가 있습니다.

<br>   

   * 이 Dataset은 Image File이름과 동일한 XML File을 제공해 주고 있으며, 각 XML File에는 Image File Name / 얼굴부분의 좌표 정보 / Mask 착용여부에 대한 정보가 모두 들어 있습니다.

<br>

   * 즉, XML File Decoding만 잘 해주면 모든 정보를 얻을 수 있습니다.

<br>

```python
def preprocessing_Face_Mask_Detection_Dataset_Kaggle():
    dir_path = "./Dataset/Face_Mask_Detection_Dataset_Kaggle/annotations/"
    image_dir_path = "./Dataset/Face_Mask_Detection_Dataset_Kaggle/images/"
    data_file_path = []

    for (root, directories, files) in tqdm(os.walk(dir_path)):
        for file in files:
            if '.xml' in file:
                file_path = os.path.join(root, file)
                data_file_path.append( file_path )

    meta_data = pd.DataFrame({"file_path":[], 
                            "mask":[],
                            "xmin":[],
                            "ymin":[],
                            "xmax":[],
                            "ymax":[]
                            })

    for path in tqdm(data_file_path):

        xtree=et.parse( path )
        xroot=xtree.getroot()

        mask_flag = []
        xmin = []
        ymin = []
        xmax = []
        ymax = []

        for node in xroot:
            
            if node.tag == 'filename':
                fname = os.path.join(image_dir_path , node.text)

            if node.tag == 'object':
                name = node.find("name")
                mask_flag.append( name.text )

                box = node.find("bndbox")
                
                t = box.find("xmin")
                if t != None:
                    xmin.append( t.text )

                t = box.find("ymin")
                if t != None:
                    ymin.append( t.text )

                t = box.find("xmax")
                if t != None:
                    xmax.append( t.text )

                t = box.find("ymax")
                if t != None:
                    ymax.append( t.text )
                


        file_name = [fname] * len(xmin)

        tmp = pd.DataFrame({"file_path":file_name , 
                            "mask":mask_flag,
                            "xmin":xmin,
                            "ymin":ymin,
                            "xmax":xmax,
                            "ymax":ymax
                            })

        meta_data = pd.concat( [meta_data,tmp] )

    meta_data = meta_data.astype({  'xmin':'int32', 
                                    'ymin':'int32',
                                    'xmax':'int32', 
                                    'ymax':'int32',
                                    })

    return meta_data
```

<br>
<br>

```python
meta_data_02 = preprocessing_Face_Mask_Detection_Dataset_Kaggle()
```

    1it [00:00, 199.98it/s]
    100%|███████████████████████████████████████████████████████████████████████████████| 853/853 [00:05<00:00, 148.33it/s]
    
<br>
<br>

* 이제 2개의 Dataset에서 Train에 필요한 모든 정보를 얻었기 때문에 하나로 합칩니다.      


```python
meta_data = pd.concat([meta_data_01 , meta_data_02])
```

<br>
<br>

* 마지막으로 실제로 Train시에 사용할 ResNet에 입력시에 문제가 없는지 확인하도록 하겠습니다.
* 실제 Train을 하다보면 Image를 열지 못하거나 다양한 이유로 Train이 중단되어버리는 경우가 종종 발생하였습니다.
* 이를 미연에 방지하고자 실제 Image Decoding이 제대로 되는지 ResNet Preprocessing을 거치는 작업을 해보도록 하겠습니다.


```python
def verify_image_file(meta_data):

    train_left = meta_data['xmin'].tolist()
    train_right = meta_data['xmax'].tolist()
    train_top = meta_data['ymin'].tolist()
    train_bottom = meta_data['ymax'].tolist()
    train_mask = meta_data['mask'].tolist()
    file_path_train = meta_data['file_path'].tolist()

    new_left = []
    new_right = []
    new_top = []
    new_bottom = []
    new_file_path = []
    new_mask = []

    for idx,image_path in tqdm(enumerate( file_path_train)):
        
        try:
            img = tf.io.read_file(image_path)
            img = tf.image.decode_image(img, channels=3)   
            
            img = tf.image.crop_to_bounding_box( img , train_top[idx] , train_left[idx], train_bottom[idx] - train_top[idx] , train_right[idx] - train_left[idx] )

            img = tf.image.resize(img, (224, 224))
            img = tf.keras.applications.resnet50.preprocess_input(img)

            new_left.append(train_left[idx])
            new_right.append(train_right[idx])
            new_top.append(train_top[idx])
            new_bottom.append(train_bottom[idx])
            new_file_path.append(image_path)
            new_mask.append(train_mask[idx])
        
        except Exception as e:
            print(e)
            continue
    
    print(len(new_file_path))

    result = pd.DataFrame(list(zip(new_file_path, new_mask , new_left , new_top , new_right , new_bottom)), columns=['file_path','mask','xmin','ymin','xmax','ymax'])

    return result
```


```python
meta_data = verify_image_file(meta_data)
```

    631it [00:09, 76.73it/s] 

    target_width must be > 0.
    target_width must be > 0.
    target_width must be > 0.
    target_width must be > 0.
    target_width must be > 0.
    target_width must be > 0.
    
    width must be >= target + offset.
    

    3842it [00:56, 30.06it/s]

    width must be >= target + offset.
    

    3997it [01:01, 18.78it/s]

    width must be >= target + offset.
    

    5100it [01:38, 28.36it/s]

    width must be >= target + offset.
    

    5282it [01:43, 39.03it/s]

    width must be >= target + offset.
    width must be >= target + offset.
    

    5971it [02:05, 35.54it/s]

    width must be >= target + offset.
    width must be >= target + offset.
    

    6043it [02:08, 31.23it/s]

    width must be >= target + offset.
    

    6509it [02:22, 66.83it/s]

    width must be >= target + offset.
    

    6904it [02:36, 39.18it/s]

    width must be >= target + offset.
    

    7260it [02:47, 61.09it/s]

    width must be >= target + offset.
    

    7269it [02:47, 43.38it/s]

    6812

<br>
<br>

* 2번째 Dataset에는 Mask 착요여부를 나타내는 값 중에, '제대로 마스크를 쓰지 않음(mask_weared_incorrect)' 값이 있습니다.
* 우선은 이 값을 Mask를 착용했음으로 변경하도록 하겠습니다.
* 최종적으로 얻은 값들을 저장하고, 이 값을 Train시에 사용하도록 하겠습니다.

<br>

```python
meta_data = meta_data.replace({'mask':'mask_weared_incorrect'},'with_mask')
meta_data.to_csv("meta_data.csv",index=False)
```
<br>
<br>
<br>

## 2. Train

* Preprocessing 작업에서 만들어진 Image File List를 가지고 Train을 하도록 하겠습니다.   

<br>
<br>

```python
import pandas as pd
import os
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GlobalAveragePooling2D , BatchNormalization , Dropout , Dense
from tensorflow.keras.callbacks import TensorBoard , ModelCheckpoint , LearningRateScheduler
```

<br>

* Batch Size와 Dropout Rate 설정합니다.   

<br>

```python
BATCH_SIZE = 32
DROP_OUT_RATE = 0.2
```
<br>

* Preprocessing에서 얻은 Meta Data File을 Open합니다.   

<br>

```python
dataset_info = pd.read_csv("meta_data.csv")
```

<br>

* Image에 대한 정보와 Label에 대한 정보를 분리합니다.

* Mask 착용여부가 Label이 되고, 이를 Label Encoder로 One-Hot으로 변환합니다.

<br>

```python
data_file_path = dataset_info[['file_path' , 'xmin' , 'ymin' , 'xmax' , 'ymax']]
mask = dataset_info['mask'].tolist()

le = LabelEncoder()
le.fit(mask)
print(le.classes_)

le_mask = le.transform(mask)
mask = tf.keras.utils.to_categorical(le_mask , num_classes=2)
```

    ['with_mask' 'without_mask']

<br>
<br>

* Train & Val. Dataset을 3:1로 나눕니다.

<br>

```python
file_path_train, file_path_val, y_train, y_val = train_test_split(data_file_path, mask, 
                                                                  test_size=0.25, 
                                                                  random_state=777, 
                                                                  stratify = mask)

print( len(file_path_train) , len(y_train) , len(file_path_val) , len(y_val) )
```

    5109 5109 1703 1703

<br>
<br>

* Tensorflow Dataset의 Map Function에서 사용하기 위해서 얼굴 좌표 값들과 File Path를 List로 바꾸어 놓습니다.

<br>

```python
train_left = file_path_train['xmin'].tolist()
train_right = file_path_train['xmax'].tolist()
train_top = file_path_train['ymin'].tolist()
train_bottom = file_path_train['ymax'].tolist()
file_path_train = file_path_train['file_path'].tolist()

val_left = file_path_val['xmin'].tolist()
val_right = file_path_val['xmax'].tolist()
val_top = file_path_val['ymin'].tolist()
val_bottom = file_path_val['ymax'].tolist()
file_path_val = file_path_val['file_path'].tolist()
```

<br>

* Dataset Map Function입니다.
* Image File Path를 받아서, 얼굴부분만을 잘라낸 후 Label값과 함께 돌려줍니다.

<br>

```python
def load_image( image_path , left , right , top , bottom , label ):
    img = tf.io.read_file(image_path)
    
    img = tf.image.decode_png(img, channels=3)   
    img = tf.image.crop_to_bounding_box( img , top , left, bottom - top , right - left )

    img = tf.image.resize(img, (224, 224))
    img = tf.keras.applications.resnet50.preprocess_input(img)    
    
    return img , label
```

<br>
<br>

* Dataset을 준비합니다.   
* from_tensor_slices의 Parameter에 얼굴 좌표 값과 File Path를 넣어줍니다.

<br>

```python
train_dataset = tf.data.Dataset.from_tensor_slices( (file_path_train , 
                                                     train_left , 
                                                     train_right , 
                                                     train_top , 
                                                     train_bottom , 
                                                     y_train) )

val_dataset = tf.data.Dataset.from_tensor_slices( (file_path_val , 
                                                   val_left , 
                                                   val_right , 
                                                   val_top , 
                                                   val_bottom ,
                                                   y_val) )
```


```python
train_dataset = train_dataset.shuffle(buffer_size=len(file_path_train))\
                                .map( load_image, num_parallel_calls=tf.data.experimental.AUTOTUNE)\
                                .repeat()\
                                .batch(BATCH_SIZE)\
                                .prefetch(tf.data.experimental.AUTOTUNE)


val_dataset = val_dataset.shuffle(buffer_size=len(file_path_val))\
                            .map( load_image, num_parallel_calls=tf.data.experimental.AUTOTUNE)\
                            .repeat()\
                            .batch(BATCH_SIZE)\
                            .prefetch(tf.data.experimental.AUTOTUNE)    #
```

<br>
<br>

* ResNet50으로 Feature Extraction해서 Dense로 분류하도록 하겠습니다.   

<br>

```python
ResNet50 = tf.keras.applications.resnet.ResNet50(
    weights=None,
    input_shape=(224, 224, 3),
    include_top=False)
```


```python
model= Sequential()

model.add( ResNet50 )

model.add( GlobalAveragePooling2D() ) 
model.add( Dropout( DROP_OUT_RATE ) ) 
model.add( BatchNormalization() ) 
model.add( Dense(128, activation='relu') )
model.add( Dropout( DROP_OUT_RATE ) ) 
model.add( BatchNormalization() ) 

model.add( Dense(2, activation='softmax') )
```

<br>

* Learning Rate Scheduler 정의합니다.   


```python
initial_learning_rate = 0.01

def lr_exp_decay(epoch, lr):
    k = 0.1
    return initial_learning_rate * np.math.exp(-k*epoch)

lr_scheduler = LearningRateScheduler(lr_exp_decay, verbose=1)
```

<br>
<br>

* Tensorboard와 Checkpoint 관련 값들을 정의합니다.   


```python
log_dir = os.path.join('Logs')
CHECKPOINT_PATH = os.path.join('CheckPoints_Mask_Detection')
tb_callback = TensorBoard(log_dir=log_dir)

cp = ModelCheckpoint(filepath=CHECKPOINT_PATH, 
                     monitor='val_accuracy',                     
                     save_best_only = True,
                     verbose = 1)
```

<br>  
<br>

* Model Compile   


```python
model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-3),
    #loss='binary_crossentropy',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)
```

<br>
<br>

* Train 시작   


```python
hist = model.fit(train_dataset,
                 validation_data=val_dataset,
                 callbacks=[lr_scheduler , cp , tb_callback],
                 steps_per_epoch = 200,
                 validation_steps = 50,
                 epochs = 20,
                 verbose = 1 
)
```

    C:\Users\Moon\anaconda3\envs\TF.2.5.0-GPU\lib\site-packages\tensorflow\python\keras\utils\generic_utils.py:494: CustomMaskWarning: Custom mask layers require a config and must override get_config. When loading, the custom mask layer must be passed to the custom_objects argument.
      warnings.warn('Custom mask layers require a config and must override '
    

    Epoch 1/20
    
    Epoch 00001: LearningRateScheduler reducing learning rate to 0.01.
    200/200 [==============================] - 116s 504ms/step - loss: 0.3580 - accuracy: 0.8692 - val_loss: 0.2935 - val_accuracy: 0.8988
    
    Epoch 00001: val_accuracy improved from -inf to 0.89875, saving model to CheckPoints_Mask_Detection
    

    C:\Users\Moon\anaconda3\envs\TF.2.5.0-GPU\lib\site-packages\tensorflow\python\keras\utils\generic_utils.py:494: CustomMaskWarning: Custom mask layers require a config and must override get_config. When loading, the custom mask layer must be passed to the custom_objects argument.
      warnings.warn('Custom mask layers require a config and must override '
    

    INFO:tensorflow:Assets written to: CheckPoints_Mask_Detection\assets
    Epoch 2/20
    
    Epoch 00002: LearningRateScheduler reducing learning rate to 0.009048374180359595.
    200/200 [==============================] - 95s 476ms/step - loss: 0.2450 - accuracy: 0.9077 - val_loss: 0.3663 - val_accuracy: 0.8525
    
    Epoch 00002: val_accuracy did not improve from 0.89875
    Epoch 3/20
    
    Epoch 00003: LearningRateScheduler reducing learning rate to 0.008187307530779819.
    200/200 [==============================] - 97s 488ms/step - loss: 0.2112 - accuracy: 0.9247 - val_loss: 0.4753 - val_accuracy: 0.8194
    
    Epoch 00003: val_accuracy did not improve from 0.89875
    Epoch 4/20
    
    Epoch 00004: LearningRateScheduler reducing learning rate to 0.007408182206817179.
    200/200 [==============================] - 97s 487ms/step - loss: 0.1983 - accuracy: 0.9278 - val_loss: 0.2442 - val_accuracy: 0.9119
    
    Epoch 00004: val_accuracy improved from 0.89875 to 0.91188, saving model to CheckPoints_Mask_Detection
    

    C:\Users\Moon\anaconda3\envs\TF.2.5.0-GPU\lib\site-packages\tensorflow\python\keras\utils\generic_utils.py:494: CustomMaskWarning: Custom mask layers require a config and must override get_config. When loading, the custom mask layer must be passed to the custom_objects argument.
      warnings.warn('Custom mask layers require a config and must override '
    

    INFO:tensorflow:Assets written to: CheckPoints_Mask_Detection\assets
    Epoch 5/20
    
    Epoch 00005: LearningRateScheduler reducing learning rate to 0.006703200460356393.
    200/200 [==============================] - 95s 474ms/step - loss: 0.1901 - accuracy: 0.9322 - val_loss: 0.1864 - val_accuracy: 0.9344
    
    Epoch 00005: val_accuracy improved from 0.91188 to 0.93437, saving model to CheckPoints_Mask_Detection
    

    C:\Users\Moon\anaconda3\envs\TF.2.5.0-GPU\lib\site-packages\tensorflow\python\keras\utils\generic_utils.py:494: CustomMaskWarning: Custom mask layers require a config and must override get_config. When loading, the custom mask layer must be passed to the custom_objects argument.
      warnings.warn('Custom mask layers require a config and must override '
    

    INFO:tensorflow:Assets written to: CheckPoints_Mask_Detection\assets
    Epoch 6/20
    
    Epoch 00006: LearningRateScheduler reducing learning rate to 0.006065306597126334.
    200/200 [==============================] - 94s 474ms/step - loss: 0.1803 - accuracy: 0.9383 - val_loss: 0.2283 - val_accuracy: 0.9050
    
    Epoch 00006: val_accuracy did not improve from 0.93437
    Epoch 7/20
    
    Epoch 00007: LearningRateScheduler reducing learning rate to 0.005488116360940264.
    200/200 [==============================] - 95s 477ms/step - loss: 0.1629 - accuracy: 0.9431 - val_loss: 0.1612 - val_accuracy: 0.9400
    
    Epoch 00007: val_accuracy improved from 0.93437 to 0.94000, saving model to CheckPoints_Mask_Detection
    

    C:\Users\Moon\anaconda3\envs\TF.2.5.0-GPU\lib\site-packages\tensorflow\python\keras\utils\generic_utils.py:494: CustomMaskWarning: Custom mask layers require a config and must override get_config. When loading, the custom mask layer must be passed to the custom_objects argument.
      warnings.warn('Custom mask layers require a config and must override '
    

    INFO:tensorflow:Assets written to: CheckPoints_Mask_Detection\assets
    Epoch 8/20
    
    Epoch 00008: LearningRateScheduler reducing learning rate to 0.004965853037914095.
    200/200 [==============================] - 95s 476ms/step - loss: 0.1611 - accuracy: 0.9422 - val_loss: 0.1756 - val_accuracy: 0.9369
    
    Epoch 00008: val_accuracy did not improve from 0.94000
    Epoch 9/20
    
    Epoch 00009: LearningRateScheduler reducing learning rate to 0.004493289641172216.
    200/200 [==============================] - 95s 476ms/step - loss: 0.1422 - accuracy: 0.9498 - val_loss: 0.2156 - val_accuracy: 0.9244
    
    Epoch 00009: val_accuracy did not improve from 0.94000
    Epoch 10/20
    
    Epoch 00010: LearningRateScheduler reducing learning rate to 0.004065696597405992.
    200/200 [==============================] - 77s 378ms/step - loss: 0.1497 - accuracy: 0.9503 - val_loss: 0.1187 - val_accuracy: 0.9600
    
    Epoch 00010: val_accuracy improved from 0.94000 to 0.96000, saving model to CheckPoints_Mask_Detection
    

    C:\Users\Moon\anaconda3\envs\TF.2.5.0-GPU\lib\site-packages\tensorflow\python\keras\utils\generic_utils.py:494: CustomMaskWarning: Custom mask layers require a config and must override get_config. When loading, the custom mask layer must be passed to the custom_objects argument.
      warnings.warn('Custom mask layers require a config and must override '
    

    INFO:tensorflow:Assets written to: CheckPoints_Mask_Detection\assets
    Epoch 11/20
    
    Epoch 00011: LearningRateScheduler reducing learning rate to 0.0036787944117144234.
    200/200 [==============================] - 95s 475ms/step - loss: 0.1374 - accuracy: 0.9513 - val_loss: 0.1832 - val_accuracy: 0.9337
    
    Epoch 00011: val_accuracy did not improve from 0.96000
    Epoch 12/20
    
    Epoch 00012: LearningRateScheduler reducing learning rate to 0.003328710836980796.
    200/200 [==============================] - 94s 474ms/step - loss: 0.1229 - accuracy: 0.9564 - val_loss: 0.1432 - val_accuracy: 0.9525
    
    Epoch 00012: val_accuracy did not improve from 0.96000
    Epoch 13/20
    
    Epoch 00013: LearningRateScheduler reducing learning rate to 0.0030119421191220205.
    200/200 [==============================] - 94s 474ms/step - loss: 0.1266 - accuracy: 0.9583 - val_loss: 0.1450 - val_accuracy: 0.9625
    
    Epoch 00013: val_accuracy improved from 0.96000 to 0.96250, saving model to CheckPoints_Mask_Detection
    

    C:\Users\Moon\anaconda3\envs\TF.2.5.0-GPU\lib\site-packages\tensorflow\python\keras\utils\generic_utils.py:494: CustomMaskWarning: Custom mask layers require a config and must override get_config. When loading, the custom mask layer must be passed to the custom_objects argument.
      warnings.warn('Custom mask layers require a config and must override '
    

    INFO:tensorflow:Assets written to: CheckPoints_Mask_Detection\assets
    Epoch 14/20
    
    Epoch 00014: LearningRateScheduler reducing learning rate to 0.002725317930340126.
    200/200 [==============================] - 95s 475ms/step - loss: 0.1183 - accuracy: 0.9606 - val_loss: 0.1348 - val_accuracy: 0.9606
    
    Epoch 00014: val_accuracy did not improve from 0.96250
    Epoch 15/20
    
    Epoch 00015: LearningRateScheduler reducing learning rate to 0.0024659696394160645.
    200/200 [==============================] - 95s 474ms/step - loss: 0.1106 - accuracy: 0.9658 - val_loss: 0.1364 - val_accuracy: 0.9619
    
    Epoch 00015: val_accuracy did not improve from 0.96250
    Epoch 16/20
    
    Epoch 00016: LearningRateScheduler reducing learning rate to 0.0022313016014842983.
    200/200 [==============================] - 95s 474ms/step - loss: 0.1120 - accuracy: 0.9619 - val_loss: 0.1379 - val_accuracy: 0.9550
    
    Epoch 00016: val_accuracy did not improve from 0.96250
    Epoch 17/20
    
    Epoch 00017: LearningRateScheduler reducing learning rate to 0.002018965179946554.
    200/200 [==============================] - 95s 476ms/step - loss: 0.0988 - accuracy: 0.9658 - val_loss: 0.1437 - val_accuracy: 0.9569
    
    Epoch 00017: val_accuracy did not improve from 0.96250
    Epoch 18/20
    
    Epoch 00018: LearningRateScheduler reducing learning rate to 0.001826835240527346.
    200/200 [==============================] - 97s 475ms/step - loss: 0.0921 - accuracy: 0.9725 - val_loss: 0.1049 - val_accuracy: 0.9688
    
    Epoch 00018: val_accuracy improved from 0.96250 to 0.96875, saving model to CheckPoints_Mask_Detection
    

    C:\Users\Moon\anaconda3\envs\TF.2.5.0-GPU\lib\site-packages\tensorflow\python\keras\utils\generic_utils.py:494: CustomMaskWarning: Custom mask layers require a config and must override get_config. When loading, the custom mask layer must be passed to the custom_objects argument.
      warnings.warn('Custom mask layers require a config and must override '
    

    INFO:tensorflow:Assets written to: CheckPoints_Mask_Detection\assets
    Epoch 19/20
    
    Epoch 00019: LearningRateScheduler reducing learning rate to 0.0016529888822158654.
    200/200 [==============================] - 95s 475ms/step - loss: 0.0883 - accuracy: 0.9722 - val_loss: 0.1417 - val_accuracy: 0.9613
    
    Epoch 00019: val_accuracy did not improve from 0.96875
    Epoch 20/20
    
    Epoch 00020: LearningRateScheduler reducing learning rate to 0.0014956861922263505.
    200/200 [==============================] - 94s 474ms/step - loss: 0.0928 - accuracy: 0.9686 - val_loss: 0.1166 - val_accuracy: 0.9613
    
    Epoch 00020: val_accuracy did not improve from 0.96875
    
<br>
<br>

* Train / Val. 모두 좋은 정확도를 보여줍니다.   


```python
plt.plot(hist.history['accuracy'])
plt.plot(hist.history['val_accuracy'])
plt.title('Model accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(['Train', 'Val'], loc='upper left')
plt.show()


plt.plot(hist.history['loss'])
plt.plot(hist.history['val_loss'])
plt.title('Model loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(['Train', 'Val'], loc='upper left')
plt.show()
```

<p align="center">
  <img src="/assets/Mask_Detection/output_57_0.png">
</p>

<p align="center">
  <img src="/assets/Mask_Detection/output_57_1.png">
</p>

<br>
<br>
<br>

## 3. Inference

<br>
  
* 이제 실제로 Cam을 통해서 Train한 Model이 잘 동작하는지 확인해 보도록 하겠습니다.
* 전체적인 방법은 Cam을 통해 들어온 영상을 Preprocess와 동일한 방법으로 전처리를 한 후 Model에 넣은 후에 결과를 출력하도록 하겠습니다.

<br>
<br>

```python
import cv2
import numpy as np
import tensorflow as tf
from keras.models import load_model, save_model
import tensorflow_addons as tfa
```
<br>

* Preprocess때와 동일하게 Image를 처리하기 위해서 Face Detector 및 상수 값들도 동일하게 사용하도록 하겠습니다.   

<br>

```python
MODEL_FILE = "opencv_face_detector_uint8.pb"
CONFIG_FILE = "opencv_face_detector.pbtxt"
SIZE = 300
CONFIDENCE_FACE = 0.9
RESULT = ['with_mask' , 'without_mask']
MARGIN_RATIO = 0.2
```
<br>

* Face Detector를 Load합니다.


```python
net = cv2.dnn.readNetFromTensorflow( MODEL_FILE , CONFIG_FILE )
```
<br>
<br>

* Train시킨 Model도 같이 Load합니다.   


```python
print("Loading Saved Model...")

model = load_model("CheckPoints_Mask_Detection")
```

    Loading Saved Model...
    
<br>
  
* Cam을 연결합니다.   

<br>

```python
cap = cv2.VideoCapture(0)
```
<br>
<br>
  
* 아래 부분은 Cam에서 영상을 받아서 Preprocess를 거치고 Model에 입력시키는 과정입니다.

    - **ret, frame = cap.read()**
        * Cam에서 Image를 한 장 받아옵니다.      
    
    <br>  
    
    - **i = np.argmax(detection[:,2])**
        * Face Detector가 Detect한 얼굴부분에 대한 정보 Index를 얻습니다.        
    
    <br>    
    
    - **if detection[i,2] < CONFIDENCE_FACE:**
        * Face Detector가 Detect한 얼굴부분의 신뢰도가 특정 값(0.9)이상인 경우에만 얼굴로 판단합니다.
    
    <br>
    
    - **left = left - int((right - left) * MARGIN_RATIO)
        top = top - int((bottom - top) * MARGIN_RATIO)
        right = right + int((right - left) * MARGIN_RATIO)
        bottom = bottom + int((bottom - top) * MARGIN_RATIO)**
        * Preprocess때와 마찬가지로 Margin을 두고 얼굴 부분을 Crop합니다.
    
    <br>
    
    - **cropped = np.array(cropped).reshape(-1,224,224,3)**
        * 얼굴부분을 Model에 넣기 위해 Numpy Array로 변환하고 Reshape합니다.
    
    <br>
    
    - **pred = model.predict( cropped )**
        * Model에 Input시켜서 결과를 받습니다.
    
    <br>
    
    - **Result = "Result : {0}".format(RESULT[int(np.argmax(np.reshape( pred , (1,-1) )))])
        cv2.putText(frame, Result, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX,0.5, (0, 255, 0), 2)**
        * Model은 Mask 착용 여부를 확률로 보여주고, 이를 출력합니다.

<br>
<br>
  
```python
while cv2.waitKey(1) < 0:
    ret, frame = cap.read()
    rows, cols, channels = frame.shape

    blob = cv2.dnn.blobFromImage(frame, 1.0)

    net.setInput(blob)
    detections = net.forward()

    detection = detections[0, 0]    
    i = np.argmax(detection[:,2])

    if i != 0:
        print("Max index is not 0")
        continue

    if detection[i,2] < CONFIDENCE_FACE:
        print("Low CONFIDENCE_FACE" , detection[i,2])
        continue

    if detection[i,3] >= 1.00 or detection[i,4] >= 1.00 or detection[i,5] >= 1.00 or detection[i,6] >= 1.00 or detection[i,3] <= 0 or detection[i,4] < 0 or detection[i,5] <= 0 or detection[i,6] <= 0:
        pass
    else:
        left = int(detection[i,3] * cols)
        top = int(detection[i,4] * rows)
        right = int(detection[i,5] * cols)
        bottom = int(detection[i,6] * rows)

        left = left - int((right - left) * MARGIN_RATIO)
        top = top - int((bottom - top) * MARGIN_RATIO)
        right = right + int((right - left) * MARGIN_RATIO)
        bottom = bottom + int((bottom - top) * MARGIN_RATIO)

        if left < 0:
            left = 0

        if right > cols:
            right = cols

        if top < 0:
            top = 0

        if bottom > rows:
            bottom = rows

        cropped = frame[top:bottom, left:right]
        cropped = cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB)
        cropped = cv2.resize( cropped , dsize=(224,224) )
        cropped = np.array(cropped).reshape(-1,224,224,3)

        cropped = tf.keras.applications.resnet50.preprocess_input(cropped)

        pred = model.predict( cropped )
        print(pred)

        Result = "Result : {0}".format(RESULT[int(np.argmax(np.reshape( pred , (1,-1) )))])

        cv2.putText(frame, Result, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX,0.5, (0, 255, 0), 2)
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)

    cv2.imshow("VideoFrame", frame)


cap.release()
cv2.destroyAllWindows()
```

<br>
  
    [[9.9958712e-01 4.1284965e-04]]
    [[9.9957865e-01 4.2130650e-04]]
    [[9.997712e-01 2.287866e-04]]
    Low CONFIDENCE_FACE 0.85542315
    [[9.9969637e-01 3.0366165e-04]]
    Low CONFIDENCE_FACE 0.47966802
    Low CONFIDENCE_FACE 0.58707684
    Low CONFIDENCE_FACE 0.42296803

<br>
<br>
  
<p align="center">
  <img src="/assets/Mask_Detection/Result.png">
</p>

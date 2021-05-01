---
title: "Tensorflow Input Pipeline"
date: 2021-04-30 08:26:28 -0400
categories: Deep Learning
---

### Tensorflow Input Pipeline

<br>
<br>
<br>
<br>

* 주어진 Data로 부터 Train에 필요한 Data형태로 변환하기까지는 매우 지루하고 험난한 과정입니다.


* Model에 입력 Foramt에 맞게 Shape을 변경하고, Data Augmentation도 고려해야 합니다.


* 가장 중요한 것은 주어진 Data가 수십, 수백만개가 있다면 Performance 또한 중요한 고려 요소가 됩니다.


* 이런 모든 고민을 해결해 주기 위해서 Tensorflow에서는 tf.data Module과 tf.data.Dataset Module을 준비놓았습니다.


* 이번 Post에서는 Tensorflow를 이용하여 효율적인 Data Input Pipeline을 만드는 방법을 알아보고자 합니다.


* tf.data.Dataset에서는 map / prefetch / cache / batch 이렇게 4가지 Fuction이 가장 중요하며, 이를 어떻게 사용하는지도 예제를 통해서 확인해 보도록 하겠습니다.
<br>
<br>
<br>
<br>
<br>
<br>
<br>
<br>

## 0. Example Dataset Download   

* Tensorflow Input Pipeline을 설명할 때 자주 사용되는 Dataset이 있더군요.

  [Image Data for Multi-Instance Multi-Label Learning](https://www.lamda.nju.edu.cn/data_MIMLimage.ashx?AspxAutoDetectCookieSupport=1)

<br>

* 저도 이 Dataset을 이용해 보도록 하겠습니다.

<br>

* Download는 여기에서 하시면 됩니다.
[Dataset Download](https://www.dropbox.com/s/0htmeoie69q650p/miml_dataset.zip)

<br>

* 압축을 풀면 Dataset 준비는 끝입니다.

<br>
<br>
<br>
<br>
<br>
<br>

## 1. Dataset 살펴보기


```python
import tensorflow as tf
import pandas as pd
import PIL
import PIL.Image
import pathlib
import matplotlib.pyplot as plt
import numpy as np
import os
```

   

   

   

* Image File은 'image'라는 Folder에 저장되어 있으며, 각 Image File에 대한 Label은 miml_labels_1.csv File에 저장되어 있습니다.


* 먼저 Label이 저장되어 있는 File부터 열어보겠습니다.


```python
df = pd.read_csv("./miml_dataset/miml_labels_1.csv")
df.head()
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Filenames</th>
      <th>desert</th>
      <th>mountains</th>
      <th>sea</th>
      <th>sunset</th>
      <th>trees</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1.jpg</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2.jpg</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3.jpg</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4.jpg</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5.jpg</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>

<br>

* Filename과 해당 Image File에 묘사되어 있는 객체들에 해당되는 항목에 1이 표시되어 있는 형태로 되어 있네요.

<br>

* 한 가지 더 눈여겨 봐야 할 것은, Image File 하나에 반드시 하나만 1이 표시되어 있지 않은 경우도 있다는 것입니다.

```python
LABELS=["desert", "mountains", "sea", "sunset", "trees"]
```

* 한 장을 살펴보면 다음과 같습니다.


```python
data_dir = pathlib.Path("miml_dataset")
filenames = list(data_dir.glob('images/*.jpg'))
print("image count: ", len(filenames))
print("first image: ", str(filenames[0]) )
```

    image count:  2000
    first image:  miml_dataset\images\1.jpg
    


```python
PIL.Image.open(str(filenames[0]))
```

<br>
<p align="center">
  <img src="/assets/Tensorflow_Pipeline/output_36_0.png">
</p>
<br>
<br>
<br>

* Dataset만들 때 사용하기 위해서 Image File Path 전체를 담고 있는 List를 만들겠습니다.   


```python
fnames=[]
for fname in filenames:
    fnames.append(str(fname))

fnames[:5]
```




    ['miml_dataset\\images\\1.jpg',
     'miml_dataset\\images\\10.jpg',
     'miml_dataset\\images\\100.jpg',
     'miml_dataset\\images\\1000.jpg',
     'miml_dataset\\images\\1001.jpg']
     
<br>
<br>
<br>
<br>
<br>
<br>

# 2. Dataset 만들기   

* Dataset을 만들기 위해서 다양한 방법이 있지만, 가장 흔하고 쉽게 사용할 수 있는 tf.data.Dataset.from_tensor_slices()을 사용하도록 하겠습니다.

    [tf.data.Dataset.from_tensor_slices()](https://www.tensorflow.org/api_docs/python/tf/data/Dataset#from_tensor_slices)


* 일반적으로 from_tensor_slices()의 인자로 Train에 사용할 File List를 인자로 넣고, map()으로 전처리를 하는 방법을 많이 쓰고, 이 방법이 많은 유연성을 제공합니다.

   
<br>
<br>
<br>

* 우리가 가진 Imgae File의 전체 갯수는 2000개 이며, Path는 모두 fnames에 담겨져 있습니다.   


```python
ds_size= len(fnames)
print("Number of images in folders: ", ds_size)
```

    Number of images in folders:  2000
    
    
<br>
<br>

* fnames를 from_tensor_slices()의 Param으로 넘기겠습니다.


* 이제 Dataset이 하나 만들어졌습니다.

#### **여기서 from_tensor_slices()의 Parameter로 들어가는 fnames를 주목해 주시기 바랍니다.**

#### **Image File Name을 저장하고 있는 List입니다.**

#### **이후에 나올 .map() Function과 .map()에 적용할 Function들과도 밀접하게 연관되어 있기 때문입니다.**


```python
filelist_ds = tf.data.Dataset.from_tensor_slices( fnames )
```

<br>
<br>
<br>

```python
ds_size= filelist_ds.cardinality().numpy()
print("Number of selected samples for dataset: ", ds_size)
```

    Number of selected samples for dataset:  2000
    

* filelist_ds의 전체 수는 2000개입니다. 
  
  ( cardinality()는 Dataset의 전체 갯수를 Return합니다. )
  
  
<br>
<br>
<br>

* 우리가 만든 Dataset이 제대로된 Data를 가지고 있는지 확인해 보겠습니다.

   

   

* Dataset에서 3개만 뽑아서 출력해 보겠습니다.   


```python
for a in filelist_ds.take(3):
    fname= a.numpy().decode("utf-8")
    print(fname)
    display(PIL.Image.open(fname)) 
```

    miml_dataset\images\1.jpg
    


<p align="left">
  <img src="/assets/Tensorflow_Pipeline/output_81_1.png">


    miml_dataset\images\10.jpg
    


<p align="left">
  <img src="/assets/Tensorflow_Pipeline/output_81_3.png">

    miml_dataset\images\100.jpg
    


<p align="left">
  <img src="/assets/Tensorflow_Pipeline/output_81_5.png">
  
<br>
<br>
<br>  
<br>
<br>
<br>

# 3. Label 만들기   

* Train Data로 이용할 Dataset을 만들었으니, 이제 그에 맞는 Label도 만들어 보겠습니다.   

   

* Label정보는 CSV에 있었죠?   


```python
df.head()
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Filenames</th>
      <th>desert</th>
      <th>mountains</th>
      <th>sea</th>
      <th>sunset</th>
      <th>trees</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1.jpg</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2.jpg</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3.jpg</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4.jpg</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5.jpg</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>




```python
LABELS
```




    ['desert', 'mountains', 'sea', 'sunset', 'trees']
    
<br>
<br>
<br>

* 아래 Function은 Image File Full Path를 받아서 해당하는 Label 정보만을 Tensor형태로 return해 줍니다.


```python
def get_label(file_path):

    parts = tf.strings.split(file_path, '\\')
    
    file_name= parts[-1]      

    # Dataframe에서 LABELS에 정의된 Column만 뽑아낸다.
    # .to_numpy()는 DataFrame인 Series를 ndarray 형태로 변환한다.
    #.squeeze()는 1차원인 축을 제거한다.
    labels= df[df["Filenames"]==file_name][LABELS].to_numpy().squeeze()
    
    return tf.convert_to_tensor(labels)
```


<br>
<br>
<br>

* 만든 Function이 작동을 제대로 하는지 확인해 보겠습니다.   


```python
for a in filelist_ds.take(5):
    print("file_name: ", a.numpy().decode("utf-8"))
    print(get_label(a).numpy())
```

    file_name:  miml_dataset\images\1.jpg
    [1 0 0 0 0]
    file_name:  miml_dataset\images\10.jpg
    [1 1 0 0 0]
    file_name:  miml_dataset\images\100.jpg
    [1 0 0 1 0]
    file_name:  miml_dataset\images\1000.jpg
    [0 0 1 0 0]
    file_name:  miml_dataset\images\1001.jpg
    [0 0 1 0 0]
    

* 네, 제대로 하네요. 저 값들을 Label로 사용하면 될 것 같습니다.


<br>
<br>
<br>
<br>
<br>
<br>

# 4. Dataset Preprocessing



* 궁극적으로 Train 시키기 위해서 Model에 넣을 때는 Tensor 형태로 변환이 되어야 합니다.


* 우리가 할 작업은 Image Classification이고, 일련의 작업을 거쳐 우리가 정의한 Model이 필요로 하는 Shape에 맞게 변형이 되어야 합니다.


* 이를 위해서 Image File을 읽어서 Model에 필요한 Shape을 만들고, 또한 그에 맞는 Label도 만들어 주는 함수를 정의하도록 하겠습니다.


* **이 작업은 모든 Dataset에 대해서 적용할 것이고, Dataset에 map()의 Parameter로 사용될 것입니다.**


* 여기서 사용할 Pre-Train Model은 VGG16이고, 이 Model은 입력을 32x32를 필요로 합니다.


```python
IMG_WIDTH, IMG_HEIGHT = 32 , 32
```

* 아래 함수는 Image File을 읽어서 Float Type Tensor형태로 바꾸어 줍니다.

  ( 참 편리하네요. )


```python
def process_img(img):
    
    # Image File을 읽습니다.
    img = tf.image.decode_jpeg(img, channels=3) 

    # Image Data를 0~1 사이의 값으로 변환해줍니다.
    img = tf.image.convert_image_dtype(img, tf.float32) 
    
    # 입력 Size에 맞게 Resize
    return tf.image.resize(img, [IMG_WIDTH, IMG_HEIGHT]) 
```

<br>
<br>
<br>

* 아래의 함수는 Image와 함께 Label 정보도 같이 생성하여 Return해 줍니다.   


* 최종적으로 아래의 combine_images_labels() Function을 Dataset의 .map()에 적용할 것입니다.


* 이 함수가 받는 Parameter와 Return Values를 잘 보시기 바랍니다.
  - Tensorflow에서 왜 이렇게 복잡하게 만들어 놨는지는 모르겠으나, **.map()에 적용될 Function, 즉, combine_images_labels()의 Parameter로 넘어오는 file_path는 이전에 from_tensor_slices()의 Parameter로 넣은 값이 여기로 넘어옵니다.**
    
  - filelist_ds = tf.data.Dataset.from_tensor_slices( fnames ) 로 Dataset이 만들어졌고, fnames가 file_path로 넘어온다는 뜻입니다.
    
  - 그리고, Return Value, 여기서는 **img와 label인데, 결과적으로 이 두 값이 .fit()에서 Train & Target Value가 됩니다.**
    
  - 이런 복잡한 관계를 잘 설명해 줬으면 좋겠는데, 아쉽네요.
  
  
* 결론적으로, Tensorflow Pipeline 작성에 가장 중요한 부분이 저는 이 부분이라고 생각합니다.


* 주어진 Raw Data File(Image , Sound 등등)을 Train을 할 수 있는 Tensor형태로 Conversion하는 중요한 부분이며, 역량과 내공이 잘 드러날 수 있는 부분이라고 생각합니다.


* 그리고, 이 Function을 작성하실 때 모든 Code를 Tensorflow Function을 이용한다면 훨씬 더 뛰어난 성능을 얻을 수 있습니다.


* 아래 Code에도 Image File을 읽는 부분을 tf.io Module을 사용하고 있는 것을 볼 수 있습니다. 또한, process_img()도 살펴보시면 모두 tf를 사용하고 있는 것을 보실 수 있습니다. 다 이유가 있는 것입니다. 다만 get_label()에는 단 한 줄이 tf Module을 사용하고 있지 않습니다. 이런 경우에는 .map() Function 사용법이 약간 달라지는데 아래에서 확인해 보도록 하겠습니다.


```python
def combine_images_labels(file_path: tf.Tensor):

    img = tf.io.read_file(file_path)

    # 위에서 정의한 함수
    img = process_img(img)

    # Label을 만들어 주는 함수
    label = get_label(file_path)
    
    return img, label
```

* Train & Val Set을 80:20으로 나눕니다.   


```python
train_ratio = 0.80
ds_train=filelist_ds.take(ds_size*train_ratio)
ds_test=filelist_ds.skip(ds_size*train_ratio)

print("train size: ", ds_train.cardinality().numpy())
print("test size: ", ds_test.cardinality().numpy())
```

    train size:  1600
    test size:  400
    
<br>
<br>

* 아래 Code는 Dataset에 .map()을 적용하는 예제입니다.


* tf.py_function은 Dataset에 적용할 Function을 정의하는 부분입니다. 앞에서 정의한 combine_images_labels을 적용하도록 하겠습니다.


* 만약 combine_images_labels이 순수 Tensorflow Module의 Function만으로 구성되어 있다면 단순히 그 Function Name만 적어주면 됩니다.


* 이 부분은 다음에 기회가 되면 다루도록 하겠습니다.


* num_parallel_calls를 tf.data.experimental.AUTOTUNE로 설정합니다.  이 부분은 Tensorflow가 동적으로 Pipeline을 Background로 동적으로 할당하여 성능 향상하도록 해줍니다.


* 반드시 prefetch를 해 주시기 바랍니다. 


```python
ds_train = ds_train.map(lambda x: 
                        tf.py_function(func=combine_images_labels,
                                       inp=[x], 
                                       Tout=(tf.float32 , tf.int64)),
                        num_parallel_calls=tf.data.experimental.AUTOTUNE,
                        deterministic=False)

ds_train.prefetch(ds_size-ds_size*train_ratio)
```




    <PrefetchDataset shapes: (<unknown>, <unknown>), types: (tf.float32, tf.int64)>




```python
ds_test = ds_test.map(lambda x: 
                      tf.py_function(func=combine_images_labels,
                                     inp=[x], Tout=(tf.float32,tf.int64)),
                      num_parallel_calls=tf.data.experimental.AUTOTUNE,
                      deterministic=False)

ds_test.prefetch(ds_size-ds_size*train_ratio)
```




    <PrefetchDataset shapes: (<unknown>, <unknown>), types: (tf.float32, tf.int64)>
    
    

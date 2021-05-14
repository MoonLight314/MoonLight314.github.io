---
title: "Making TFReocrd File(EN)"
date: 2021-05-13 08:26:28 -0400
categories: Deep Learning
---

### Making TFReocrd File

<br>
<br>
<br>
<br>

* TFRecord File Format은 Tensorflow의 자체적인 Binary File Format입니다.


* 대규모 Dataset으로 작업을 할 경우 Binary File로 작업을 한다면 Data Input Pipeline의 효율을 높일 수 있으며, 결과적으로 전체적인 Model의 Training 시간도 향상될 수 있습니다.


* Binary Data는 Storage에 공간도 덜 차지할 뿐 아니라, Read / Write시에도 더 효율적입니다. 더욱이, Storage가 Motor를 사용하는 장치라면 더욱 그렇습니다.


* 단순히 TFReocrd File Format이 Binary여서 성능 향상을 이룬다는 것이 아니라, TFRecord가 Tensorflow에 최적화 되어 있기 때문에 Tensorflow가 제공하는 다양한 Library와 같이 사용될 경우에 그 성능은 최고가 됩니다.


* TFRecord File Format에 대한 공식 문서는 아래 Link를 참고해 주시기 바랍니다.

  [TFRecord and tf.train.Example](https://www.tensorflow.org/tutorials/load_data/tfrecord?hl=en)
  
  

* 제가 TFReocrd를 사용하기 위해서 여러 자료를 확인해 본 결과, 이 File Format을 생성하거나 사용하기가 쉽지 않다는 것입니다.


* 그래서 이번 Post에서는 TFRecord File Format을 직접 만들어보도록 해 보겠습니다.

<br>
<br>
<br>
<br>

## 0. Dataset 선택

* 이번 Post의 목적은 기존에 존재하던 Dataset을 TFRecord Format으로 변환하는 것입니다.


* 이를 위해 다음의 'Dog & Cat' Image File Dataset을 준비했습니다. 아래 Link에서 Download할 수 있습니다.

  [Dog & Cat Dataset Download](https://www.microsoft.com/en-us/download/details.aspx?id=54765)
  
  
* 이 Dataset을 선택한 이유는 적절한 Data의 수(Dog & Cat 각각 12500장씩)가 있고, CNN을 이용하여 Classification하기에 적절하다고 생각했기 때문입니다.


* Download 받아서 압축을 풀면, 'PetImages'라는 Folder가 보이고 그 안에는 Dog / Cat이라는 개별 Folder가 있고 그 안에 각각 12500장씩 Image File이 있습니다.

<br>
<br>
<br>
<br>
<br>
<br>
<br>
<br>

## 1. Dataset 살펴보기   

* 먼저 Dataset을 살짝 살펴보기로 하죠   

<br>
<br>
<br>

* 필요한 Module을 Load합니다.   


```python
import tensorflow as tf
from tqdm import tqdm
from matplotlib.pyplot import imshow
from PIL import Image
```

* Dog와 Cat의 File List들을 Full Path로 저장해 놓습니다.


* 이후 작업에 필요한 작업이니 미리 해 두도록 합시다.


```python
CatFileList = tf.io.gfile.glob("./PetImages/Cat/*.jpg")
DogFileList = tf.io.gfile.glob("./PetImages/Dog/*.jpg")
```


```python
CatFileList = [c.replace("\\","/") for c in CatFileList]
DogFileList = [c.replace("\\","/") for c in DogFileList]
```

<br>
<br>
<br>


* 어떤 그림들이 있는지 볼까요?   


```python
pil_im = Image.open(CatFileList[0])
imshow(pil_im)
```

<br>
<br>
<br>

<p align="left">
  <img src="/assets/Making_TFRecord/output_32_1.png">
</p>

<br>
<br>
<br>

```python
pil_im = Image.open(DogFileList[0])
imshow(pil_im)
```

<br>
<br>
<br>

<p align="left">
  <img src="/assets/Making_TFRecord/output_33_1.png">
</p>

<br>
<br>
<br>

* 좋습니다. 이런 Image File들이 12500장씩 있는 것 같네요.    


<br>
<br>
<br>
<br>
<br>
<br>

## 2. Preprocess   

* 변환에 앞서 미리 해야할 일이 있습니다.


* TFRecord File Format으로 변환하기 위해서 TF module로 Image File을 Load하는 Step이 있는데, 이 과정에서 몇몇 File들을 TF Module이 읽지 못하는 문제가 발생합니다.


* 정확한 원인은 모르나, 무언가 Image File의 문제로 TF Module이 읽지 못하는 것으로 보이며, 미리 이런 File들을 걸러내도록 하겠습니다.


* 방법은 전체 Image File들을 하나씩 읽어보며 실제 Error가 발생하는지 확인하는 단순하지만 확실한 방법을 사용하도록 하겠습니다.

<br>
<br>
<br>

* 아래 Function은 Imgae File을 읽어서 Model에 입력할 수 있는 Format으로 변환하는 함수입니다.   


* 이 Function을 이용해 Image File을 Test해 보도록 하겠습니다.


```python
def load_image(image_path):
    img = tf.io.read_file(image_path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, (224, 224))
    img = tf.keras.applications.efficientnet.preprocess_input(img)
    return img , image_path
```

<br>
<br>
<br>

### 2.1 Filtering Image File

* 아래 Function으로 걸러내도록 하겠습니다.   

   


```python
def FilterOpenErrorFile():
    ErrorFileList = []

    for fname in tqdm(CatFileList):
        try:
            load_image(fname)
        except:
            ErrorFileList.append(fname)
            dst = fname.replace("Cat","Error")
            tf.io.gfile.rename(fname , dst)
            
    for e in ErrorFileList:
        CatFileList.remove(e)

    
    ErrorFileList = []

    for fname in tqdm(DogFileList):
        try:
            load_image(fname)
        except:
            ErrorFileList.append(fname)
            dst = fname.replace("Dog","Error")
            tf.io.gfile.rename(fname , dst)
            
    for e in ErrorFileList:
        DogFileList.remove(e)
```


```python
ErrorFile = tf.io.gfile.glob("./PetImages/Error/*.jpg")

if len(ErrorFile) == 0:
    FilterOpenErrorFile()
```

    100%|███████████████████████████████████████████████████████████████████████████| 12427/12427 [01:34<00:00, 131.10it/s]
    100%|███████████████████████████████████████████████████████████████████████████| 12397/12397 [01:00<00:00, 205.36it/s]
    


```python
print("Num of Cat File : ",len(CatFileList))
print("Num of Dog File : ",len(DogFileList))
```

    Num of Cat File :  12427
    Num of Dog File :  12397
    

   

* Error가 있는 Image File 들을 제거하고 나니, Image File 수가 약간 줄어들었네요.


<br>
<br>
<br>
<br>
<br>
<br>

## 3. Writing TFRecord Format File   

* 이제 본격적으로 시작해 보도록 하겠습니다.


* 우선 앞에서 만들어 놓은 각각의 Full Path List를 이용해서 Dataset을 각각 만듭니다.


```python
cat_dataset = tf.data.Dataset.from_tensor_slices( CatFileList )
dog_dataset = tf.data.Dataset.from_tensor_slices( DogFileList )
```

<br>
<br>
<br>

* 우리가 이번에 TFRecord File로 저장하려고 하는 내용은 EfficientNet으로 추출된 Image들의 Feature들입니다.


* 그래서 앞의 load_image() Function에서 efficientnet의 preprocess() Function을 사용했습니다.


* **알아두시면 좋은 것은 TFRecord나 Tensorflow Pipeline을 구성할 때 가능하면 모든 Function은 Tensorflow에서 제공하는 Function을 사용하면 성능면에서 훨씬 뛰어난 결과를 얻을 수 있습니다.**

  ( 살펴보시면 load_image() Function에서도 모든 Code를 TF Module로만 작성하였습니다. )
  
<br>
<br>
<br>

* Cat Dataset / Dog Dataset 2개를 만듭니다.

```python
cat_dataset = cat_dataset.map(
                        load_image, 
                        num_parallel_calls=tf.data.experimental.AUTOTUNE)

dog_dataset = dog_dataset.map(
                        load_image, 
                        num_parallel_calls=tf.data.experimental.AUTOTUNE)
```


```python
print(cat_dataset , dog_dataset)
```

    <ParallelMapDataset shapes: ((224, 224, 3), ()), types: (tf.float32, tf.string)> <ParallelMapDataset shapes: ((224, 224, 3), ()), types: (tf.float32, tf.string)>
    

* Dataset에 Map Function(load_image)를 적용하면, Return Data의 Shape과 Type을 주목해 주시기 바랍니다.


* (224,224,3)은 EfficientNet의 Input Shape이며, Data Type은 Float입니다.


* Label 생성에 사용할 목적으로 Full Path도 같이 Return하고 있으며, String형태입니다.


<br>
<br>
<br>

* Feature를 Extract할 CNN Model을 만듭니다.


* EfficientNetB0를 이용해서 Feature Extraction을 하도록 하겠습니다.

<br>
<br>
<br>

```python
image_model = tf.keras.applications.EfficientNetB0(include_top=False,
                                                   weights='imagenet')


new_input = image_model.input
hidden_layer = image_model.layers[-1].output

image_features_extract_model = tf.keras.Model(new_input, hidden_layer)
```


```python
print(cat_dataset.cardinality().numpy())
print(dog_dataset.cardinality().numpy())
```

    12427
    12397
    

* 각 Dataset의 전체 Image File의 개수는 위와 같습니다.   

<br>
<br>
<br>
<br>
<br>
<br>

* 아래 Function이 TFRecord File로 저장하는 부분을 구현한 내용이며 이번 Post의 핵심입니다.
<br>
<br>

* 제가 TFRecord File로 저장하려는 내용은 2가지입니다.
  - EfficientNet으로 추출한 **Image의 Feature. Shape은 (7,7,1280) 입니다.**
  - 해당 Image에 대한 **Label** 입니다. **Cat은 0, Dog는 1**로 하도록 하겠습니다.
<br>
<br> 
  
* **feature = image_features_extract_model(img)**
  - EfficientNet으로부터 Feature를 추출합니다. 
  - Shape은 (7,7,1280) 입니다.
<br>
<br>

* **feature = feature.numpy().flatten()**
  - 제가 TFRecord에 관련해서 Test해 본 결과, **TFRecord는 2차원 이상의 Array는 저장하지 못하더라구요.**
  - 위와 같이 (7,7,1280)을 Flatten할 필요가 있습니다.
<br>
<br>  
  
* **writer = tf.io.TFRecordWriter( tf_path )**
  - TFRecordWriter를 하나 정의합니다.
  - Parameter는 TFRecord File의 Full Path입니다.
<br>
<br> 

* tf.train.Example을 이용해서 TFRecord File에 저장될 내용의 Format을 정의합니다.
<br>
<br>

* tf.train.Features()로 Contents를 정의합니다.
<br>
<br>

* feature라는 Dict.으로 정의하는데, Dict.의 **Key는 Item의 Name, Value는 해당 Item의 Data Type과 값을 가지고 있는 변수를 지정합니다.**
  - 이번 예제에서는 2가지 값만 정의했지만, 필요에 따라서는 더 많이 추가하시면 됩니다.
  - 'Feature': tf.train.Feature(float_list=tf.train.FloatList(value = feature))
     * Key는 'Feature'이고, Image의 Feature를 저장할 부분입니다.
     * value는 앞에서 구한 feature를 넣습니다.
     
  - 'Label': tf.train.Feature(int64_list=tf.train.Int64List(value = label))
     * Key는 'Label'이고, Image가 Cat인지 Dog인지를 저장합니다.


* Value의 Data Type에 따라서 어떤 tf.train.XXX를 사용할지는 아래를 참고해 주시기 바랍니다.
  - **tf.train.BytesList**
    * string
    * byte

  - **tf.train.FloatList**
    * float (float32)
    * double (float64)
    
  - **tf.train.Int64List**
    * bool
    * enum
    * int32
    * uint32
    * int64
    * uint64
<br>
<br>
    
* writer.write(example.SerializeToString())    
  - 준비가 다 되었으면, 실제 TFRecord File로 기록합니다.


```python
for img, path in tqdm(cat_dataset):
    
    feature = image_features_extract_model(img)
    
    tf_path = path.numpy().decode("utf-8")
    tf_path = tf_path.replace("Cat" , "Cat/TFRecord")
    tf_path = tf_path.replace("jpg" , "tfrecord")
    
    feature = feature.numpy().flatten()    #TensorShape([1, 7, 7, 1280])
    
    writer = tf.io.TFRecordWriter( tf_path )
    label = [0] # Cat

    example = tf.train.Example(features=tf.train.Features(
                                feature={'Feature': tf.train.Feature(float_list=tf.train.FloatList(value = feature)),
                                         'Label': tf.train.Feature(int64_list=tf.train.Int64List(value = label))
                                        }
                                )
                              )

    writer.write(example.SerializeToString())

```

<br>
<br>

* Dog도 Cat과 동일하게 작성합니다.   


```python
for img, path in tqdm(dog_dataset):
    
    feature = image_features_extract_model(img)
    
    tf_path = path.numpy().decode("utf-8")
    tf_path = tf_path.replace("Dog" , "Dog/TFRecord")
    tf_path = tf_path.replace("jpg" , "tfrecord")
    
    feature = feature.numpy().flatten()    #TensorShape([1, 7, 7, 1280])
    
    writer = tf.io.TFRecordWriter( tf_path )
    label = [1] # Dog

    example = tf.train.Example(features=tf.train.Features(
                                feature={'Feature': tf.train.Feature(float_list=tf.train.FloatList(value = feature)),
                                         'Label': tf.train.Feature(int64_list=tf.train.Int64List(value = label))
                                        }
                                )
                              )

    writer.write(example.SerializeToString())

```


<br>
<br>
<br>
<br>

* 한참 시간이 지나면 Cat / Dog Folder 내에 TFReocrd라는 Folder가 생기고, 그 안에 Image File Name과 동일하지만 확장자만 .tfrecord인 TFReocrd File들이 생겼을 것입니다.


* 다음 Post에서는 이렇게 만든 TFReocrd File들을 이용하여 Image Classification을 해보도록 하겠습니다.

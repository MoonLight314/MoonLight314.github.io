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

* TFRecord file format is Tensorflow's own binary file format.


* In case of working with a large dataset, working with a binary file can increase the efficiency of the data input pipeline and as a result, the training time of the overall model can be improved.


* Binary data not only takes up less space in the storage, but is more efficient when reading / writing. Moreover, even more so if storage is a device that uses motors.


* It does not mean that performance is simply improved because TFReocrd file format is binary but because TFRecord is optimized for Tensorflow, its performance is the best when it is used with various libraries provided by Tensorflow.


* Please, refer to the below link about official TFRecord file format

  [TFRecord and tf.train.Example](https://www.tensorflow.org/tutorials/load_data/tfrecord?hl=en)
  
  

* After studying various materials to use TFReocrd, it is not easy to create or use this file format.


* So, in this post, I will try to create the TFRecord file format myself.

<br>
<br>
<br>
<br>

## 0. Dataset Selection

* The purpose of this post is to convert the existing dataset into TFRecord format.


* I've selected 'Dog & Cat' image file dataset for this post. Please, refer to the below link for download

  [Dog & Cat Dataset Download](https://www.microsoft.com/en-us/download/details.aspx?id=54765)
  
  
* The reason why I selected this dataset is that this dataset has appropriate number of data (12500 each of Dog & Cat) and I thought it was appropriate for classification using CNN.


* After downloading and unzip, you will see a folder called'PetImages' and there is a separate folder called Dog / Cat in it and there are 12500 image files each in it.

<br>
<br>
<br>
<br>
<br>
<br>
<br>
<br>

## 1. Look into Dataset 

* Let's take a look at the Dataset.

<br>
<br>
<br>

* Load necessary modules


```python
import tensorflow as tf
from tqdm import tqdm
from matplotlib.pyplot import imshow
from PIL import Image
```

* Store full path of all file list of Dog & Cat


* This is necessary for the subsequent work, so let's do it in advance.


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


* Let's take a look what kind of pictures they have.


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

* OK, good. It seems that there are 12500 images of these image files each.


<br>
<br>
<br>
<br>
<br>
<br>

## 2. Preprocess   

* Some processing is required before conversion.


* In order to convert to TFRecord file format, there is a step of loading an image file with TF module. In this process, I've found that TF module cannot read some files.


* I'm not sure what the exact cause is, some image files have problem. So, I'll filter those files in advance.


* As for the method, I will use a simple but reliable method of reading all image files one by one and checking if an actual error occurs.

<br>
<br>
<br>

* The following function is to read one image file and convert it into the format that model.


* I'll verify all image files by this function.


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

* I'll filter them by the below function.
   


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
    

   

* We can see that the total numbers of image files are a little bit decreased after filtering.


<br>
<br>
<br>
<br>
<br>
<br>

## 3. Writing TFRecord Format File   

* Now, let's start in earnest.


* First, I'd like to make datasets with full path lists made previous step.


```python
cat_dataset = tf.data.Dataset.from_tensor_slices( CatFileList )
dog_dataset = tf.data.Dataset.from_tensor_slices( DogFileList )
```

<br>
<br>
<br>

* The contents I'm trying to save as TFRecord file are the features of images extracted by EfficientNet.


* So, I've used preprocess() function of EfficientNet in load_image() Function.


* **I'd like to recommend to use modules or functions provided by Tensorflow in configuring TFReocrd or data input pipeline and this makes you get much better results in performance.**

  ( If you look at load_image() function, you can easily notice that all the codes in function were written only in TF module. )
  
<br>
<br>
<br>

* Making 2 datasets ( Cat & Dog )

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
    

* When applying map function(load_image) to dataset, please, pay attention to the shape and type of return data.


* The shape,(224,224,3), is the input shape of EfficientNet and the data type is float.


* It also returns full path information to make a label and its type is string.


<br>
<br>
<br>

* Now, let's make a CNN model to extract features from image files.


* I'll apply EfficientNetB0 model to extract features

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
    

* Please, refer to the total number of each datasets as above

<br>
<br>
<br>
<br>
<br>
<br>

* The following function is the implementation of writing TFRecord file format and it's the **essential part** of this post.
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

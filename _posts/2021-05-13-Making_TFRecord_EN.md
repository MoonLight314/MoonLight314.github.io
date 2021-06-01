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

* There are two things I want to save as a TFRecord File.
  - **Feature of Image** extracted with EfficientNet. **Shape is (7,7,1280)** .
  - **Label** for that Image. **Cat is 0, Dog is 1**.
<br>
<br> 
  
* **feature = image_features_extract_model(img)**
  - Extracting features by EfficientNet
  - Shape is (7,7,1280)
<br>
<br>

* **feature = feature.numpy().flatten()**
  - As a result of testing TFRecord, **TFRecord cannot store multi dimension array.**
  - So we need to flatten (7,7,1280) as above.
<br>
<br>  
  
* **writer = tf.io.TFRecordWriter( tf_path )**
  - Define a TFRecordWriter.
  - The parameter is full path info. of TFRecord files
<br>
<br> 

* Define the format of the content to be saved in the TFRecord file by using tf.train.Example.
<br>
<br>

* Define the contents by tf.train.Features()
<br>
<br>

* It is defined as a Dict. called feature, **Key of Dict. is the name of the item, and value is the data type of the item and the variable that has the value.**
  - In this example, we have only defined two values, but you can add more as needed.
  - 'Feature': tf.train.Feature(float_list=tf.train.FloatList(value = feature))
     * Key is 'Feature', and it is the part to save the image feature.
     * Value stroes the previously obtained feature.
     
  - 'Label': tf.train.Feature(int64_list=tf.train.Int64List(value = label))
     * Key is 'Label' and it stores whether image is 'Cat' or 'Dog'.


* Please refer to the following for which tf.train.XXX to use depending on the data type of value.
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
  - When it is ready, it is recorded as an actual TFRecord file.


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

* Dog is written in the same way as Cat.


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

* After a while, a folder called 'TFReocrd' will be created in the 'Cat' / 'Dog' folder and TFReocrd files with the same image file name but only the extension .tfrecord will be created in it.


* In the next post, I will do image classification using the TFReocrd files created in this way.

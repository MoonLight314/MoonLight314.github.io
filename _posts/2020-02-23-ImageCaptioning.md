---
title: "Image Captioning"
date: 2017-10-20 08:26:28 -0400
categories: project ImageCaption
---

개인적으로 대량의 사진에 Caption을 달아야 할 일이 생겨서 이것저것 알아보던 중에 찾은 예제입니다.   
원본은 [Image Captioning with Keras](https://towardsdatascience.com/image-captioning-with-keras-teaching-computers-to-describe-pictures-c88a46a311b8)이며
제가 개인적인 Comment와 설명을 위한 그림을 추가하였습니다.

이 예제에서는 Flickr 8k Image Data Set을 사용할 예정인데, Flickr 8k Data Set에는 총 8091개의 Image File이 있고 각 Image File당 5개의 Caption이 제공됩니다.   
'Flickr8k.token.txt'에 각 Image의 Caption이 저장되어 있습니다.
   
    
우선 필요한 Package를 Load합니다.   
주로 Keras를 사용할 예정이며, Image에서 Feature Extract를 위해 VGG16 Model을 사용할 예정입니다.   

```python
import numpy as np
from numpy import array
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline
import string
import os
from PIL import Image
import glob
from pickle import dump, load
from time import time
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import LSTM, Embedding, TimeDistributed, Dense, RepeatVector,\
                         Activation, Flatten, Reshape, concatenate, Dropout, BatchNormalization
from keras.optimizers import Adam, RMSprop
from keras.layers.wrappers import Bidirectional
from keras.layers.merge import add
from keras.applications.inception_v3 import InceptionV3
from keras.preprocessing import image
from keras.models import Model
from keras import Input, layers
from keras import optimizers
from keras.applications.inception_v3 import preprocess_input
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical

from tqdm import tqdm_notebook
```


```python
# load doc into memory
def load_doc(filename):
    # open the file as read only
    file = open(filename, 'r')
    # read all text
    text = file.read()
    # close the file
    file.close()
    return text

filename = "../Flickr8k_text/Flickr8k.token.txt"

# load descriptions
doc = load_doc(filename)

print(doc[:866])
```

    1000268201_693b08cb0e.jpg#0	A child in a pink dress is climbing up a set of stairs in an entry way .
    1000268201_693b08cb0e.jpg#1	A girl going into a wooden building .
    1000268201_693b08cb0e.jpg#2	A little girl climbing into a wooden playhouse .
    1000268201_693b08cb0e.jpg#3	A little girl climbing the stairs to her playhouse .
    1000268201_693b08cb0e.jpg#4	A little girl in a pink dress going into a wooden cabin .
    1001773457_577c3a7d70.jpg#0	A black dog and a spotted dog are fighting
    1001773457_577c3a7d70.jpg#1	A black dog and a tri-colored dog playing with each other on the road .
    1001773457_577c3a7d70.jpg#2	A black dog and a white dog with brown spots are staring at each other in the street .
    1001773457_577c3a7d70.jpg#3	Two dogs of different breeds looking at each other on the road .
    1001773457_577c3a7d70.jpg#4	Two dogs on pavement moving toward each other .
    
    
   
* 위와 같은 형태로 한 Line에 Image File Name과 Caption의 Index , Caption 문자열의 순서로 저장되어 있습니다.
   
   
* 이렇게 그냥 두고 사용하기엔 불편하니, 이후에 사용하기 쉽도록 Dict. 형태로 바꾸어 보겠습니다.
* Dict.의 Key는 Image File Name , Value는 5개의 Caption이 담긴 List형태로 만들겠습니다.


```python
def load_descriptions(doc):

    mapping = dict()

    # process lines
    for line in doc.split('\n'):
    
        # split line by white space
        tokens = line.split()
        
        if len(line) < 2:
            continue

        # take the first token as the image id, the rest as the description
        image_id, image_desc = tokens[0], tokens[1:]
        
        # extract filename from image id
        image_id = image_id.split('.')[0]
        
        # convert description tokens back to string
        image_desc = ' '.join(image_desc)
        
        # create the list if needed
        if image_id not in mapping:
            mapping[image_id] = list()
        
        # store description
        mapping[image_id].append(image_desc)
    
    return mapping

# parse descriptions
descriptions = load_descriptions(doc)
```


```python
print('Loaded: %d ' % len(descriptions))

print( descriptions['1000268201_693b08cb0e'] )
```

    Loaded: 8092 
    ['A child in a pink dress is climbing up a set of stairs in an entry way .', 'A girl going into a wooden building .', 'A little girl climbing into a wooden playhouse .', 'A little girl climbing the stairs to her playhouse .', 'A little girl in a pink dress going into a wooden cabin .']
    
    
    
   
   
* 위와 같이 우리가 원하는대로 Dict. 형태의 자료구조가 만들어졌습니다.
* Image File 갯수는 8091개인데, Caption File에는 총 8092개가 있네요.
<br/>
<br/>
<br/>
<br/>


![title](/assets/dict.png)

<br/>
<br/>
<br/>
<br/>
<br/>
<br/>
<br/>
<br/>
<br/>
<br/>
<br/>
<br/>

* Caption의 Text에 특별한 전처리를 하겠습니다.
  - 추후에 처리를 편하게 하기 위해 모든 문자를 소문자로 변환
  - 불필요한 문자를 삭제
  - 부정관사 , '&' , 숫자 삭제


```python
def clean_descriptions(descriptions):

    # prepare translation table for removing punctuation
    table = str.maketrans('', '', string.punctuation)
    
    for key, desc_list in descriptions.items():
        for i in range(len(desc_list)):
            desc = desc_list[i]
            
            # tokenize
            desc = desc.split()
            
            # convert to lower case
            desc = [word.lower() for word in desc]
            
            # remove punctuation from each token
            desc = [w.translate(table) for w in desc]
            
            # remove hanging 's' and 'a'
            desc = [word for word in desc if len(word)>1]
            
            # remove tokens with numbers in them
            desc = [word for word in desc if word.isalpha()]
            
            # store as string
            desc_list[i] =  ' '.join(desc)
```


```python
# clean descriptions
clean_descriptions(descriptions)
```
  
  
```python
print( descriptions['1000268201_693b08cb0e'] )
```
  
  
    ['child in pink dress is climbing up set of stairs in an entry way', 'girl going into wooden building', 'little girl climbing into wooden playhouse', 'little girl climbing the stairs to her playhouse', 'little girl in pink dress going into wooden cabin']
<br/>
<br/>
<br/>
<br/>
<br/>
<br/>
<br/>
<br/>

* 미리 만들어둔 Image File & Caption의 Dict.에서 Caption에서 사용된 단어들을 분리하여 Set로 만듭니다.
* Set은 중복을 허용하지 않는 Python의 자료구조입니다.
* set.update()는 복수개의 값을 Set에 추가하고자 할 때 사용하는 함수입니다.


```python
# convert the loaded descriptions into a vocabulary of words
def to_vocabulary(descriptions):
    
    # build a list of all description strings
    all_desc = set()
    
    for key in descriptions.keys():
        # key(Image File Name)당 5개의 Caption이 있으므로
        # 각각의 Caption에서 공백을 기준으로 Split하여 
        # set에 추가합니다.
        [all_desc.update(d.split()) for d in descriptions[key]]
        
    return all_desc
```


```python
# summarize vocabulary
vocabulary = to_vocabulary(descriptions)
print('Original Vocabulary Size: %d' % len(vocabulary))
```

    Original Vocabulary Size: 8763
    

   

   

   

   

* 깔끔하게 정리된 Caption을 저장해 둡니다.   


```python
# save descriptions to file, one per line
def save_descriptions(descriptions, filename):
    lines = list()
    for key, desc_list in descriptions.items():
        for desc in desc_list:
            lines.append(key + ' ' + desc)
    data = '\n'.join(lines)
    file = open(filename, 'w')
    file.write(data)
    file.close()
```


```python
save_descriptions(descriptions, 'descriptions.txt')
```
<br/>
<br/>
<br/>
<br/>
<br/>
<br/>
<br/>
<br/>

* Flicker 8K Data Set에는 친절하게도 Train에 사용할 6000개의 Image File Name List를 제공해 줍니다.
* Flickr_8k.trainImages.txt을 열어서 File Name을 Set에 담아둡시다.


```python
# load a pre-defined list of photo identifiers
def load_set(filename):
    doc = load_doc(filename)
    dataset = list()
    
    # process line by line
    for line in doc.split('\n'):
        
        # skip empty lines
        if len(line) < 1:
            continue
            
        # get the image identifier
        identifier = line.split('.')[0]
        dataset.append(identifier)
        
    return set(dataset)
```


```python
# load training dataset (6K)
filename = '../Flickr8k_text/Flickr_8k.trainImages.txt'
train = load_set(filename)
print('Dataset: %d' % len(train))
```

    Dataset: 6000
    

   

   

   

* Flicker 8K Dataset의 전체 File Name List를 뽑아놓습니다.   


```python
# Below path contains all the images
images = '../Flickr8k_Dataset/'

# Create a list of all image names in the directory
img = glob.glob(images + '*.jpg')
```


```python
print( len( img ) )
img[:10]
```

    8091
    ['../Flickr8k_Dataset\\1000268201_693b08cb0e.jpg',
     '../Flickr8k_Dataset\\1001773457_577c3a7d70.jpg',
     '../Flickr8k_Dataset\\1002674143_1b742ab4b8.jpg',
     '../Flickr8k_Dataset\\1003163366_44323f5815.jpg',
     '../Flickr8k_Dataset\\1007129816_e794419615.jpg',
     '../Flickr8k_Dataset\\1007320043_627395c3d8.jpg',
     '../Flickr8k_Dataset\\1009434119_febe49276a.jpg',
     '../Flickr8k_Dataset\\1012212859_01547e3f17.jpg',
     '../Flickr8k_Dataset\\1015118661_980735411b.jpg',
     '../Flickr8k_Dataset\\1015584366_dfcec3c85a.jpg']



   

   

   

* 전체 8091개 중에 Train Image의 Full Path를 뽑습니다.   


```python
# Below file conatains the names of images to be used in train data
train_images_file = '../Flickr8k_text/Flickr_8k.trainImages.txt'

# Read the train image names in a set
train_images = set(open(train_images_file, 'r').read().strip().split('\n'))

# Create a list of all the training images with their full path names
train_img = []

for i in img: # img is list of full path names of all images
    if i[len(images):] in train_images: # Check if the image belongs to training set
        train_img.append(i) # Add it to the list of train images
```


```python
print( len(train_img) )
train_img[:10]
```

    6000
    ['../Flickr8k_Dataset\\1000268201_693b08cb0e.jpg',
     '../Flickr8k_Dataset\\1001773457_577c3a7d70.jpg',
     '../Flickr8k_Dataset\\1002674143_1b742ab4b8.jpg',
     '../Flickr8k_Dataset\\1003163366_44323f5815.jpg',
     '../Flickr8k_Dataset\\1007129816_e794419615.jpg',
     '../Flickr8k_Dataset\\1007320043_627395c3d8.jpg',
     '../Flickr8k_Dataset\\1009434119_febe49276a.jpg',
     '../Flickr8k_Dataset\\1012212859_01547e3f17.jpg',
     '../Flickr8k_Dataset\\1015118661_980735411b.jpg',
     '../Flickr8k_Dataset\\1015584366_dfcec3c85a.jpg']


* Test에 사용할 Image File도 Flickr_8k.testImages.txt 내용을 참고해 Full Path List를 작성해 둡시다.


```python
# Below file conatains the names of images to be used in test data
test_images_file = '../Flickr8k_text/Flickr_8k.testImages.txt'
# Read the validation image names in a set# Read the test image names in a set
test_images = set(open(test_images_file, 'r').read().strip().split('\n'))

# Create a list of all the test images with their full path names
test_img = []

for i in img: # img is list of full path names of all images
    if i[len(images):] in test_images: # Check if the image belongs to test set
        test_img.append(i) # Add it to the list of test images
```


```python
print( len(test_img) )
test_img[:10]
```

    1000
    ['../Flickr8k_Dataset\\1056338697_4f7d7ce270.jpg',
     '../Flickr8k_Dataset\\106490881_5a2dd9b7bd.jpg',
     '../Flickr8k_Dataset\\1082379191_ec1e53f996.jpg',
     '../Flickr8k_Dataset\\1084040636_97d9633581.jpg',
     '../Flickr8k_Dataset\\1096395242_fc69f0ae5a.jpg',
     '../Flickr8k_Dataset\\1107246521_d16a476380.jpg',
     '../Flickr8k_Dataset\\1119015538_e8e796281e.jpg',
     '../Flickr8k_Dataset\\1122944218_8eb3607403.jpg',
     '../Flickr8k_Dataset\\1131800850_89c7ffd477.jpg',
     '../Flickr8k_Dataset\\1131932671_c8d17751b3.jpg']



   

   

   

* 나중에 Train할 때 사용하기 위해서, Train Image(6000개)의 Caption을 따로 뽑아서 저장합니다.


```python
# load clean descriptions into memory
def load_clean_descriptions(filename, dataset):
    # load document
    doc = load_doc(filename)
    
    descriptions = dict()
    
    for line in doc.split('\n'):
        
        # split line by white space
        tokens = line.split()
        
        # split id from description
        image_id, image_desc = tokens[0], tokens[1:]
        
        # skip images not in the set
        if image_id in dataset:
            # create list
            if image_id not in descriptions:
                descriptions[image_id] = list()
                
            # wrap description in tokens
            desc = 'startseq ' + ' '.join(image_desc) + ' endseq'
            
            # store
            descriptions[image_id].append(desc)
            
    return descriptions
```


```python
# descriptions
train_descriptions = load_clean_descriptions('descriptions.txt', train)
print('Descriptions: train=%d' % len(train_descriptions))
```

    Descriptions: train=6000
    


```python
train_descriptions['1000268201_693b08cb0e']
```




    ['startseq child in pink dress is climbing up set of stairs in an entry way endseq',
     'startseq girl going into wooden building endseq',
     'startseq little girl climbing into wooden playhouse endseq',
     'startseq little girl climbing the stairs to her playhouse endseq',
     'startseq little girl in pink dress going into wooden cabin endseq']
     
<br/>
<br/>
<br/>
<br/>
<br/>
<br/>
<br/>
<br/>
<br/>
<br/>
<br/>
<br/>

* Keras에서 제공해주는 수많은 Pre-Trained Model 중에서 우리는 Inception V3 Model을 사용하도록 하겠습니다.
  - https://cloud.google.com/tpu/docs/inception-v3-advanced
* 아래 Code를 실행하면 최초 한번 학습된 Model의 Weight값을 받아옵니다.
* 아래 Link에서 좀 더 자세한 내용을 확인 가능
  - https://keras.io/applications/


```python
# Load the inception v3 model
model = InceptionV3(weights='imagenet')
```

    WARNING:tensorflow:From C:\Users\csyi\AppData\Local\Continuum\anaconda3\lib\site-packages\keras\backend\tensorflow_backend.py:74: The name tf.get_default_graph is deprecated. Please use tf.compat.v1.get_default_graph instead.
    
    WARNING:tensorflow:From C:\Users\csyi\AppData\Local\Continuum\anaconda3\lib\site-packages\keras\backend\tensorflow_backend.py:517: The name tf.placeholder is deprecated. Please use tf.compat.v1.placeholder instead.
    
    WARNING:tensorflow:From C:\Users\csyi\AppData\Local\Continuum\anaconda3\lib\site-packages\keras\backend\tensorflow_backend.py:4138: The name tf.random_uniform is deprecated. Please use tf.random.uniform instead.
    
    WARNING:tensorflow:From C:\Users\csyi\AppData\Local\Continuum\anaconda3\lib\site-packages\keras\backend\tensorflow_backend.py:174: The name tf.get_default_session is deprecated. Please use tf.compat.v1.get_default_session instead.
    
    WARNING:tensorflow:From C:\Users\csyi\AppData\Local\Continuum\anaconda3\lib\site-packages\keras\backend\tensorflow_backend.py:181: The name tf.ConfigProto is deprecated. Please use tf.compat.v1.ConfigProto instead.
    
    WARNING:tensorflow:From C:\Users\csyi\AppData\Local\Continuum\anaconda3\lib\site-packages\keras\backend\tensorflow_backend.py:186: The name tf.Session is deprecated. Please use tf.compat.v1.Session instead.
    
    WARNING:tensorflow:From C:\Users\csyi\AppData\Local\Continuum\anaconda3\lib\site-packages\keras\backend\tensorflow_backend.py:190: The name tf.global_variables is deprecated. Please use tf.compat.v1.global_variables instead.
    
    WARNING:tensorflow:From C:\Users\csyi\AppData\Local\Continuum\anaconda3\lib\site-packages\keras\backend\tensorflow_backend.py:199: The name tf.is_variable_initialized is deprecated. Please use tf.compat.v1.is_variable_initialized instead.
    
    WARNING:tensorflow:From C:\Users\csyi\AppData\Local\Continuum\anaconda3\lib\site-packages\keras\backend\tensorflow_backend.py:206: The name tf.variables_initializer is deprecated. Please use tf.compat.v1.variables_initializer instead.
    
    WARNING:tensorflow:From C:\Users\csyi\AppData\Local\Continuum\anaconda3\lib\site-packages\keras\backend\tensorflow_backend.py:1834: The name tf.nn.fused_batch_norm is deprecated. Please use tf.compat.v1.nn.fused_batch_norm instead.
    
    WARNING:tensorflow:From C:\Users\csyi\AppData\Local\Continuum\anaconda3\lib\site-packages\keras\backend\tensorflow_backend.py:133: The name tf.placeholder_with_default is deprecated. Please use tf.compat.v1.placeholder_with_default instead.
    
    WARNING:tensorflow:From C:\Users\csyi\AppData\Local\Continuum\anaconda3\lib\site-packages\keras\backend\tensorflow_backend.py:3976: The name tf.nn.max_pool is deprecated. Please use tf.nn.max_pool2d instead.
    
    WARNING:tensorflow:From C:\Users\csyi\AppData\Local\Continuum\anaconda3\lib\site-packages\keras\backend\tensorflow_backend.py:3980: The name tf.nn.avg_pool is deprecated. Please use tf.nn.avg_pool2d instead.
    
    

   

   

   

* 이제 기존에 학습된 Inception V3 Model을 우리가 쓸 수 있도록 약간 수정하도록 하겠습니다.
* Inception V3 Model의 마지막 Dense Layer(Classifier , Softmax Layer)만 빼버리고 사용하도록 하겠습니다.
* Inception V3 Model은 ImageNet의 Dataset을 Training Set으로 학습된 Model이기 때문입니다.
* 마지막 Dense Layer만 뺀 Model은 Image의 Feature들을 Extract하는 기능만 가지고 있습니다.
* 우리는 Extract된 Feature들만 이용하여 Image Caption을 학습하는데 사용하도록 하겠습니다.

![title](/assets/Inception_V3_01.png)

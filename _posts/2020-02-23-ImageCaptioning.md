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
  - [https://cloud.google.com/tpu/docs/inception-v3-advanced](https://cloud.google.com/tpu/docs/inception-v3-advanced)
* 아래 Code를 실행하면 최초 한번 학습된 Model의 Weight값을 받아옵니다.
* 아래 Link에서 좀 더 자세한 내용을 확인 가능
  - [https://keras.io/applications/](https://keras.io/applications/)


```python
# Load the inception v3 model
model = InceptionV3(weights='imagenet')
```
<br/>
<br/>
<br/>
<br/>
<br/>

* 이제 기존에 학습된 Inception V3 Model을 우리가 쓸 수 있도록 약간 수정하도록 하겠습니다.
* Inception V3 Model의 마지막 Dense Layer(Classifier , Softmax Layer)만 빼버리고 사용하도록 하겠습니다.
* Inception V3 Model은 ImageNet의 Dataset을 Training Set으로 학습된 Model이기 때문입니다.
* 마지막 Dense Layer만 뺀 Model은 Image의 Feature들을 Extract하는 기능만 가지고 있습니다.
* 우리는 Extract된 Feature들만 이용하여 Image Caption을 학습하는데 사용하도록 하겠습니다.
<br/>
<br/>
<br/>
<br/>
<br/>

![title](/assets/Inception_V3_01.png)

<br/>
<br/>
<br/>
<br/>
<br/>


* Keras의 함수형 API를 사용하여 새로운 Model을 만듭니다.



```python
# Create a new model, by removing the last layer (output layer) from the inception v3
model_new = Model(model.input, model.layers[-2].output)
```

   

* 기존 Model은 총 313개의 Layer가 있고, 마지막 Layer는 Softmax Layer이며, Classifier의 기능을 합니다.   


```python
print( type(model.layers) )
print( len(model.layers) )
print( model.layers[0] )
print( model.layers[1] )

print( model.layers[-2] )
print( model.layers[-1] )
```

    <class 'list'>
    313
    <keras.engine.input_layer.InputLayer object at 0x00000237899DE788>
    <keras.layers.convolutional.Conv2D object at 0x00000237899DE148>
    <keras.layers.pooling.GlobalAveragePooling2D object at 0x000002380F475408>
    <keras.layers.core.Dense object at 0x000002380F48DD08>

<br/>
<br/>
<br/>
<br/>
<br/>
<br/>

* 수정된 Model은 Layer수가 한개 줄었으며, 마지막층은 Fully Connected Layer이며, Feature Extractor 역할을 합니다.


```python
print( len(model_new.layers) )
print( model_new.layers[0] )
print( model_new.layers[1] )

print( model_new.layers[-2] )
print( model_new.layers[-1] )
```

    312
    <keras.engine.input_layer.InputLayer object at 0x00000237899DE788>
    <keras.layers.convolutional.Conv2D object at 0x00000237899DE148>
    <keras.layers.merge.Concatenate object at 0x000002380F468B08>
    <keras.layers.pooling.GlobalAveragePooling2D object at 0x000002380F475408>

<br/>
<br/>
<br/>

* Training Set의 6000개의 Image File에 대해서 Inception V3 Model을 이용해 Feature를 추출해 냅니다.

![title](/assets/encode.png)

<br/>
<br/>
<br/>
<br/>
<br/>
<br/>

```python
def preprocess(image_path):
    
    # Convert all the images to size 299x299 as expected by the inception v3 model
    # load_img(path, grayscale = FALSE, target_size = NULL)
    img = image.load_img(image_path, target_size=(299, 299))
    
    # Convert PIL image to numpy array of 3-dimensions
    # img_to_array(img, data_format = NULL)
    x = image.img_to_array(img) # (299, 299, 3)
    
    # Add one more dimension
    x = np.expand_dims(x, axis=0)    # (1, 299, 299, 3)
    
    # preprocess the images using preprocess_input() from inception module
    # preprocess_input()은 미리 학습된 Inception V3 Model에 맞게 Image Data를 변환시켜 줍니다.
    # 실제 여기서 하는 일은 -1 ~ 1 사이의 값으로 변환시켜 줍니다.
    x = preprocess_input(x)   # (1, 299, 299, 3)
    
    return x
```


```python
# Function to encode a given image into a vector of size (2048, )
def encode(img):
    img = preprocess(img) # preprocess the image
    
    fea_vec = model_new.predict( img ) # Get the encoding vector for the image
    fea_vec = np.reshape(fea_vec, fea_vec.shape[1]) # reshape from (1, 2048) to (2048, )
    
    return fea_vec
```

<br/>
<br/>
<br/>
<br/>

* img(train_img)는 Train에 사용할 Image들의 Full Path를 담고 있습니다.
* img[len(images):]는 Full Path의 앞쪽 부분을 잘라내고, File Name만으로 Dict.의 Key로 사용하겠다는 의미입니다.
* encoding_train Dict.()는 각 Train Image들을 앞에서 선언한 Inception V3의 마지막 Classifier부분을 제외한 Model에 넣어  
Feature만 모아 놓은 것입니다.


```python
import pickle

if os.path.isfile("encoded_train_images.pkl"):
    train_features = load(open("encoded_train_images.pkl", "rb"))
    print('Photos: train=%d' % len(train_features))

else:
    # Call the funtion to encode all the train images
    # This will take a while on CPU - Execute this only once
    start = time()
    encoding_train = {}

    for img in tqdm_notebook( train_img ):
        # images : '../Flicker8k_Dataset/'
        # img[len(images):] : Training Image에서 Full Path 앞부분을 잘라낸, Image File Name
        encoding_train[img[len(images):]] = encode(img)

    print("Time taken in seconds =", time()-start)   

    # * Training Image들의 Feature Vector를 File로 저장해 놓읍시다.
    with open("encoded_train_images.pkl", "wb") as encoded_pickle:
        pickle.dump(encoding_train, encoded_pickle)
```

    Photos: train=6000
    
<br/>
<br/>
<br/>
<br/>

* Test에 사용할 Image들도 마찬가지로 Feature Vector를 만들어 놓읍시다.   


```python
if os.path.isfile("encoded_test_images.pkl"):
    with open("encoded_test_images.pkl", "rb") as encoded_pickle:
        encoding_test = load(encoded_pickle)
    
else:
    # Call the funtion to encode all the test images - Execute this only once
    start = time()
    encoding_test = {}

    for img in tqdm_notebook( test_img ):
        encoding_test[img[len(images):]] = encode(img)

    print("Time taken in seconds =", time()-start)
    
    # Save the bottleneck test features to disk
    with open("encoded_test_images.pkl", "wb") as encoded_pickle:
        pickle.dump(encoding_test, encoded_pickle)
```
<br/>
<br/>
<br/>
<br/>
<br/>
<br/>
<br/>
<br/>
<br/>

### 이제 Train Set의 Caption에서 사용된 개별 단어들의 목록을 만들어 보겠습니다.   

   

* 단어들을 수치화하고 Indexing을 하는 이유는 최종적으로 Supervised Learning을 하기 위해서는     
  어떤 형태의 Data이든지 이를 수치화하여 하기 때문입니다.

   

* Train Data의 모든 Caption을 하나의 List로 만듭니다.
* 6000개 x 각각 5개의 Caption = 30000


```python
# Create a list of all the training captions
all_train_captions = []

for key, val in train_descriptions.items():
    for cap in val:
        all_train_captions.append(cap)

len(all_train_captions)
```




    30000




```python
all_train_captions[:10]
```




    ['startseq child in pink dress is climbing up set of stairs in an entry way endseq',
     'startseq girl going into wooden building endseq',
     'startseq little girl climbing into wooden playhouse endseq',
     'startseq little girl climbing the stairs to her playhouse endseq',
     'startseq little girl in pink dress going into wooden cabin endseq',
     'startseq black dog and spotted dog are fighting endseq',
     'startseq black dog and tricolored dog playing with each other on the road endseq',
     'startseq black dog and white dog with brown spots are staring at each other in the street endseq',
     'startseq two dogs of different breeds looking at each other on the road endseq',
     'startseq two dogs on pavement moving toward each other endseq']
     
     
<br/>
<br/>
<br/>
<br/>

* 다음과 같이 Caption들의 전체 단어의 출현 빈도를 Count해서 List에 채웁니다.   
* 10번 이하로 나온 단어는 버립니다.   


```python
# Consider only words which occur at least 10 times in the corpus
word_count_threshold = 10
word_counts = {}
nsents = 0

for sent in all_train_captions:    
    nsents += 1
    
    for w in sent.split(' '):
        word_counts[w] = word_counts.get(w, 0) + 1

vocab = [w for w in word_counts if word_counts[w] >= word_count_threshold]
print('preprocessed words %d -> %d' % (len(word_counts), len(vocab)))
```

    preprocessed words 7578 -> 1651
    

   


```python
vocab[:20]
```




    ['startseq',
     'child',
     'in',
     'pink',
     'dress',
     'is',
     'climbing',
     'up',
     'set',
     'of',
     'stairs',
     'an',
     'way',
     'endseq',
     'girl',
     'going',
     'into',
     'wooden',
     'building',
     'little']



   

   

* 이 단어들의 List를 좀 더 편하게 사용하기 위해서 단어와 그 단어에 해당하는 Index를 개별적으로 담고 있는 각각의 Dict.를
정의합니다.


```python
ixtoword = {}
wordtoix = {}

ix = 1
for w in vocab:
    wordtoix[w] = ix
    ixtoword[ix] = w
    ix += 1
```


```python
wordtoix['in']
```




    3




```python
ixtoword[3]
```




    'in'




```python
vocab_size = len(ixtoword) + 1 # one for appended 0's
vocab_size
```




    1652



   

   

* Caption 중에 가장 긴 단어를 가진 Caption의 단어 갯수는 34개네요
* 이 값은 이후 LSTM에 들어갈 Word Data 전체 길이를 결정할 때 사용됩니다.


```python
# convert a dictionary of clean descriptions to a list of descriptions
def to_lines(descriptions):
    all_desc = list()
    for key in descriptions.keys():
        [all_desc.append(d) for d in descriptions[key]]
    return all_desc

# calculate the length of the description with the most words
def max_length(descriptions):
    lines = to_lines(descriptions)    
    return max(len(d.split()) for d in lines)
```


```python
# determine the maximum sequence length
max_length = max_length(train_descriptions)

print('Description Length: %d' % max_length)
```

    Description Length: 34
    
<br/>
<br/>
<br/>
<br/>
<br/>
<br/>
<br/>
<br/>

### Feature Engineering ( Data Generator )

   

* ML or DL의 Training에 사용할 Data는 Integer or Float 형태의 Numberical Value이어야 합니다..
* Training Image Set과 Caption을 어떤 방식으로 RNN에 넣을 수 있는 Numberical Value로 변환하는지 알아보겠습니다.
<br/>
<br/>
<br/>

* 다음과 같은 그림이 있다고 가정해보자. 이 Image과 이 Image의 Caption( the black cat sat on grass )을 RNN에 넣을 수 있는 형태로 바꿔보겠습니다.

<br/>
<br/>
<br/>
<br/>
<br/>

![title](/assets/black_cat.jpeg) 
<br/>
<br/>
<br/>
<br/>
<br/>

* 우선 Image에 대한 Data와 Caption에 대한 정보를 분리해서 생각해봅시다.
* Image에 대한 Feature는 앞서 Pre-Trained Model인 Inception V3 Model에서 Classifier를 제거한 Model을 이용해 Feature를 뽑을 수 있습니다.
* New Inception V3 Model은 Image에 대해서 2048개의 Feature를 뽑아 줄 것입니다.

* 우리가 Predict하려고 하는 값은, 임의의 Image에 대한 Caption입니다. 
* 그래서, Training시에 Label은 Caption이 될 것입니다.

* 그러나, 한 번의 Predict로 전체 Caption을 전부 알아내는 것이 아니라, Caption의 일부로 다음 Caption의 단어를 유추하는 방식을 사용할 것입니다.
* 이와 같이 Sequence 형태의 Data를 다루기 위해서 우리는 RNN(Recurrent Neural Network)을 사용할 것이고,  
  이에 따라 RNN Input에 적절한 형태의 Training Data 형태를 만들어야 합니다.
  
* 즉, 아래와 같은 형태의 Training Data Feature Format을 사용할 것입니다.
<br/>
<br/>
<br/>
<br/>

![title](/assets/Data_Gen_01.png)   

<br/>
<br/>
<br/>
<br/>

* RNN에 입력될 Caption의 Word를 Numberical Value로 변환하기 위해서 앞에서 작성한 wordtoix를 이용하여  
  아래와 같이 Word를 Index로 변환해 주어야 한다.

![title](/assets/Data_Gen_02.png)

<br/>
<br/>
<br/>
<br/>

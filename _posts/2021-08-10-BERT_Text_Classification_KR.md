---
title: "BERT Text Classification"
date: 2021-08-10 08:26:28 -0400
categories: Deep Learning
---
### BERT Text Classification

<br>
<br>
<br>
<br>
<br>
<br>

* 이번 Post에서는 BERT Model을 이용하여, Text 분류 작업을 해보도록 하겠습니다.   

<br>

* 영화 감상평이 긍정적인지 부정적인지 분류해 놓은 Data Set을 이용할 예정입니다.

<br>

* 실제로 사용할 Data Set은 [Large Movie Review Dataset](https://ai.stanford.edu/~amaas/data/sentiment/) 입니다.

<br>
<br>
<br>
<br>
<br>

## 0. About BERT

<br>

* BERT 및 기타 Transformer Encoder Architecture는 Natural Language Process의 다양한 분야에서 좋은 성능을 보여주고 있습니다.

<br>

* BERT or Transformer Encoder는 Natural Language Process의 다양한 분야에 사용할 수 있는 Vector Space를 계산해 줍니다.

<br>

* BERT(Bidirectional Encoder Representations from Transformers)는 이름에서 유추할 수 있듯이, Transformer Encoder를 사용하여 전체 Text의 각 Toekn의 앞/뒤 Token을 고려하여 Token을 처리하는 구조를 가지고 있습니다.

<br>

* BERT는 large text corpus에서 Train한 후에, 특정 Task에 적합하도록 Fine Tuning을 합니다.

<br>

* BERT의 Paper는 아래 Link를 참고하시면 됩니다.

   [BERT(Bidirectional Encoder Representations from Transformers)](https://arxiv.org/abs/1810.04805)


<br>
<br>
<br>
<br>
<br>

## 1. Load Package

<br>

* Tensorflow를 사용할 예정이며, GPU가 제대로 동작하는지도 한 번 확인해 보았습니다.   

<br>

```python
import os
import shutil

import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_text as text

import matplotlib.pyplot as plt
import tensorflow_addons as tfa

tf.get_logger().setLevel('ERROR')
```


```python
tf.__version__
```




    '2.5.0'




```python
tf.config.list_physical_devices('GPU')
```




    [PhysicalDevice(name='/physical_device:GPU:0', device_type='GPU')]

<br>
<br>
<br>
<br>
<br>

## 2. Download the IMDB dataset

<br>

* 이번 Post에서 사용할 IMDB Dataset에 대해서 알아보겠습니다.

<br>

* IMDB Dataset에 대해서는 들어보신 분이 많을 것입니다. 영화 Review Site에서 영화에 대한 긍정/부정 의견을 모아서 만든 Dataset입니다.

<br>

* 아래 Code로 Download할 수 있습니다.

<br>
<br>

```python
url = 'https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz'

dataset = tf.keras.utils.get_file('aclImdb_v1.tar.gz', url,
                                  untar=True, cache_dir='.',
                                  cache_subdir='')

dataset_dir = os.path.join(os.path.dirname(dataset), 'aclImdb')

train_dir = os.path.join(dataset_dir, 'train')

# remove unused folders to make it easier to load the data
remove_dir = os.path.join(train_dir, 'unsup')
shutil.rmtree(remove_dir)
```

<br>
<br>

* Dataset을 받으면 자동으로 Unzip을 해줍니다.

<br>

* 'aclImdb'라는 Folder가 생기고 그 아래에 다음과 같은 구조로 Folder가 구성되어 있습니다.

<br>

<p align="left">
  <img src="/assets/BERT_Text_Classification/pic_00.png">
</p>

<br>

* 친절하게도 Train / Test Data가 분리되어 있네요.

<br>

* Folder 구조가 Train / Test으로 이미 나누어져 있고, 게다가 긍정/부정도 Folder로 나누어져 있습니다.

<br>

* 이런 상황에서 사용하기 좋은 함수가 text_dataset_from_directory()입니다.

<br>

* text_dataset_from_directory()에 대한 자세한 사항은 아래 Link에서 확인하도록 합니다.
  https://www.tensorflow.org/api_docs/python/tf/keras/preprocessing/text_dataset_from_directory

<br>

* Test Data은 이미 있으니, Train Data를 이용하여 Validation Data를 만들도록 합니다.

<br>
<br>

* Dataset을 만들어 봅시다.   

<br>

```python
AUTOTUNE = tf.data.AUTOTUNE

# Batch Size는 개별 상황에 맞게 적절하게 바꿔주세요
#batch_size = 32
batch_size = 4

seed = 42
```

<br>
<br>
<br>

* 먼저 Train Folder에 있는 Data로 Train Dataset / Validation Dataset 나눕니다.

<br>

* 'validation_split' 과 'subset'을 이용해서 나눌 수 있는데, 'subset'은  "training" or "validation"을 가질 수 있고, 'validation_split'이 설정된 경우에 기능을 합니다.

<br>

* 먼저 Train Dataset을 만듭니다.

<br>

```python
raw_train_ds = tf.keras.preprocessing.text_dataset_from_directory(
    'aclImdb/train',
    batch_size=batch_size,
    validation_split=0.2,
    subset='training',
    seed=seed)

class_names = raw_train_ds.class_names
train_ds = raw_train_ds.cache().prefetch(buffer_size=AUTOTUNE)
```

    Found 25000 files belonging to 2 classes.
    Using 20000 files for training.

<br>
<br>

* 그리고, Validation Dataset을 만듭니다.   

<br>

```python
val_ds = tf.keras.preprocessing.text_dataset_from_directory(
    'aclImdb/train',
    batch_size=batch_size,
    validation_split=0.2,
    subset='validation',
    seed=seed)

val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)
```

    Found 25000 files belonging to 2 classes.
    Using 5000 files for validation.

<br>
<br>

* 마지막으로, Test Dataset을 만듭니다.   

<br>

```python
test_ds = tf.keras.preprocessing.text_dataset_from_directory(
    'aclImdb/test',
    batch_size=batch_size)

test_ds = test_ds.cache().prefetch(buffer_size=AUTOTUNE)
```

    Found 25000 files belonging to 2 classes.
    
<br>

```python
type(class_names)
```

<br>


    list

<br>


```python
class_names
```

<br>


    ['neg', 'pos']

<br>
<br>

* Data가 잘 만들어 졌는지 한 번 살펴보겠습니다.

<br>

```python
for text_batch, label_batch in train_ds.take(1):
    for i in range(3):
        print(f'Review: {text_batch.numpy()[i]}')
        label = label_batch.numpy()[i]
        print(f'Label : {label} ({class_names[label]})')
```

<br>

    Review: b'Mr Perlman gives a standout performance (as usual). Sadly, he has to struggle with an underwritten script and some nonsensical set pieces.<br /><br />Larsen is in "Die Hard" mode complete with singlet and bulging muscles, I\'m sure he could do better but seems satisfied to grimace and snarl through his part.<br /><br />The lovely Erika is very decorative (even though fully clothed!) and shows some signs of "getting" acting at last.<br /><br />SFX are mainly poor CGI and steals from other movies.<br /><br />The shootouts are pitiful - worthy of the A-Team<br /><br />Not even worth seeing for Perlman - AVOID'
    Label : 0 (neg)
    Review: b"During the whole Pirates of The Caribbean Trilogy Craze Paramount Pictures really dropped the ball in restoring this Anthony Quinn directed Cecil B. DeMille supervised movie and getting it on DVD and Blu Ray with all the extras included. It is obvious to me that Paramount Pictures Execs are blind as bats and ignorant of the fact that they have a really good pirate movie in their vault about a real pirate who actually lived in New Orleans, Louisiana which would have helped make The Crescent City once again famous for it's Pirate Connections. When the Execs at Paramount finally get with the program and release this movie in digital format then I will be a happy camper. Paramount Pictures it is up to you to get off your duff and get this film restored now !"
    Label : 1 (pos)
    Review: b"1st watched 12/7/2002 - 3 out of 10(Dir-Steve Purcell): Typical Mary Kate & Ashley fare with a few more kisses. It looks to me like the girls are getting pretty tired of this stuff and it will be interesting what happens to them if they ever decide to split up and go there own ways. In this episode of their adventures they are interns in Rome for a `fashion' designer who puts them right into the mailroom to learn what working hard is all about(I guess..). Besides the typical flirtations with boys there is nothing much else except the Rome scenario until about \xc2\xbe way into the movie when it's finally revealed why they are getting fired, then re-hired, then fired again, then re-hired again. This is definetly made by people who don't understand the corporate world and it shows in their interpretation of it. Maybe the real world will be their next adventure(if there is one.). Even my kids didn't seem to care for this boring `adventure' in the make-believe. Let's see they probably only have a couple of years till their legal adults. We'll see what happens then."
    Label : 0 (neg)

<br>

* 위와 같이 시청자가 작성한 Review와 함께 그 Review가 긍정적인지 부정적인지 나타내는 Label이 있습니다.   

<br>
<br>
<br>
<br>
<br>


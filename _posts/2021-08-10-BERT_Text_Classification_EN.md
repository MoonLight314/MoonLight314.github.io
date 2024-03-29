---
title: "BERT Text Classification (EN)"
date: 2021-08-10 08:26:28 -0400
categories: Deep Learning
---
# BERT Text Classification

<br>
<br>
<br>
<br>
<br>
<br>

* In this post, we will use the BERT model to classify text.

<br>

* We plan to use a data set that classifies whether movie reviews are positive or negative.

<br>

* The data set to actually use is [Large Movie Review Dataset](https://ai.stanford.edu/~amaas/data/sentiment/).

<br>
<br>
<br>
<br>
<br>

## 0. About BERT

<br>

* BERT and other transformer encoder architectures are making good performance in various fields of Natural Language Process.

<br>

* BERT or transformer encoder calculates vector space that can be used in various fields of Natural Language Process.

<br>

* BERT (Bidirectional Encoder Representations from Transformers), as inferred from the name, has a structure that uses the transformer encoder to process tokens by considering the tokens before and after each token of the entire text.

<br>

* After training in a large text corpus, BERT performs fine tuning to apply to a specific task.

<br>

* You can refer to the link below for BERT's paper.

   [BERT(Bidirectional Encoder Representations from Transformers)](https://arxiv.org/abs/1810.04805)


<br>
<br>
<br>
<br>
<br>

## 1. Load Package

<br>

* I'm going to use Tensorflow, and check that the GPU is working properly.

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

* Let's take a look at IMDB Dataset to be used in this post.

<br>

* Many of you may have heard of the IMDB dataset. This is a dataset created by collecting positive/negative opinions about movies from the movie review site.

<br>

* You can download by the below code.

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

* The code automatically unzip after downloading.

<br>

* A folder named 'aclImdb' is created, and the folder is composed of the following structure below it.

<br>

<p align="left">
  <img src="/assets/BERT_Text_Classification/pic_00.png">
</p>

<br>

* Kindly note that Train / Test Data is separated.

<br>

* The folder structure is already divided into Train / Test, and positive/negative is also divided into folders.

<br>

* A good function to use in this situation is **text_dataset_from_directory().**

<br>

* For details about **text_dataset_from_directory()**, check the link below.
  [https://www.tensorflow.org/api_docs/python/tf/keras/preprocessing/text_dataset_from_directory](https://www.tensorflow.org/api_docs/python/tf/keras/preprocessing/text_dataset_from_directory)

<br>

* Since test data already exists, so make validation data from train data.

<br>
<br>

* Let's make dataset.

<br>

```python
AUTOTUNE = tf.data.AUTOTUNE

# Please change the batch size appropriately according to individual circumstances.
#batch_size = 32
batch_size = 4

seed = 42
```

<br>
<br>
<br>

* First, divide train dataset / validation dataset by data in train folder.

<br>

* It can be divided using 'validation_split' and 'subset'. 'subset' can have "training" or "validation", and works only when 'validation_split' is set.

<br>

* First, let's make train dataset.

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

* And then, making validation dataset.

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

* Lastly, making test dataset.

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

* Let's take a look at whether the data is well created.

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

* As above, along with the review written by the viewer, there is a label indicating whether the review is positive or negative.

<br>
<br>
<br>
<br>
<br>

## 3. Select BERT Model

<br>

* BERT has various scale models for different purposes.

<br>

* You can check the types in the link below.
  [https://tfhub.dev/google/collections/bert/1](https://tfhub.dev/google/collections/bert/1)

<br>
<br>
  
* In addition, a function that preprocesses text according to the input shape for each model is also prepared.

<br>
<br>

* Let's begin with Small BERT model.

<br>

* Let's set it to small_bert/bert_en_uncased_L-4_H-512_A-8 of a suitable size.

<br>

* The code below maps the location of the model we set and the corresponding preprocessor.

<br>
<br>

```python
bert_model_name = 'small_bert/bert_en_uncased_L-4_H-512_A-8' 

map_name_to_handle = {
    'bert_en_uncased_L-12_H-768_A-12':
        'https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/3',
    'bert_en_cased_L-12_H-768_A-12':
        'https://tfhub.dev/tensorflow/bert_en_cased_L-12_H-768_A-12/3',
    'bert_multi_cased_L-12_H-768_A-12':
        'https://tfhub.dev/tensorflow/bert_multi_cased_L-12_H-768_A-12/3',
    'small_bert/bert_en_uncased_L-2_H-128_A-2':
        'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-2_H-128_A-2/1',
    'small_bert/bert_en_uncased_L-2_H-256_A-4':
        'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-2_H-256_A-4/1',
    'small_bert/bert_en_uncased_L-2_H-512_A-8':
        'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-2_H-512_A-8/1',
    'small_bert/bert_en_uncased_L-2_H-768_A-12':
        'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-2_H-768_A-12/1',
    'small_bert/bert_en_uncased_L-4_H-128_A-2':
        'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-4_H-128_A-2/1',
    'small_bert/bert_en_uncased_L-4_H-256_A-4':
        'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-4_H-256_A-4/1',
    'small_bert/bert_en_uncased_L-4_H-512_A-8':
        'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-4_H-512_A-8/1',
    'small_bert/bert_en_uncased_L-4_H-768_A-12':
        'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-4_H-768_A-12/1',
    'small_bert/bert_en_uncased_L-6_H-128_A-2':
        'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-6_H-128_A-2/1',
    'small_bert/bert_en_uncased_L-6_H-256_A-4':
        'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-6_H-256_A-4/1',
    'small_bert/bert_en_uncased_L-6_H-512_A-8':
        'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-6_H-512_A-8/1',
    'small_bert/bert_en_uncased_L-6_H-768_A-12':
        'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-6_H-768_A-12/1',
    'small_bert/bert_en_uncased_L-8_H-128_A-2':
        'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-8_H-128_A-2/1',
    'small_bert/bert_en_uncased_L-8_H-256_A-4':
        'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-8_H-256_A-4/1',
    'small_bert/bert_en_uncased_L-8_H-512_A-8':
        'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-8_H-512_A-8/1',
    'small_bert/bert_en_uncased_L-8_H-768_A-12':
        'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-8_H-768_A-12/1',
    'small_bert/bert_en_uncased_L-10_H-128_A-2':
        'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-10_H-128_A-2/1',
    'small_bert/bert_en_uncased_L-10_H-256_A-4':
        'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-10_H-256_A-4/1',
    'small_bert/bert_en_uncased_L-10_H-512_A-8':
        'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-10_H-512_A-8/1',
    'small_bert/bert_en_uncased_L-10_H-768_A-12':
        'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-10_H-768_A-12/1',
    'small_bert/bert_en_uncased_L-12_H-128_A-2':
        'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-12_H-128_A-2/1',
    'small_bert/bert_en_uncased_L-12_H-256_A-4':
        'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-12_H-256_A-4/1',
    'small_bert/bert_en_uncased_L-12_H-512_A-8':
        'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-12_H-512_A-8/1',
    'small_bert/bert_en_uncased_L-12_H-768_A-12':
        'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-12_H-768_A-12/1',
    'albert_en_base':
        'https://tfhub.dev/tensorflow/albert_en_base/2',
    'electra_small':
        'https://tfhub.dev/google/electra_small/2',
    'electra_base':
        'https://tfhub.dev/google/electra_base/2',
    'experts_pubmed':
        'https://tfhub.dev/google/experts/bert/pubmed/2',
    'experts_wiki_books':
        'https://tfhub.dev/google/experts/bert/wiki_books/2',
    'talking-heads_base':
        'https://tfhub.dev/tensorflow/talkheads_ggelu_bert_en_base/1',
}

map_model_to_preprocess = {
    'bert_en_uncased_L-12_H-768_A-12':
        'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
    'bert_en_cased_L-12_H-768_A-12':
        'https://tfhub.dev/tensorflow/bert_en_cased_preprocess/3',
    'small_bert/bert_en_uncased_L-2_H-128_A-2':
        'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
    'small_bert/bert_en_uncased_L-2_H-256_A-4':
        'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
    'small_bert/bert_en_uncased_L-2_H-512_A-8':
        'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
    'small_bert/bert_en_uncased_L-2_H-768_A-12':
        'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
    'small_bert/bert_en_uncased_L-4_H-128_A-2':
        'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
    'small_bert/bert_en_uncased_L-4_H-256_A-4':
        'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
    'small_bert/bert_en_uncased_L-4_H-512_A-8':
        'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
    'small_bert/bert_en_uncased_L-4_H-768_A-12':
        'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
    'small_bert/bert_en_uncased_L-6_H-128_A-2':
        'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
    'small_bert/bert_en_uncased_L-6_H-256_A-4':
        'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
    'small_bert/bert_en_uncased_L-6_H-512_A-8':
        'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
    'small_bert/bert_en_uncased_L-6_H-768_A-12':
        'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
    'small_bert/bert_en_uncased_L-8_H-128_A-2':
        'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
    'small_bert/bert_en_uncased_L-8_H-256_A-4':
        'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
    'small_bert/bert_en_uncased_L-8_H-512_A-8':
        'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
    'small_bert/bert_en_uncased_L-8_H-768_A-12':
        'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
    'small_bert/bert_en_uncased_L-10_H-128_A-2':
        'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
    'small_bert/bert_en_uncased_L-10_H-256_A-4':
        'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
    'small_bert/bert_en_uncased_L-10_H-512_A-8':
        'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
    'small_bert/bert_en_uncased_L-10_H-768_A-12':
        'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
    'small_bert/bert_en_uncased_L-12_H-128_A-2':
        'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
    'small_bert/bert_en_uncased_L-12_H-256_A-4':
        'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
    'small_bert/bert_en_uncased_L-12_H-512_A-8':
        'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
    'small_bert/bert_en_uncased_L-12_H-768_A-12':
        'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
    'bert_multi_cased_L-12_H-768_A-12':
        'https://tfhub.dev/tensorflow/bert_multi_cased_preprocess/3',
    'albert_en_base':
        'https://tfhub.dev/tensorflow/albert_en_preprocess/3',
    'electra_small':
        'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
    'electra_base':
        'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
    'experts_pubmed':
        'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
    'experts_wiki_books':
        'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
    'talking-heads_base':
        'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
}

tfhub_handle_encoder = map_name_to_handle[bert_model_name]
tfhub_handle_preprocess = map_model_to_preprocess[bert_model_name]

print(f'BERT model selected           : {tfhub_handle_encoder}')
print(f'Preprocess model auto-selected: {tfhub_handle_preprocess}')
```

    BERT model selected           : https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-4_H-512_A-8/1
    Preprocess model auto-selected: https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3
    
<br>
<br>

* Convert the preprocessor into a layer form and prepare it for later use.

<br>

```python
bert_preprocess_model = hub.KerasLayer(tfhub_handle_preprocess)
```

<br>
<br>   

* Let's do a simple test.

<br>

* Let's check if the output is correct by putting any text into BERT through preprocessing.

<br>

* The test text is ' **this is such an amazing movie!** '

<br>

```python
text_test = ['this is such an amazing movie!']
text_preprocessed = bert_preprocess_model(text_test)

print(f'Keys       : {list(text_preprocessed.keys())}')
print(f'Shape      : {text_preprocessed["input_word_ids"].shape}')
print(f'Word Ids   : {text_preprocessed["input_word_ids"][0, :12]}')
print(f'Input Mask : {text_preprocessed["input_mask"][0, :12]}')
print(f'Type Ids   : {text_preprocessed["input_type_ids"][0, :12]}')
```

    Keys       : ['input_type_ids', 'input_mask', 'input_word_ids']
    Shape      : (1, 128)
    Word Ids   : [ 101 2023 2003 2107 2019 6429 3185  999  102    0    0    0]
    Input Mask : [1 1 1 1 1 1 1 1 1 0 0 0]
    Type Ids   : [0 0 0 0 0 0 0 0 0 0 0 0]
    
<br>

* BERT model requires 3 values as input

<br>

* If you review at BERT paper, the terms 'Token embeddings' , 'Sentence Embeddings' and 'Transformer positional embeddings'** appear.

<br>

* 'Token embeddings' is the vocabulary embedding ID of each token (word).

<br>

* 'Sentence Embeddings' has the same value in one sentence. In the example above, since only one sentence was input, you can see that 1 is output and all others are 0. If there are multiple sentences, different numbers will appear.

<br>

* The values corresponding to each of the above items become **'input_word_ids' , 'input_mask' , and 'input_type_ids'** .

<br>
<br>

* It would be nice to have the same name, but why did they give it a different name...

<br>
<br>

* Values that have passed through the preprocessor can only be used as input values to BERT.

<br>

* Download the BERT Model and convert it to a layer.

<br>

```python
bert_model = hub.KerasLayer(tfhub_handle_encoder)
```

<br>
<br>

* Let's put the preprocessed text into BERT model and let's check the result.

<br>

```python
bert_results = bert_model(text_preprocessed)

print(f'Loaded BERT: {tfhub_handle_encoder}')
print(f'Pooled Outputs Shape:{bert_results["pooled_output"].shape}')
print(f'Pooled Outputs Values:{bert_results["pooled_output"][0, :12]}')
print(f'Sequence Outputs Shape:{bert_results["sequence_output"].shape}')
print(f'Sequence Outputs Values:{bert_results["sequence_output"][0, :12]}')
```

    Loaded BERT: https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-4_H-512_A-8/1
    Pooled Outputs Shape:(1, 512)
    Pooled Outputs Values:[ 0.76262873  0.99280983 -0.18611865  0.3667382   0.15233754  0.6550446
      0.9681154  -0.948627    0.00216129 -0.9877731   0.06842719 -0.9763059 ]
    Sequence Outputs Shape:(1, 128, 512)
    Sequence Outputs Values:[[-0.2894631   0.3432125   0.33231503 ...  0.21300879  0.71020824
      -0.05771083]
     [-0.28741956  0.31981033 -0.23018478 ...  0.58455044 -0.21329702
       0.72692096]
     [-0.66156983  0.68876874 -0.87432986 ...  0.10877332 -0.26173213
       0.4785546 ]
     ...
     [-0.22561109 -0.2892557  -0.07064363 ...  0.47566098  0.8327722
       0.40025353]
     [-0.29824188 -0.27473086 -0.05450502 ...  0.48849785  1.0955354
       0.18163365]
     [-0.4437817   0.00930776  0.07223777 ...  0.17290121  1.1833248
       0.07898013]]
    
<br>

* As looking at the output values by BERT, there are 'Pooled Outputs' and 'Sequence Outputs'.

<br>

* The value we will use is 'Pooled Outputs', and we will use this value to classify.

<br>
<br>
<br>
<br>
<br>

## 4. Build Model   

<br>

* Now that we have all the necessary preparations, let's create the classification model.

<br>   
<br>

### 4.1. Define Model

<br>
<br>

```python
def build_classifier_model():    
    text_input = tf.keras.layers.Input(shape=(), dtype=tf.string, name='text')    
    preprocessing_layer = hub.KerasLayer(tfhub_handle_preprocess, name='preprocessing')    
    encoder_inputs = preprocessing_layer(text_input)    
    encoder = hub.KerasLayer(tfhub_handle_encoder, trainable=True, name='BERT_encoder')    
    outputs = encoder(encoder_inputs)    
    net = outputs['pooled_output']
    net = tf.keras.layers.Dropout(0.1)(net)    
    net = tf.keras.layers.Dense(1, activation=None, name='classifier')(net)
    
    return tf.keras.Model(text_input, net)
```

<br>

* **text_input = tf.keras.layers.Input(shape=(), dtype=tf.string, name='text')**  
  - This is the input layer that receives the raw movie review text.
  
<br>
  
* **preprocessing_layer = hub.KerasLayer(tfhub_handle_preprocess, name='preprocessing')**
  - Download the preprocessor corresponding to BERT model we selected and make it in the form of a layer.

<br>

* **encoder_inputs = preprocessing_layer(text_input)**
  - Text that has passed through the preprocessor is converted as BERT inputs.

<br>

* **encoder = hub.KerasLayer(tfhub_handle_encoder, trainable=True, name='BERT_encoder')**
  - It's BERT layer

<br>

* **outputs = encoder(encoder_inputs)**
  - Output is the result of putting encoder_inputs into BERT layer.

<br>

* **net = outputs['pooled_output']**
  - From the outputs of BERT, the actual value we will use is the 'pooled_output'.
  
<br>

* **net = tf.keras.layers.Dropout(0.1)(net)**
* **net = tf.keras.layers.Dense(1, activation=None, name='classifier')(net)**
  - The output is extracted directly without any dense Layer.
  - Since it is a binary classification, there is only one output.

<br>
<br>

* Let's check if the created model outputs values properly.

<br>

* Let's put the test sentence as input. Of course, since this model is not trained yet, the output value has no meaning.

<br>

* Just to check if it works.

<br>

```python
classifier_model = build_classifier_model()
bert_raw_result = classifier_model(tf.constant(text_test))
print(tf.sigmoid(bert_raw_result))
```

    tf.Tensor([[0.66958743]], shape=(1, 1), dtype=float32)

<br>

* Only one value is output. That value may be a probability value, right?

<br>
<br>

### 4.2. Evaluation Metric

<br>

* Evaluation metric is selected according to binary classification.

<br>

```python
loss = tf.keras.losses.BinaryCrossentropy(from_logits=True)
metrics = tf.metrics.BinaryAccuracy()
```
<br>
<br>

### 4.3. Optimizer

<br>

* We'll use Adam as Optimizer.

<br>

```python
init_lr = 3e-5

step = tf.Variable(0, trainable=False)
schedule = tf.optimizers.schedules.PiecewiseConstantDecay(
        [1407*20, 1407*30], [1e-3, 1e-4, 1e-5])
wd = lambda: 1e-1 * schedule(step)

optimizer = tfa.optimizers.AdamW(learning_rate=init_lr, weight_decay=wd)
```


```python
epochs = 5
```


```python
classifier_model.compile(optimizer=optimizer,
                         loss=loss,
                         metrics=metrics)
```

<br>
<br>
<br>
<br>
<br>

## 5. Training Model      

<br>

* Now that we're all set, let's start the Train.

<br>

* I trained on a GTX 960, 1 epoch took about 11 minutes.

<br>

```python
print(f'Training model with {tfhub_handle_encoder}')

history = classifier_model.fit(x=train_ds,
                               validation_data=val_ds,
                               epochs=epochs)
```

    Training model with https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-4_H-512_A-8/1
    Epoch 1/5
    5000/5000 [==============================] - 675s 134ms/step - loss: 0.4439 - binary_accuracy: 0.7833 - val_loss: 0.4130 - val_binary_accuracy: 0.7922
    Epoch 2/5
    5000/5000 [==============================] - 665s 133ms/step - loss: 0.3583 - binary_accuracy: 0.8393 - val_loss: 0.3925 - val_binary_accuracy: 0.8202
    Epoch 3/5
    5000/5000 [==============================] - 669s 134ms/step - loss: 0.2691 - binary_accuracy: 0.8934 - val_loss: 0.4323 - val_binary_accuracy: 0.8196
    Epoch 4/5
    5000/5000 [==============================] - 672s 134ms/step - loss: 0.2085 - binary_accuracy: 0.9220 - val_loss: 0.5172 - val_binary_accuracy: 0.8052
    Epoch 5/5
    5000/5000 [==============================] - 749s 150ms/step - loss: 0.1722 - binary_accuracy: 0.9399 - val_loss: 0.5505 - val_binary_accuracy: 0.7980
    
<br>
<br>  

* Accuracy continues to rise in train data, but validation accuracy falls from the second epoch.

<br>

* It's overfitting.

<br>
<br>

```python
history_dict = history.history
print(history_dict.keys())

acc = history_dict['binary_accuracy']
val_acc = history_dict['val_binary_accuracy']
loss = history_dict['loss']
val_loss = history_dict['val_loss']

epochs = range(1, len(acc) + 1)
fig = plt.figure(figsize=(10, 6))
fig.tight_layout()

plt.subplot(2, 1, 1)
# "bo" is for "blue dot"
plt.plot(epochs, loss, 'r', label='Training loss')
# b is for "solid blue line"
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
# plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.subplot(2, 1, 2)
plt.plot(epochs, acc, 'r', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend(loc='lower right')
```

    dict_keys(['loss', 'binary_accuracy', 'val_loss', 'val_binary_accuracy'])

<br>    
<br>

<p align="left">
  <img src="/assets/BERT_Text_Classification/output_166_2.png">
</p>

<br>
<br>
<br>
<br>
<br>

## 6. Evaluation

<br>

* Let's evaluate the model with test data.

<br>

```python
loss, accuracy = classifier_model.evaluate(test_ds)

print(f'Loss: {loss}')
print(f'Accuracy: {accuracy}')
```

    6250/6250 [==============================] - 291s 46ms/step - loss: 0.6742 - binary_accuracy: 0.7563
    Loss: 0.6742448210716248
    Accuracy: 0.7562800049781799
    
<br>

* It will show the similar result as val. accuracy.

<br>
<br>

* This is how to save the model and reload it.

<br>

* You can call .save()

<br>

```python
dataset_name = 'imdb'
saved_model_path = './{}_bert'.format(dataset_name.replace('/', '_'))

classifier_model.save(saved_model_path, include_optimizer=False)
```
<br>
<br>

* When you want to load a saved model, you can use tf.saved_model.load().

<br>

```python
reloaded_model = tf.saved_model.load(saved_model_path)
```
<br>
<br>

* The results of the original model and the loaded model are the same, right?

<br>

```python
def print_my_examples(inputs, results):
    result_for_printing = \
    [f'input: {inputs[i]:<30} : score: {results[i][0]:.6f}'
                         for i in range(len(inputs))]
    print(*result_for_printing, sep='\n')
    print()


examples = [
    'this is such an amazing movie!',  # this is the same sentence tried earlier
    'The movie was great!',
    'The movie was meh.',
    'The movie was okish.',
    'The movie was terrible...'
]

reloaded_results = tf.sigmoid(reloaded_model(tf.constant(examples)))
original_results = tf.sigmoid(classifier_model(tf.constant(examples)))

print('Results from the saved model:')
print_my_examples(examples, reloaded_results)
print('Results from the model in memory:')
print_my_examples(examples, original_results)
```

    Results from the saved model:
    input: this is such an amazing movie! : score: 0.983353
    input: The movie was great!           : score: 0.983342
    input: The movie was meh.             : score: 0.824279
    input: The movie was okish.           : score: 0.137596
    input: The movie was terrible...      : score: 0.018345
    
    Results from the model in memory:
    input: this is such an amazing movie! : score: 0.983353
    input: The movie was great!           : score: 0.983342
    input: The movie was meh.             : score: 0.824279
    input: The movie was okish.           : score: 0.137596
    input: The movie was terrible...      : score: 0.018345

<br>
<br>

* In this post, we learned how to use BERT simply.

<br>

* Next, let's look at another example using BERT.

<br>

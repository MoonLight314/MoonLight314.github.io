---
title: "Image Captioning"
date: 2017-10-20 08:26:28 -0400
categories: project ImageCaption
---

개인적으로 대량의 사진에 Caption을 달아야 할 일이 생겨서 이것저것 알아보던 중에 찾은 예제입니다.   
원본은 https://towardsdatascience.com/image-captioning-with-keras-teaching-computers-to-describe-pictures-c88a46a311b8 이며
제가 개인적인 Comment와 설명을 위한 그림을 추가하였습니다.

이 예제에서는 Flickr 8k Image Data Set을 사용할 예정인데, Flickr 8k Data Set에는 총 8091개의 Image File이 있고 각 Image File당 5개의 Caption이 제공됩니다.   
'Flickr8k.token.txt'에 각 Image의 Caption이 저장되어 있습니다.
   
    
우선 필요한 Package를 Load합니다.   
주로 Keras를 사용할 예정이며, Image에서 Feature Extract를 위해 VGG16 Model을 사용할 예정입니다.   

```
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

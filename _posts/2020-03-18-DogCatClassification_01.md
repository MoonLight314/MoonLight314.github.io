---
title: "Dog & Cat Classification Exercise #01"
date: 2017-10-20 08:26:28 -0400
categories: study CNN Exercise
---
# CNN Exercise #01 - 개와 고양이 사진 분류하기

* Exercise #00에서는 Dropout 과 Data Augmentation으로 정확도를 올리는 방법을 사용해 보았습니다.
* 이번에는 다른 사람이 Train 시켜 놓은 훌륭한 Pre-Trained Model을 이용하여 정확도를 더 끌어올려 보겠습니다.

<br>
<br>
<br>
<br>

* 앞 Exercise와 마찬가지로 이번에도 Keras를 사용하도록 하겠습니다.   

```python
import keras
keras.__version__
```
    Using TensorFlow backend.
    '2.2.4'

```python
from IPython.core.interactiveshell import InteractiveShell 
InteractiveShell.ast_node_interactivity = "all"
```

<br>
<br>

# 0. Load Pre-Trained Model   

* 이번 Exercise에서 사용할 Pre-Trained Model은 VGG16입니다.
* VGG16은 구조가 간단하고 ImageNet 데이터셋에 널리 사용되는 Conv. Net입니다.
* VGG16은 조금 오래되었고 최고 수준의 성능에는 못미치며 최근의 다른 모델보다는 조금 무겁습니다.
* 하지만 이 모델의 구조가 이전에 보았던 것과 비슷해서 새로운 개념을 도입하지 않고 이해하기 쉽기 때문에 선택하였습니다.  

  
* CNN의 전반적인 설명과 다른 Pre-Trained Model에 대해서 알아보시려면 아래 Link의 Post를 참고하시기 바랍니다.
  - https://moonlight314.github.io/study/cnn/CNN/
  
<br>
<br>
<br>
<br>

* VGG16 Model은 아래와 같이 구성되어 있습니다.   

![title](CNN_Exer_01_Assets/pic_01.png)

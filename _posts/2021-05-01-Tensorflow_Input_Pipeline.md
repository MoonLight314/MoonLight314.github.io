---
title: "Tensorflow Input Pipeline"
date: 2021-05-01 08:26:28 -0400
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

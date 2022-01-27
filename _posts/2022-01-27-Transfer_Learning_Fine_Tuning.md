---
title: "The Strategy of Transfer Learning & Fine Tunung"
date: 2021-10-29 08:26:28 -0400
categories: Deep Learning
---
# The Strategy of Transfer Learning & Fine Tunung

<br>
<br>
<br>
<br>

## 0. Transfer Learning

<br>

* 다른 Dataset으로 이미 학습된(Pre-Trained) Model을 가져와서 내가 하고자 하는 작업에 적용하는 것을 말합니다.

<br>
<br>

## 1. Fine Tuning

<br>

* Pre-Trained Model은 다른 Dataset에서 학습된 Weight & Bias를 가지고 있기 때문에 새롭게 적용하려는 작업에 잘 맞지 않을 수가 있다.

<br>

* Pre-Trained Model을 새로운 작업에 맞게 Weight & Classifier를 새롭게 조정하는 작업을 Fine Tuning이라고 한다.

<br>

* Pre-Trained Model 전체를 다시 Tuning할 지 혹은 일부만 Tuning할 지는 여러가지 상황을 고려하여 선택한다.   

<br>
<br>
<br>
<br>

<p align="center">
  <img src="/assets/Transfer_Learning/pic_00.png">
</p>

<br>
<br>

## 3. Dataset의 특성과 양에 따른 Fine-Tuning 전략   

<br>
<br>

<p align="center">
  <img src="/assets/Transfer_Learning/pic_01.png">
</p>

<br>
<br>

### 3.1. Quadrant 1

<br>

* Large Dataset이 있지만, Dataset의 특징이 Pre-Trained Model이 학습한 Dataset의 특성과 다른 경우.

<br>

* 이런 경우에는 Strategy 1을 적용하면 된다.

<br>

* 비록 Dataset의 특성이 다르지만, 학습할 수 있는 Dataset이 많이 있으므로, Pre-Trained Model의 구조만 차용하고 Parameter는 새롭게 학습하면 된다.

<br>
<br>
   

### 3.2. Quadrant 2

<br>

* Large Dataset & Dataset의 특징이 Pre-Trained Model이 학습한 Dataset의 특성과 유사한 경우.

<br>

* 이 경우에는 어떤 방법을 사용해도 별 상관없지만, Strategy 2를 사용하면 된다.

<br>
<br>

### 3.3. Quadrant 3

<br>

* Dataset도 부족하고 게다가 Dataset의 특징이 Pre-Trained Model이 학습한 Dataset의 특성과도 다른 최악의 경우이다.

<br>

* Strategy 2를 사용하여, 적절한 수의 Freeze Layer를 찾아야 한다.

<br>

* 또한, Dataset의 수가 부족하므로, Data Augmentation을 사용해야 한다.

<br>
<br>

### 3.4. Quadrant 4

<br>

* Dataset의 수는 적지만, Dataset의 특징이 Pre-Trained Model이 학습한 Dataset의 특성과 유사한 경우.

<br>

* 이런 경우에는 Strategy 3이 가장 적당하다.

<br>

* 기존 Conv. Layer를 Feature Extractor로 사용하고, Classifier만 학습하면 된다.

<br>
<br>

<p align="center">
  <img src="/assets/Transfer_Learning/pic_02.png">
</p>

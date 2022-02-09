---
title: "비전 시스템을 위한 딥러닝(Deep Learning For Vision System)"
date: 2021-10-29 08:26:28 -0400
categories: Deep Learning
---
# 비전 시스템을 위한 딥러닝(Deep Learning For Vision System)

### 한빛미디어 <나는 리뷰어다> 활동을 위해서 책을 제공받아 작성된 서평입니다.

<br>
<br>

Deep Learning이 대중들에게 이름을 알리기 시작한 것은 무엇보다도 Image Data를 처리하는 데에 있어서 인간에 버금가는 능력을 보여주면서부터라고 생각합니다.

그 이후로 Deep Learning / AI가 우리가 알게 모르게 일상생활에 깊숙이 파고들고 있는 것이 사실입니다.

하지만, Python을 익히고 Deep Learning에 자주 사용되는 Package들의 사용법을 익힌 후에 실제로 간단한 Image Classification 작업을 해보는 것은
생각만큼 쉽지 않습니다.

다양한 예제들이 많지만 대부분 수학적인 원리나 해당 Code가 어떤 일을 하는지 그리고 왜 필요한지에 대해서는 자세히 설명되어 있지 않거나
알 수 없는 수학 기호들만 나열되어 있을 뿐 무슨 말인지 이해하기 힘든 것이 입문자들의 현실입니다.

Deep Learning, 특히 Image 관련 쪽에 괜찮은 입문서가 하나 나와서 소개해 드리려고 합니다.

<br>

<p align="center">
  <img src="/assets/Book_Review_Deep_Learning_For_Vision_System/pic_00.png">
</p>

<br>

표지는 푸른색 옷을 입고 패션감각을 뽐내는 이븐 알하이삼입니다.

이분은 현대 광학의 큰 공헌을 한 '광학의 아버지'라고 불리고 계신다고 합니다.

이 책을 접하는 여러분들도 이븐 알하이삼처럼 Image 분야에 한 획을 그으시기를 바랍니다.

<br>
<br>

## 1. 전체 소개

<br>

### 1.1. 목차

이 책은 총 3개의 Part, 10개의 Chapter로 구성되어 있습니다.

Part 1은 Deep Learning 기초에 대해서 다루고 있으며, Part 2는 실제 Image를 다루는 다양한 Deep Learning Model의 이론적 배경와 함께 실제 사용법에 대해서 다루고 있습니다.

마지막 Part 3는 Image 분야의 최신 트렌드인 GAN(Generative Adversarial Network)과 Image Embedding에 대해서 소개하고 있습니다.

<br>

### 1.2. Part 1 - Deep Learning 기초

이 부분에서는 Deep Learning의 기본적인 부분과 Deep Learning을 어떤 방식으로 Image 처리에 응용할 수 있는가를 소개하는 것으로 시작됩니다.

Deep Learning의 Input은 어떤 Data이던지 반드시 숫자로 표현이 되어야 하기 때문에, Image를 어떻게 전처리(Pre Processing)를 거쳐서 Deep Learning Model에 적용하는지에 대해서도 다룹니다.

또한, Deep Learning이 어떻게 스스로 학습을 하는지에 대한 중요한 개념들(Perceptron , Activation Function , Loss Function , Backpropagation 등)에 대한 설명을 쉽게 되어 있습니다.

마지막으로, Image 뿐만 아니라 다양한 분야에도 응용되는 CNN(Convolution Neural Network)에 대한 설명도 하고 있습니다.

<br>

### 1.3. Part 2 - 이미지 분류와 탐지

기본 개념에 대한 워밍업 후에 본격적으로 AlexNet , VGGNet , Inception , ResNet 등과 같은 다양한 Image 처리 관련 Deep Learning Model들에 대한 설명을 합니다.

이 Model들을 이용한(그리고 가장 흔히 사용되는) Tranfer Learning에 대해서도 매우 상세하게 다룹니다.

뿐만 아니라, R-CNN, SSD, YOLO와 같은 Object Detection / Segmentation 기법들도 소개하고 있습니다.

<br>

### 1.4. Part 3 - 생성 모델과 시각 임베딩

마지막 Part에서는 현재 Deep Learning을 이용한 Image 처리 분야의 최신 트렌드인 GAN(Generative Adversarial Network)을 소개하고 있습니다.

사실, GAN은 Dataset의 확률분포를 학습하여 이와 유사한 Data를 생성한다는 기본 속성을 Image 뿐만 아니라 다양한 분야에도 응용할 수 있습니다.

응용 가능성이 높은 GAN을 Image에 분야에 사용하면서 감을 익힐 수 있는 좋은 기회라고 생각합니다.

<br>
<br>

## 2. 장점

<br>

1) 복잡한 수식이 나오지만 매우 쉽게 개념을 설명하고 있습니다.
   예를 들어, Activation Function 이나 Backpropagation과 같은 개념을 설명하기 위해서는 복잡한 수식없이 설명하기 힘들지만, 이 책에서는 이해하기 쉽게 설명하고 있습니다.

<br>

2) 코드 한줄 한줄 그 의미를 자세하게 설명해줍니다.
   각 예제들에서 해당 코드가 전체 Model 구성에 있어서 어떤 역할을 하는지 또한, 특정 API 사용할 때는 API들의 기능과 함께 각 Parameter의 상세한 기능을 잘 설명해 줍니다.

<br>

3) 그림이 전체적으로 심플하면서도 해당 개념의 핵심을 잘 표현하고 있습니다.
   이해를 돕기 위해 삽입된 그림이 Simple하면서도 핵심을 잘 나타내준다는 것이 큰 장점입니다.

<br>

4) 어떤 개념을 설명하는 경우, 그 개념이 필요한 이유와 구현 방식을 잘 설명하고 있다

<br>

5) 이 책의 예제 Code들은 Tensorflow 2.1로 작성되어 있으며, Sample Code가 군더더기 없이 깔끔합니다.

<br>

6) 딥러닝의 기초부터 최신 트렌드까지 소개하고 있기때문에 자칫 수박 겉핡기 식이 아닐까 생각할 수 있지만, 중요한 개념에 대해서는 매우 상세하게 설명하고 있습니다.

<br>
<br>

## 3. 대상 독자

<br>

이 책은 어느 정도 기초 지식을 필요로 합니다. 

Python은 어느 정도 능숙하게 다룰 줄 안다는 가정하에, Numpy , OpenCV, Tensorflow, Keras 등과 같은 Framework & Package들을 사용하고 있습니다.

위에 언급한 Framework & Package에 대한 지식이 어느 정도 있어야 읽기가 수월할 것 같습니다.

또한, Deep Learning Framework으로 Tensorflow를 사용하고 있기 때문에, PyTorch나 그 외 Framework을 사용하고 있다면 Sample Code들이 생소하게 느껴질 수 있습니다.

<br>
<br>

## 4. 마치며

<br>

이 책은 Deep Learning을 이용해서 Image 관련 업무를 시작하려는 분들에게 아주 훌륭한 길라잡이가 될 것이라고 생각합니다.

쉬우면서도 핵심을 요약한 설명, 풍부한 예시, 간결한 Example Code 등은 이 책의 가장 큰 장점이라고 할 수 있습니다.

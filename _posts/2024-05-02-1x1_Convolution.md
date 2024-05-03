---
title: "1x1 Convolution"
date: 2024-05-02 08:26:28 -0400
categories: Deep Learning
---

<br>
<br>

이번 Post에서는 1x1 Convolution 연산에 대해서 알아보도록 하겠습니다.

<br>
<br>

# 0. Introduction

<br>

저는 처음 1x1 Convolution이라는 이름을 봤을 때, '1x1이면 1개의 Pixel인데, 크기 1짜리 Kernel 연산이 의미가 있을까'라는 의문이 들더군요

1x1 Convolution 연산은 일반적인 Convolution 연산과 몇 가지 중요한 차이점이 있습니다.

가장 중요한 차이는 일반적인 Convolution 연산이 공간적 정보를 처리하는 데 중점을 두는 반면, 1x1 Convolution은 Channel 간의 정보를 조합하고 차원을 조절하는 데 주로 사용됩니다. 

말로 설명하니 약간 이해가 되지 않는데요, 추가 설명과 연산 방법에 살펴보면 이해가 될 것 같습니다.

<br>

## 0.0. 차이점

<br>
<br>

### 0.0.0. Kernel 크기

<br>

일반적인 Convolution에서는 보통 3x3, 5x5, 7x7 등의 Kernel 크기를 사용하여 Kernel 내의 여러 픽셀 간의 공간적 관계를 학습하고 이 값들을 바탕으로 새로운 Feature를 생성하는 것을 반복하는 과정을 거치게 됩니다.

반면 1x1 Convolution에서는 Kernel 크기가 1x1이며 단 하나의 Pixel에서만 연산을 수행하며, 공간적인 정보는 전혀 고려하지 않고 오직 Channel 차원에서의 정보만을 조작한다는 의미입니다.


### 0.0.1. 목적

<br>

일반적인 Convolution의 목적은 Image의 텍스처, 가장자리, 형태 등 공간적인 패턴을 인식하고 이를 기반으로 특징을 추출하는 데 가장 큰 목적이 있습니다.

반면에, 1x1 Convolution은 주로 Neural Network의 깊이를 변경하는 데 사용되며 이는 Channel 수를 늘리거나 줄이는 역할, 각 Channel의 정보를 조합하여 새로운 Feature을 생성하는데도 사용됩니다.

<br>

### 0.0.2. 연산 효율성

<br>

일반적인 Convolution은 더 넓은 영역의 정보를 계산에 포함하기 때문에 비교적 많은 계산량이 요구되지만, 1x1 Convolution은 계산량이 상대적으로 적으며, 매우 효율적인 차원 변환 도구로 사용됩니다.

Neural Network의 복잡성을 조절하거나, Convolution Layer 사이에서 병목 현상을 줄이는 데 유용합니다.

<br>
<br>

1x1 Convolution은 이러한 특성 덕분에 매우 다양하게 활용되며, 특히 복잡한 아키텍처에서 중간 차원의 축소나 증가, Channel 정보의 재조합 등에 사용되어 Neural Network의 성능을 최적화하는 데 큰 역할을 합니다.

<br>
<br>

# 1. 연산 방법

<br>

실제 1x1 Convolution 연산을 하는 방법을 소개해 드리도록 하겠습니다.

Input Feature Map의 Size가 56x56x512라고 하고, 우리는 이 Feature Map의 Channel을 2배인, 1024개로 늘리고자 합니다.

<br>
<br>

<p align="center">
  <img src="/assets/1x1_Convolution/00.png">
</p>

<br>
<br>



### Step 01

<br>

먼저 Input Feature Map의 Channel 수(=512)와 동일한 수의 Weight를 최종 목표 Channel 수와 동일한 1024개 준비합니다.

<br>
<br>

<p align="center">
  <img src="/assets/1x1_Convolution/01.png">
</p>

<br>
<br>



### Step 02

<br>

Input Feature Map에서 하나의 Pixel의 위치를 선택합니다. 아래의 그림에서는 (0,0) 위치의 Pixel을 선택했다고 가정합니다.

<br>
<br>

<p align="center">
  <img src="/assets/1x1_Convolution/02.png">
</p>

<br>
<br>


Input Feature Map의 Channel 수가 512개이기 때문에, (0,0) 위치의 Pixel은 모두 512개가 있습니다.

이 (0,0) 위치의 Pixel 512개를 첫 번째 Kernel Filter와 Element-wise 곱셈을 수행합니다.

곱셈을 수행하면 512개의 결과가 나올 것이고, 이 값들 모두 더하면 하나의 결괏값이 나옵니다.

이 과정을 1024개 Kernel Filter에 반복하면 Pixel 하나에 1024개의 값이 생깁니다.

<br>

### Step 03

<br>

이번에는 Input Feature Map의 (0,1) 위치의 Pixel을 선택하고, Step 02와 같이 Kernel Filter 1024개와 각각 모두 곱하고 더해서 하나의 값을 계산한 후 Concatenate 합니다.

<br>
<br>

<p align="center">
  <img src="/assets/1x1_Convolution/03.png">
</p>

<br>
<br>




### Step 04

<br>

이와 같은 연산을 Pixel 수만큼 반복합니다.

그러면 총 56x56개의 1024개 값이 나옵니다.

결과적으로 최초 Input Feature Map의 Size인 56x56x512에서 56x56x1024의 Feature Map이 생겨나면서

Channel 수가 2배 증가하는 효과를 가져옵니다.


<br>
<br>

<p align="center">
  <img src="/assets/1x1_Convolution/04.png">
</p>

<br>
<br>




이번 Post에서는 1x1 Convolution의 설명과 실제 연산하는 방법에 대해서 살펴보았습니다.

앞으로 다룰 내용의 중요한 일부분이어서 미리 내용 정리해 보았습니다.

도움이 되셨기를 바라며, 다음에 또 뵙겠습니다.


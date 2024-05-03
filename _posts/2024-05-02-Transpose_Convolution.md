---
title: "Transpose Convolution"
date: 2024-05-02 08:26:28 -0400
categories: Deep Learning
---

<br>
<br>

이번 Post에서는 Transpose Convolution에 대해서 알아보도록 하겠습니다.

<br>
<br>

# 0. Introduction

<br>

Transpose Convolution은 우리가 흔히 알고 있는 CNN Model에서 주로 사용되는 Convolution과 반대되는 연산을 수행합니다.

Convolution 연산은 특정 크기의 Kernel이라는 Filter를 이용해서 Image를 Scan하면서 Feature를 뽑아내는 동작을 하면서

점점 작아지는 Feature Map을 생성하는 연산입니다.

반면, Transpose Convolution은 Feature Map의 차원을 증가(Up-Sampling) 시키면서 원래 값을 복원하는 용도로 주로 사용됩니다.(Upscaler)

이러한 Transpose Convolution의 속성 때문에 다음과 같은 용도로 많이 사용됩니다.

<br>
<br>

### 1. Image Up-Sampling

Image의 해상도를 높이기 위해 사용됩니다. Transpose Convolution의 연산자체가 Feature Map의 각 Element에 Kernel Filter를 적용해 더 높은 차원으로 변환하기 때문에 가능합니다.

예를 들어, 저해상도의 이미지를 고해상도로 변환할 때 사용할 수 있습니다. 

이는 디지털 줌이나 고품질 이미지 복원과 같은 작업에 적용됩니다.

<br>
<br>

### 2. Feature Map Up-sampling

앞서 말한 Image Up-Sampling과 같은 맥락입니다.

컴퓨터 비전에서 객체 검출이나 세그먼테이션 작업(Ex. U-Net)에서 고해상도의 특징 맵을 생성하기 위해 사용됩니다.

예를 들어, Deep Neural Network에서 저차원의 정보를 다시 고차원으로 확장하여 원본 이미지의 크기와 같은 출력을 생성할 때 사용됩니다.

<br>
<br>

### 3. Deep Learning에서의 Image Generation

생성적 적대 신경망(GANs)이나 Auto-Encoder 같은 아키텍처에서 이미지를 생성할 때 사용됩니다. 

이러한 모델은 낮은 차원의 잠재 공간에서 시작하여 점차적으로 이미지의 크기를 확장해 나가며, 이 과정에서 transpose convolution이 핵심적인 역할을 합니다.

<br>
<br>

# 1. 계산 방법

그럼 실제로 어떻게 Transpose Convolution 연산을 하는지 몇 가지 예제를 통해 알아보도록 하겠습니다.

<br>

## 1.0. 기본 연산 방법

Transpose Convolution 연산도 Convolution 연산과 마찬가지로 Kernel , Stride , Padding의 개념이 있습니다.

Convolution과 마찬가지로 Input Image의 각 원소에 Kernel의 각 원소와 곱 연산을 한 결과를 Stride와 Padding을 적용해서 결과를 계산하는 데 사용합니다.

아래 Example을 살펴보도록 하겠습니다.

<br>
<br>

## 1.1. Example 1

첫 번째 Example은 Input Image Size가 2x2이고, Kernel Size가 2x2, Stride가 1, Padding이 0인 경우입니다.

<br>
<br>

<p align="center">
  <img src="/assets/Transpose_Convolution/0.png">
</p>

<br>
<br>

먼저 Input Image의 각 원소에 Kernel을 곱해줍니다. 그럼 전체 Input Image의 원소 개수만큼의 결과가 나올 것입니다.

이 값들을 Stride와 Padding 값에 따라서 적절히 배열합니다.

이 예제에서는 Stride가 1이고, 결과 행렬의 Size가 2x2이기 때문에 겹치는 부분이 생깁니다.

겹치는 부분은 겹치는 값들을 모두 더해줍니다.

그럼 최종적으로 Output을 얻을 수 있습니다.


<br>
<br>

## 1.2. Example 2

두 번째 Example은 Input Image Size가 2x2이고, Kernel Size가 3x3, Stride가 2, Padding이 1인 경우입니다.

<br>
<br>

<p align="center">
  <img src="/assets/Transpose_Convolution/1.png">
</p>

<br>
<br>

원리는 첫 번째 Example과 동일합니다. Input Image의 각 원소를 Kernel에 곱해줍니다.

이번에는 Kernel Size가 3x3이기 때문에 결과도 3x3 크기의 행렬이 나오게 됩니다.

Padding이 1이기 때문에 아래 그림과 같이 흰색 부분에 해당하는 Padding 영역이 생깁니다.

Stride 만큼 이동한 후 Padding 영역을 고려해서 배치하면 아래와 같은 결과가 나오게 됩니다.

각 원소 위치에 해당하는 값들을 모두 더하면 최종 Output을 얻을 수 있습니다.

<br>
<br>

이상으로 Transpose Convolution의 개념과 용도, 구체적인 계산 방법까지 알아보았습니다.

제가 앞으로 진행할 주제를 미리 준비하기 위한 준비 단계라고 생각해 주시면 되겠습니다.

도움이 되셨으면 좋겠네요.

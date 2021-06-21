---
title: "DCGAN(Deep Convolutional Generative Adversarial Networks)"
date: 2021-06-21 08:26:28 -0400
categories: Deep Learning
---

### DCGAN(Deep Convolutional Generative Adversarial Networks)

<br>
<br>
<br>
<br>
<br>
<br>

## 0. Introduction

* Goodfellow, Ian이 2014년에 GAN을 소개한 이후에 아래 그림과 같이, 매우 다양한 GAN 응용이 나왔습니다.

  ( GAN의 다양한 종류. 출처 : https://ysbsb.github.io/gan/2020/06/17/GAN-newbie-guide.html )



* 이번 Post에서는 그 중에서도 모든 GAN응용의 시작이라고 할 수 있는 **DCGAN(Deep Convolutional Generative Adversarial Networks)**을 살펴보도록 하겠습니다.
<br>
<p align="center">
  <img src="/assets/DCGAN_Paper_Review/pic_00.png">
</p>
<br>
<br>
<br>

* DCGAN은 2016년에 Alec Radford & Luke Metz , Soumith Chintala에 의해서 발표됩니다.
  
<br>

* Paper 정식 제목은 'UNSUPERVISED REPRESENTATION LEARNING WITH DEEP CONVOLUTIONAL GENERATIVE ADVERSARIAL NETWORKS'이며, Paper Link는 아래를 참고하시기 바랍니다.

   [UNSUPERVISED REPRESENTATION LEARNING WITH DEEP CONVOLUTIONAL GENERATIVE ADVERSARIAL NETWORKS](https://arxiv.org/pdf/1511.06434v2.pdf)
   
<br>
<br>
<br>
<br>
<br>
<br>

## 1. Issues of GAN

<br>

* 이 Paper에서는 기존 GAN의 몇 가지 한계점을 언급하고 있습니다.

<br>

  - **GANs have been known to be unstable to train, often resulting in generators that produce nonsensical outputs.**
    * GAN의 구조 자체가 불안정해서 Train시키기가 쉽지 않습니다.

<br>

  - **One constant criticism of using neural networks has been that they are black-box methods, with little understanding of what the networks do in the form of a simple human-consumable algorithm.**
    * Neural Net 구조가 가지는 고질적인 한계인데, 학습된 Model의 해석이 어렵습니다.
    * Model이 어떤 근거로 판단을 했는지 알수 없기 때문에 흔히, Model을 'Black-Box'같다라고 표현합니다.

<br>

  - **GAN Model 성능을 정량적으로 평가하기가 어렵다.**
    * 기존 Sample로부터 얼마나 잘 생성되었는지 정량적으로 판단하기가 애매하고, 사람이 평가하기도 힘듭니다.

<br>
<br>
<br>
<br>
<br>
<br>

## 2. Purpose of DCGAN

<br>

* **Generator가 단순히 Sample Data를 Memorize해서 보여주는 것이 아니라는 것을 증명**
  - Model의 구조가 Sample Data 대비 를 충분히 크고, Train을  충분히 많이 진행한다면 Model이 Sample Data를 모두 외워버릴 수도 있습니다.
  - 이것은 Generating이라고 보기 힘들다.
  - 얼핏 Overfitting과 일맥상통하는 것 같기도 하고...

<br>

* **WALKING IN THE LATENT SPACE**
  - Latent Space의 z의 작은 변화에도 Generation 결과가 자연스럽게 이루어져야 한다. ( 이를 Paper에서는 WALKING IN THE LATENT SPACE 라고 표현했습니다. )

<br>
<br>
<br>
<br>
<br>

## 3. Architecture Guidelines

<br>

### 3.1. Overall

<br>

* GAN에서 제시한 구조에서 DCGAN으로 넘어오면서 전체적인 구조는 그대로 유지되지만 몇몇 세부적인 사항들이 바뀌었습니다.

<br>

* 저자들이 이런 구조를 확립한 방법을 Paper에서 아래와 같이 말하고 있습니다.
    - after extensive model exploration we identified a family of architectures that resulted in stable training across a range of datasets and allowed for training higher resolution and deeper generative models.
      * Paper에서는 있어 보이게 거창하게 말하고 있지만, 한 마디로 생노가다로 최상의 Model 구조를 알아냈다는 의미입니다.
      * 근성만은 인정해줘야 할 것 같네요.

<br>
<br>
<br>

### 3.2. Modification Points

<br>

* DCGAN에서 제안한 기존 GAN의 성능을 향상 시킬 수 있는 방법들은 다음과 같습니다.

<br>

  **1. G / D에서 Pooling Layer대신 D에는 strided convolutions, G에는 fractional-strided convolutions 사용**

<br>

  **2. Batch Normalization 사용**

<br>

  **3. FC(Fully Connected) Hidden Layer를 모두 삭제**

<br>

  **4. G에서 마지막 Layer를 제외하고 Activation Function을 RELU사용. 마지막 Layer는 tanh 사용**

<br>

  **5. D에서는 모든 Layer의 Activation Function을 LeakyRELU 사용**
  
<br>

* 수정 내용들에 대해서 하나씩 살펴보도록 하겠습니다.  

<br>
<br>

#### 3.2.1. Add Convolution Layer

<br>

* 우선, Pooling Layer대신 사용한 **strided convolutions**과 **fractional-strided convolutions**에 대해서 알아보겠습니다.

<br>

* strided convolutions과 fractional-strided convolutions은 Convolution 방식 중의 하나지만, 기존 CNN에 적용되었던 Convolution은 Kernel을 거치면서 Size가 점점 줄어드는데 비해서 strided convolutions과 fractional-strided convolutions은 특정 연산을 거치면서 Size가 증가하는 차이가 있습니다.

<br>

* 아래는 strided convolutions을 설명한 것이며, 한 마디로 strided convolutions은 stride 만큼 이동하는 Convolution입니다.

<br>

<p align="center">
  <img src="/assets/DCGAN_Paper_Review/pic_01.gif">
</p>

<br>
<br>
<br>

* fractional-strided convolutions은 Transposed Convolution이라고 하는데(Deconvolution이라고 하는 곳도 있는데 이는 정확한 개념이 아닙니다. Transposed Convolution과 Deconvolution의 차이는 여기를 참조. ), DCGAN의 Generator에서 사용하는 방식은 정확하게 말하면, **Transposed Convolution**입니다.

<br>

<p align="center">
  <img src="/assets/DCGAN_Paper_Review/pic_02.gif">
</p>

<br>

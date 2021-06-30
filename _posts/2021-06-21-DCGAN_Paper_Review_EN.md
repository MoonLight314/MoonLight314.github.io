---
title: "DCGAN(Deep Convolutional Generative Adversarial Networks) (EN)"
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

* After Goodfellow, Ian introduced GAN in 2014, a wide variety of GAN applications have emerged, as shown in the figure below.

  ( Various Type of GAN. https://ysbsb.github.io/gan/2020/06/17/GAN-newbie-guide.html )



* In this post, we will take a look at **DCGAN (Deep Convolutional Generative Adversarial Networks)**, which can be the beginning of all GAN applications.
<br>
<p align="center">
  <img src="/assets/DCGAN_Paper_Review/pic_00.png">
</p>
<br>
<br>
<br>

* DCGAN was released in 2016 by Alec Radford & Luke Metz , Soumith Chintala.

<br>

* The paper title is 'UNSUPERVISED REPRESENTATION LEARNING WITH DEEP CONVOLUTIONAL GENERATIVE ADVERSARIAL NETWORKS', and please refer to the paper link below.
   [UNSUPERVISED REPRESENTATION LEARNING WITH DEEP CONVOLUTIONAL GENERATIVE ADVERSARIAL NETWORKS](https://arxiv.org/pdf/1511.06434v2.pdf)
   
<br>
<br>
<br>
<br>
<br>
<br>

## 1. Issues of GAN

<br>

* This paper pointed out some limitations of existing GAN.
<br>

  - **GANs have been known to be unstable to train, often resulting in generators that produce nonsensical outputs.**
    * It is not easy to train because the structure of GAN itself is unstable.

<br>

  - **One constant criticism of using neural networks has been that they are black-box methods, with little understanding of what the networks do in the form of a simple human-consumable algorithm.**
    * This is a chronic limitation of the Neural Net structure. It is difficult to interpret the trained model.
    * It is often expressed as a 'Black-Box' because it is not known on what makes the model the judgment.

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
  - Model의 구조가 Sample Data 대비해서 충분히 크고, Train을  충분히 많이 진행한다면 Model이 Sample Data를 모두 외워버릴 수도 있습니다.
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
<br>
<br>

* Transposed Convolution에 대한 자세한 내용은 아래 Link의 글을 한 번 읽어보시기 바랍니다.
  
  [CS231n의 Transposed Convolution은 Deconvolution에 가까운 Transposed Convolution이다](https://realblack0.github.io/2020/05/11/transpose-convolution.html#Transposed-Convolution)

<br>

* Transposed Convolution은 Tensorflow에서 **Conv2DTranspose()**라는 함수로 구현되어 있습니다.

<br>

* 다양한 Convolution 방식들에 대한 설명은 아래 Link를 참조하시면 많은 정보를 얻을 수 있습니다.   

  [An Introduction to different Types of Convolutions in Deep Learning](https://towardsdatascience.com/types-of-convolutions-in-deep-learning-717013397f4d)
  
<br>
<br>
<br>

#### 3.2.2. Apply Batch Normalization

<br>

* Batch Normalization은 2015년에 발표된 Paper, [Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift](https://arxiv.org/abs/1502.03167)에서 소개된 개념이며, 성능이 좋아서 요즘 대부분의 Neural Network에 적용되고 있습니다.

<br>

* Paper Link( [http://proceedings.mlr.press/v37/ioffe15.pdf](http://proceedings.mlr.press/v37/ioffe15.pdf) )

<br>

* Batch Normalization의 목적은 Gradient Vanishing / Gradient Exploding을 방지하기 위함입니다.

<br>

* Batch Normalization가 나오기 이전에도 Activation Function을 ReLU를 사용한다던지, Weight Initialization을 할 때, He or Xavier initialization을 사용하곤 했습니다.

<br>

* Batch Normalization은 이런 간접적인 방식과는 다르게 Training 과정에 직접적으로 관여하면서 Gradient Vanishing / Gradient Exploding을 억제합니다.

<br>

* 자세한 설명은 아래 Link를 참조해 주시기 바랍니다. 

  [A Gentle Introduction to Batch Normalization for Deep Neural Networks](https://machinelearningmastery.com/batch-normalization-for-training-of-deep-neural-networks/)
  
<br>
<br>
<br>

#### 3.2.3. Others

<br>

* 나머지 변경 내용들에 대해서 추가적인 설명은 생략하겠습니다.

<br>

* 내용을 보시면 특별하게 설명이 필요한 사항은 없을 것 같습니다.

<br>

* 다시 한 번 말씀드리지만, 왜 Fully Connected를 없애고, strided convolutions과 fractional-strided convolutions를 사용했으며, 특정 위치에만 Batch Normalization을 사용하고 Activation Function을 위치에 따라 다르게 사용했는지에 대한 이론적 배경은 전혀 없고, 단순히 **노가다를 통해 결과를 관찰**하여서 알아낸 결과들입니다.   

<br>
<br>
<br>

#### 3.2.4. Overall Architecture

<br>

* 위에서 언급한 내용을 적용한 Generator의 전체적인 Architecture는 다음과 같습니다.   

<br>

<p align="center">
  <img src="/assets/DCGAN_Paper_Review/pic_03.png">
</p>
   
<br>

* 최초에 z에서 64x64의 Image가 되어가는 과정에서 fractional-strided convolutions가 적용되었으며, Fully Connected Layer나 Pooling Layer가 사용되지 않았습니다.   

<br>
<br>
<br>
<br>
<br>
<br>

## 4. Result

<br>

### 4.1. Image Generation

<br>

* 아래의 Bedroom Image들은 모두 1 Epoch Training 후에 DCGAN으로 생성된 Image라고 합니다.   

<br>

<p align="center">
  <img src="/assets/DCGAN_Paper_Review/pic_04.png">
</p>

<br>
<br>
<br>

* 아래의 Bedroom Image들은 5 Epoch Training 후에 DCGAN으로 생성된 Image라고 합니다.   

<br>

<p align="center">
  <img src="/assets/DCGAN_Paper_Review/pic_05.png">
</p>
   
<br>

* 딱 봐도 Quality가 훌륭합니다.

<br>

* Paper에서 말하기를 이론적으로는 Generator가 Sample Data를 Memorize 할 수 있지만, 낮은 Learning Rate와 Mini Batch를 적용함으로써 실험적으로 그것이 불가능하다고 말하고 있습니다.   

<br>
<br>
<br>

### 4.2. WALKING IN THE LATENT SPACE

<br>

* Paper에서 DCGAN의 목표중의 하나가 z의 작은 변화에도 Smooth하게(Walking) 결과가 변화하도록 하는 것이다라고 했습니다.   

<br>

<p align="center">
  <img src="/assets/DCGAN_Paper_Review/pic_06.png">
</p>
   
<br>

* 위의 사진은 왼쪽의 원본 Image에서 오른쪽의 Generated Image로 점점 변화하는 것을 나타낸 것입니다.

<br>

* 벽이 있던 곳이 어느새 창문이 생기고, 전등이 있던 곳이 창문이 생기기도 합니다.   

<br>
<br>
<br>

### 4.3. Black Box 극복

<br>

* DCGAN의 Discriminator의 Feature를 시각화해서 Model이 어떻게 동작하는지를 좀 더 명확하게 알 수 있게 되었습니다.

<br>

<p align="center">
  <img src="/assets/DCGAN_Paper_Review/pic_07.png">
</p>

<br>
<br>
<br>

### 4.4. VECTOR ARITHMETIC ON IMAGE

<br>

* Paper에서는 DCGAN을 하면서 NLP에서 사용된 Word2Vec의 특성을 Image에서도 사용할 수 있었다고 합니다.

<br>

* 예를 들어, Word2Vec에서 vector(”King”) - vector(”Man”) + vector(”Woman”) = Queen 이 되는 것처럼, Image에서도 이와 유사한 VECTOR ARITHMETIC 연산을 할 수 있었다고 하네요.   

<br>

<p align="center">
  <img src="/assets/DCGAN_Paper_Review/pic_08.png">
</p>

<br>

<p align="center">
  <img src="/assets/DCGAN_Paper_Review/pic_09.png">
</p>

<br>
<br>

* 이번 Post에서는 DCGAN Paper Review를 해 보았습니다.

<br>

* 다음에는 DCGAN의 실제 Code를 살펴보도록 하겠습니다.

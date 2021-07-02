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

  - **It is difficult to quantitatively evaluate GAN model performance.**
    * It is difficult to quantitatively determine how well model is trained from the sample, and it is also difficult for humans to evaluate it.

<br>
<br>
<br>
<br>
<br>
<br>

## 2. Purpose of DCGAN

<br>

* **Proof that the generator does not simply memorize and display sample data**
  - If the structure of the model is large enough compared to the sample data, and the training is carried out sufficiently, the model may memorize all the sample data.
  - It is hard to accept as Generating.
  - At first glance, it seems to have something to do with Overfitting...

<br>

* **WALKING IN THE LATENT SPACE**
  - The result of generation should occur naturally even with small changes in z of latent space. ( This is called as 'WALKING IN THE LATENT SPACE' in paper)

<br>
<br>
<br>
<br>
<br>

## 3. Architecture Guidelines

<br>

### 3.1. Overall

<br>

* Moving from the structure proposed by GAN to DCGAN, the overall structure remains the same, but some details have changed.

<br>

* Here's how the authors established the structures in the paper:
    - after extensive model exploration we identified a family of architectures that resulted in stable training across a range of datasets and allowed for training higher resolution and deeper generative models.
      * In paper, it is said in a grandiose way but it means that the best model structure was found with so many trials and errors.


<br>
<br>
<br>

### 3.2. Modification Points

<br>

* The methods to improve the performance of the existing GAN proposed by DCGAN are as follows.

<br>

  **1. Instead of pooling layers in G/D, strided convolutions are used for D and fractional-strided convolutions are used for G.**

<br>

  **2. Using Batch Normalization**

<br>

  **3. Not using FC(Fully Connected) Hidden Layer**

<br>

  **4. Except for the last layer in G, the activation function is RELU. The last layer uses tanh**

<br>

  **5. In D, LeakyRELU is used for activation functions of all layers.**
  
<br>

* Let's take a look at the modifications one by one.

<br>
<br>

#### 3.2.1. Add Convolution Layer

<br>

* First, let's look at **strded convolutions** and **fractional-strded convolutions** used instead of the Pooling Layer.

<br>

* Strided convolutions and fractional-strided convolutions are one of the convolution methods, but the size of the convolution applied to the existing CNN gradually decreases as it goes through the kernel. There is a difference that the size of strided convolutions and fractional-strded convolutions increase the kernel size through a specific operation.

<br>

* The following describes strided convolutions. Strided convolution is a convolution that move by stride.

<br>

<p align="center">
  <img src="/assets/DCGAN_Paper_Review/pic_01.gif">
</p>

<br>
<br>
<br>

* Fractional-strded convolutions are called transposed convolutions (sometimes called deconvolution, but this is not an exact concept).
* Strictly speaking, the method used by DCGAN's generator is **Transposed Convolution**.

<br>

<p align="center">
  <img src="/assets/DCGAN_Paper_Review/pic_02.gif">
</p>

<br>
<br>
<br>

* For more information on Transposed Convolution, please refer to the article at the link below.
  
  [CS231n의 Transposed Convolution은 Deconvolution에 가까운 Transposed Convolution이다](https://realblack0.github.io/2020/05/11/transpose-convolution.html#Transposed-Convolution)

<br>

* Transposed Convolution is implemented in Tensorflow as a function called **Conv2DTranspose()**.

<br>

* For a description of the various convolution methods, please, refer to the link below for more information.

  [An Introduction to different Types of Convolutions in Deep Learning](https://towardsdatascience.com/types-of-convolutions-in-deep-learning-717013397f4d)
  
<br>
<br>
<br>

#### 3.2.2. Apply Batch Normalization

<br>

* Batch Normalization is a concept introduced in paper, [Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift] (https://arxiv.org/abs/1502.03167) published in 2015 and it is widely applied neural network.

<br>

* Paper Link( [http://proceedings.mlr.press/v37/ioffe15.pdf](http://proceedings.mlr.press/v37/ioffe15.pdf) )

<br>

* The purpose of Batch Normalization is to prevent Gradient Vanishing / Gradient Exploding.

<br>

* Even before Batch Normalization, ReLU was used for the activation function or He or Xavier initialization was used for weight initialization.

<br>

* Unlike those indirect methods, Batch Normalization suppresses Gradient Vanishing / Gradient Exploding while directly participating in the training process.

<br>

* Please refer to the link below for detailed explanation.

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

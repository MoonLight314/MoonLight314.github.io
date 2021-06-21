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
<p align="left">
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

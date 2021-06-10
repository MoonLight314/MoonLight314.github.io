---
title: "GAN"
date: 2021-06-10 08:26:28 -0400
categories: Deep Learning
---

### GAN(Generative Adversarial Nets)

<br>
<br>
<br>
<br>
<br>
<br>

## 0. Introduction   

* 2014년에 Ian J. Goodfellow가 GAN(Generative Adversarial Nets)이라는 새로운 방식의 Model을 발표하였습니다.



* 논문은 아래 Link에서 확인할 수 있습니다.

  [Generative Adversarial Nets](https://papers.nips.cc/paper/2014/file/5ca3e9b122f61f8f06494c97b1afccf3-Paper.pdf)
  
  
* 'Nets'은 흔히 알고 있는 Network인 것은 알겠지만, Adversarial이나 Generative의 정확한 의미는 조금 어렵습니다.

  
* 이번 Post에서는 GAN이 무엇인지 알아보도록 하겠습니다.
<br>
<br>
<br>
<br>
<br>
<br>

## 1. Background

### 1.1. Generative Model   

* GAN은 실제로 존재하지 않지만, **그럴싸한(있을법한) Data를 생성할 수 있는 Model**의 종류를 말합니다.


* 아래 Image는 Ian Goodfellow의 Papaer에 나와있는 결과물입니다.


* 파란색은 실제 Image Data이며, 빨간색은 GAN이 생성한 Image입니다.  정말 진짜 같네요.

<br>
<br>

<p align="left">
  <img src="/assets/GAN/pic_00.png">
</p>

<br>
<br>


---
title: "Backpropagation"
date: 2024-06-23 08:26:28 -0400
categories: Deep Learning
---

<br>
<br>

이번 Post에서는 Backpropagation에 관해서 알아보도록 하겠습니다.

<br>

# 0. Introduction

<br>

Backpropagation은 Deep Learning 학습의 핵심 메커니즘입니다.

Deep Learning의 학습은 학습하려는 Train Data를 Network에 넣어서 값을 출력하는 순서를 거치게 됩니다.

이 과정을 Feedforward라고 합니다. 물론 이 과정에서 출력되는 값은 Network이 학습이 진행되기 전이기 때문에 실제 Target 값과 많이 차이가 나게 됩니다.

Deep Learning은 Target 값과 실제 정답과의 차이를 이용하여 Network을 구성하는 Parameter(Weight , Bias)를 적절하게 Update 합니다.

이 과정을 Backpropagation이라고 하며, 이번 Post에서는 Backpropagation 과정이 실제로 어떻게 동작하는지 알아보겠습니다.

<br>

# 1. Feedforward

<br>

먼저, Train Data로 Network이 Target 값을 계산하는지 알아보겠습니다.

다음과 같은 Network이 있다고 해보겠습니다. 매우 Simple한 구조지만, 모든 Network은 이 구조의 확장이기 때문에 설명하는데 충분하다고 생각합니다.

Activation Function은 Sigmoid라고 가정하겠습니다. 아시다시피 Sigmoid의 수식은 아래와 같습니다.

![image](https://github.com/MoonLight314/MoonLight314.github.io/assets/41887456/0e6958f2-8d9b-4a6b-b616-f7686505b093)


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

<br>
<br>

<p align="center">
  <img src="/assets/Backpropagation/pic_00.png">
</p>

<br>
<br>

Activation Function은 Sigmoid라고 가정하겠습니다. 아시다시피 Sigmoid의 수식은 아래와 같습니다.

<br>

![image](https://github.com/MoonLight314/MoonLight314.github.io/assets/41887456/0e6958f2-8d9b-4a6b-b616-f7686505b093)

<br>

Loss Function은 MSE(Mean Squared Error)라고 가정하겠습니다. MSE를 구하는 수식은 아래와 같습니다.

<br>

![image](https://github.com/MoonLight314/MoonLight314.github.io/assets/41887456/3ac22cc7-4c5a-46a7-901f-9dfe63741e00)

<br>


먼저, 첫번째 Activation Function을 통과한 $ℎ_1=𝜎(𝑥_1𝑤_1+𝑥_2 𝑤_2+𝑏_1)$이고, $ℎ_2=𝜎(𝑥_1𝑤_3+𝑥_2𝑤_4+𝑏_2)$가 됩니다.

그리고, 두번째 Activation Function을 통과한 $𝑦_1=𝜎(ℎ_1𝑤_5+ℎ_2𝑤_7+𝑏_3)$이고, $𝑦_2=𝜎(ℎ_1𝑤_6+ℎ_2𝑤_8+𝑏_4)$가 됩니다.

<br>

# 2. Backpropagation

<br>

입력값 $x_1,x_2$ 가 신경망을 거쳐서 출력값 $\hat{𝑦_1}, \hat{𝑦_2}$가 계산되어 나왔습니다.

이 $\hat{𝑦_1}, \hat{𝑦_2}$가 실제 Target Value인 $𝑦_1, 𝑦_2$와 얼마나 차이가 나느냐 계산해서 이 차이만큼 신경망의 parameter(w,b)들을 update 시켜야 합니다.

이 동작의 반복을 ‘학습’이라고 합니다.

<br>

## 2.1. Loss Function

<br>

다양한 Loss Function이 존재하고 상황에 맞는 Loss Function을 선택해야 합니다.

이 Post에서는 가장 심플하다고 할 수 있는 MSE를 선택하기로 했습니다.

MSE는 신경망 출력값과 실제 Target 값의 차이를 구해서 제곱한 후 평균을 구하는 방식을 취합니다.

<br>

## 2.2. 편미분

<br>

Loss Function을 통해 실제값과 신경망이 구한 값의 차이, 즉, 에러 신호를 구했으면 이 에러신호가 신경망의 어떤 Parameter에 의해서 얼마나 영향을 받는지를 확인해서 신경망의 Parameter를 Update해 주어야 합니다.

구체적으로, 에러 신호를 Feedforward의 반대 방향으로 넘어가면서(Backpropagation) 편미분을 통해서 특정 Parameter가 에러 신호에 어느 정도 영향을 주는지 계산(미분)해서 개별적으로 Parameter를 에러 신호가 줄어드는 방향으로 Update해 나가는 동작을 반복하게 됩니다.

<br>

# 3. Example of Backpropagation

<br>

실제 편미분을 이용해 Parameter를 Update하는 방법을 알아보겠습니다.

먼저 Backpropagation은 Feedforward를 구성하는 각 Function들의 미분을 해야 하기 때문에, 

Activation Function와 Loss Function의 도함수부터 먼저 알아보도록 하겠습니다.

<br>

## 3.1. Activation Function의 도함수

<br>

앞서 살펴봤듯이, Sigmoid의 형태는 아래와 같습니다.

![image](https://github.com/MoonLight314/MoonLight314.github.io/assets/41887456/0e6958f2-8d9b-4a6b-b616-f7686505b093)

<br>

Sigmoid의 도함수는 다음과 같습니다.

<br>

![image](https://github.com/MoonLight314/MoonLight314.github.io/assets/41887456/3a54687e-88f3-41cf-b506-6895c76aea56)

<br>

## 3.2. Loss Function의 도함수

<br>

우리는 Loss Function으로 MSE를 사용하기로 했으며, 수식은 아래와 같습니다.

<br>

![image](https://github.com/MoonLight314/MoonLight314.github.io/assets/41887456/edabf4b0-73d1-4850-9fa1-e4066700228f)

<br>

MSE의 도함수의 형태는 아래와 같습니다.

<br>

![image](https://github.com/MoonLight314/MoonLight314.github.io/assets/41887456/6dd4f2ca-7fdd-4271-ae5a-b533fe71214f)

<br>

도함수를 구하는 구체적인 순서는 생략하기로 하겠습니다.

<br>

## 3.3. Overall Process

<br>

![image](https://github.com/MoonLight314/MoonLight314.github.io/assets/41887456/105c8c4c-30ef-4fe9-87f0-f2aa58992bdc)

<br>


---
title: "Batch Normalization"
date: 2021-07-05 08:26:28 -0400
categories: Deep Learning
---

### Batch Normalization

<br>
<br>
<br>
<br>
<br>
<br>

## 0. Introduction   

<br>

### 0.1. Gradient Vanishing / Exploding

<br>

* Neural Network의 Train시에 Gradient 값의 변화를 보고 Parameter를 조절합니다.

<br>

* Gradient는 변화량, 즉 미분값입니다.  Neural Network의 깊이가 깊어질수록 Backpropagation시에 Gradient 값들이 Input Layer의 입력값의 변화를 적절하게 학습에 반영하지 못합니다.

<br>

* Backpropagation시에, Non-Linear Activation Function(Ex. Sigmoid / Tanh )들을 사용하면 Layer를 지날수록 Gradient 값들이 점점 작아지거나(Gradient Vanishing) 혹은 반대로 Gradient 값들이 점점 커져서(Gradient Exploding), Input Layer의 변화량에 따른 Output Layer의 변화량을 Neural Network의 Parameter에 제대로 반영을 하지 못하는 상황이 발생합니다.

<br>

* 이 문제를 개선하기 위해서 몇가지 방법들이 제시되었습니다.         

<br>
<br>
<br>

### 0.2. Countermeasure of Gradient Vanishing / Exploding

* **ReLU 사용**

  근본적인 원인이 Non Linear Activation Function을 사용했기 때문에, Linear Activation Function인 ReLU를 사용하는 방법입니다.

<br>

* **Special Weight Initialization**

  Layer의 Weight 값을 초기화를 잘하면 이 문제를 개선할 수 있다고 알려져 있습니다.
  He Initialization 혹은 Xavier Initialization이 잘 알려진 Weight Initialization 방법들입니다.

<br>

* **작은 Learning Rate 사용**      

<br>
<br>
<br>

### 0.3. Fundamental Solution   

<br>

* 위에서 언급한 방법들은 모두 간접적으로 Gradient Vanishing / Exploding를 개선하는 방법들입니다.

<br>

* 근본적인 해결 방안을 연구하다가 나온 것은 Batch Normalization입니다.   

<br>
<br>
<br>
<br>
<br>
<br>

## 1. Batch Normalization

<br>

* Batch Normalization이 소개된 Paper의 제목은 'Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift'입니다.

<br>

* Paper의 Link는 여기를 참조해 주시기 바랍니다.

<br>

  [https://arxiv.org/pdf/1502.03167.pdf](https://arxiv.org/pdf/1502.03167.pdf)   

<br>
<br>
<br>

### 1.1. Advantage of Batch Normalization

<br>

알려진 Batch Normalization장점은 다음과 같습니다.   

<br>   

- **학습 속도가 빨라집니다.**
  
  가장 큰 장점이기도 합니다. 성능은 Batch Normalization을 적용하지 않는 것과 동일 혹은 더 좋아지면서도 적은 Epoch으로도 빠르게 수렴합니다.
  어떤 분은 최소 10배 이상 빨라진다고 하네요.

<br>

- **Network의 Hyper Parameter에 대한 민감도가 감소합니다.**
  
  성능이 잘 나오지 않을 때 열심히 Hyper Parameter Tuning을 했는데, 이제 그것에 대한 부담을 크게 줄일 수 있습니다.

<br>

- **일반화(Regularization) 효과가 있습니다.**
  
  Inference 시에 성능이 더 좋아집니다.

<br>
<br>
<br>


### 1.2. Adding Batch Normalization Layer

<br>

* Batch Normalization Layer은 아래 그림과 같이 Hidden Layer 중간에 넣어주면 됩니다.   

<br>

<p align="center">
  <img src="/assets/Batch_Normalization/pic_00.png">
</p>

<br>
<br>


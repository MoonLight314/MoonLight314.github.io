---
title: "Batch Normalization (EN)"
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

* It tunes parameters by examining changes in gradient values during neural network training.

<br>

* Gradient(derivative) is the amount of change. As the depth of the neural network depp, the gradient values do not properly reflect the change in the input value of the input layer during backpropagation in training.

<br>

* During backpropagation, if you use non-linear activation functions (Ex. Sigmoid / Tanh ), the gradient values become smaller (Gradient Vanishing) or conversely, the gradient values become larger (Gradient Exploding) as the layer passes through. A situation arises that the amount of change in the output layer cannot be properly reflected in the parameters of the neural network.

<br>

* Several methods have been proposed to improve this problem.

<br>
<br>
<br>

### 0.2. Countermeasure of Gradient Vanishing / Exploding

* **Applying ReLU**

  Since the root cause is the use of the non linear activation function, this is the method of using the linear activation function like ReLU.

<br>

* **Special Weight Initialization**

  It is known that this problem can be improved by properly initializing the layer's weight value.
  He initialization or Xavier initialization are well-known weight initialization methods.

<br>

* **Applying Small Learning Rate**

<br>
<br>
<br>

### 0.3. Fundamental Solution   

<br>

* The methods mentioned above are all indirect ways to improve Gradient Vanishing / Exploding.

<br>

* Batch Normalization came out of researching a fundamental solution.

<br>
<br>
<br>
<br>
<br>
<br>

## 1. Batch Normalization

<br>

* The paper introducing Batch Normalization is titled 'Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift'.

<br>

* Please, refer to the below link of paper

  [https://arxiv.org/pdf/1502.03167.pdf](https://arxiv.org/pdf/1502.03167.pdf)   

<br>
<br>
<br>

### 1.1. Advantage of Batch Normalization

<br>

Known Batch Normalization benefits are :

<br>   

- **Training speed gets faster.**
  
  It might be the most important advantage. The performance is the same or better than that of not applying batch normalization, but converges quickly with fewer epochs.
  Some says it's at least 10 times faster.

<br>

- **The sensitivity of the network to hyper parameter is reduced.**
  
  Generally, we tune hyper parameter  when the performance wasn't good. It can greatly reduce the burden on it.

<br>

- **There is a regularization effect.**
  
  Better performance during inference.

<br>
<br>
<br>


### 1.2. Adding Batch Normalization Layer

<br>

* Add Batch Normalization layer in the middle of the hidden layer as shown in the figure below.

<br>

<p align="center">
  <img src="/assets/Batch_Normalization/pic_00.png">
</p>

<br>
<br>
<br>
 

* More specifically, it is said that putting it in front of the activation function gives better results experimentally.

<br>

* **The key to batch normalization is to properly control the output value of the previous layer before passing it on to the next layer by adding two parameters γ and β as many as the number of output neurons in the previous layer.**

<br>

* There is no reason not to apply batch normalization because the performance of the network increases dramatically by adding only the cost of calculating two parameters.

<br>
<br>
<br>
<br>
<br>
<br>

## 2. Background

<br>

### 2.1. Normalization

<br>

* The normalization technique that makes each feature have a similar range of values is used in many fields, and it improves the training speed.

<br>

* The reason why the training speed increases when applying normalization is that it is difficult to use a large learning rate if the variation for each feature is different, so the training speed cannot be increased quickly.

<br>
<br>
<br>

### 2.2. Standardization

<br>

* A method similar to normalization is standardization.

<br>

* This is a method to make the distribution of features a mean of 0 and a variance of 1.

<br>
<br>
<br>
<br>

## 3. Implementation of Batch Normalization

<br>

* It is simple to put a normalized feature in the first input layer. It can be normailzed while pre-processing and input into network.

<br>

* However, if the input data distribution of the hidden layer is different every time and it is not normalized, the network would not be trained well.

<br>

* Then, how would it be if the input data is normalized each hidden layer ? The simplest and most ignorant way is to normalize it before passing it on to each hidden layer.

<br>
<br>
<br>

### 3.1. Batch Normalization

<br>

* The formula below is the calculation method of the input and output values of batch normalization shown in the paper.

<br>

<p align="center">
  <img src="/assets/Batch_Normalization/pic_01.png">
</p>

<br>
<br>

* **1** : First, let's look at the values that are input to batch normalization.
   
   In mini-batch, it comes in as input as x1 ~ xm. Here, you can think of m as the batch size. More specifically, it will be the activation output value of the previous layer.

<br>

* **2** : These two values of γ and β are the values we actually need to train. With these values, you control the values that will go into the next layer. Two γ and β for each input value.

<br>

* **3** : Calculating the average value of mini-batch.

<br>

* **4** : Calculating the variance of mini-batch.

<br>

* **5** : Normalizing.

<br>

* **6** : Adding a very small value to prevent division by zero.

<br>

* **7** : Finally, it is the output value that has been applied to batch normalization.

<br>
<br>

* γ,(Scale 역할) ,  β(Bias 역할)는 Neural Network 성능이 향상되는 방향(Loss가 작아지는 방향)으로  학습이 진행되는 과정을 거칩니다.   

<br>
<br>
<br>
<br>

## 4. Effect of Batch Normalization

<br>

* Batch Normalization으로 얻을 수 있는 효과는 논란의 여지가 없습니다.

<br>

* 앞서 말했듯이, Train Speed가 빠르며 Hyper Parameter Tuning으로부터 자유로와 집니다.

<br>

* 아래의 다양한 조건에서의 Train 결과를 보면, Batch Normalization을 사용한 경우가 확실히 Train Speed가 빠르며, 큰 Learning Rate를 사용해도 빠르게 수렴하는 것을 알 수 있습니다.   

<br>   
<br>

<p align="center">
  <img src="/assets/Batch_Normalization/pic_02.png">
</p>

<br>
<br>
<br>
<br>
<br>

## 5. Algorithm of Training & Inference

<br>

* 이제 Paper에서 소개된 Train / Inference시의 Batch Normalization 적용법에 대해서 알아보도록 하겠습니다.

<br>

* Batch Normalization 적용시에 다른 Network과 다른 점이라면, **Train시의 Network과 Inference시의 Network이 다르다는 점입니다.**

<br>

* 다른 이유는 Train시에 사용된 Data의 평균과 분산이 Inference시의 Data의 평균과 분산이 다르기 때문입니다.   

<br>
<br>

<p align="center">
  <img src="/assets/Batch_Normalization/pic_03.png">
</p>

<br>

* **1** : 이 Algorithm의 최종 목적은 Batch Normalization이 적용된 Inference를 하기 위한 Network을 얻는 것입니다.

<br>

* **2** : 기존 Network에 Batch Normalization Layer를 추가하겠다는 의미입니다.

<br>

* **3** : Batch Normalization Layer가 추가된 상태에서 γ, β를 학습을 합니다.

<br>

* **4** : 학습을 마친 후에, Parameter를 고정 한 후에 이제부터는 Inference Mode로 Network을 바꿉니다.

<br>

* **5** : 몇 개의 Train용 Mini Batch를 이용해 평균 분산을 구합니다.

<br>

* **6** : Inference시에는 이동 평균값을 이용합니다.   

<br>
<br>
<br>
<br>

## 6. Essence of Batch Normalization

* 이번에는 Batch Normalization이 어떤 원리로 이렇게 좋은 성능을 내는지 알아보도록 하겠습니다.   

<br>
<br>

### 6.1. Covariate Shift ( 공변량 변화)

<br>

* Covariate Shift란 학습에 사용된 Data Distribution과 Test시에 사용된 Data Distribution이 변경되는 현상을 말합니다.

<br>   
<br>

<p align="center">
  <img src="/assets/Batch_Normalization/pic_04.png">
</p>
   
<br>
<br>
<br>  

### 6.2. Internal Covariate Shift ( ICS )

<br>

* Internal Covariate Shift가 Network 내부에서 발생하는 현상을 의미하며, 이는 Batch Normalization이 최초 해결하고자 했던 문제입니다.

<br>

* 뒤쪽 Layer에서는 매번 Input Data Distribution이 달라진다는 것을 의미하며 Layer가 깊어질수록 이는 점점 더 심각해집니다.

<br>

* Batch Normalization이 발표된 이후에는 Batch Normalization의 성능 향상 요인이 ICS가 감소되기 때문이라고 알고 있었으나, **후속 연구에서 Batch Normalization의 효과가 ICS 감소와는 상관 없다는 주장이 제기됩니다.**

<br>

* Paper - How Does Batch Normalization Help Optimization? 
  
  ( [https://www.notion.so/Batch-Normalization-96e9fd9a661d4762b7cd978ce2092809#48cb8f4c97234971966943ce33a6f1c5](https://www.notion.so/Batch-Normalization-96e9fd9a661d4762b7cd978ce2092809) )   
  
<br>
<br>
<br>

### 6.3. Batch Normalization & Internal Covariate Shift ( ICS )

<br>

* Batch Normalization Layer 바로 다음에 Random Noise를 삽입하여 강제로 ICS를 발생시켜도 여전히 좋은 성능을 보여줍니다.   

<br>
<br>

<p align="center">
  <img src="/assets/Batch_Normalization/pic_05.png">
</p>
   
<br>
<br>
<br>

### 6.4. Smoothing Effect of Batch Normalization

<br>

* **Batch Normalization는 Optimization Landscape를 부드럽게(Smoothing)하는 효과가 있습니다.**

<br>

* Optimization Landscape란 Weight 따른 Loss의 변화를 시각화한 것입니다.

<br>

* 아래 그림이 Optimization Landscape인데, 왼쪽이 Batch Normalization을 적용하지 않는 것이고, 왼쪽이 Batch Normalization을 적용한 것입니다.

<br>

* 보시다시피, 왼쪽 Optimization Landscape의 경우, Weight Domain에서 Loss가 굉장히 들쑥날쑥 한 것을 확인할 수 있습니다.

<br>

* 반면에 오른쪽 Optimization Landscape의 경우, Loss값 분포가 매우 Smooth한 것을 알 수 있습니다.   

<br>

<p align="center">
  <img src="/assets/Batch_Normalization/pic_06.png">
</p>
   
<br>

* 이런 Smoothing 효과가 있는 것은 **Batch Normalization 뿐만 아니라, Residual Connection도** 가지고 있다고 알려져 있습니다.

<br>

* Predictiveness(기울기 예측성)이란 현재의 기울기 방향성을 얼마나 신뢰할 수 있는 가를 나타내는 수치입니다.

<br>

* 첫번째 그림에서 Batch Normalization 적용시에는 Loss 자체의 변화가 작고, 많이 이동을 해도 Loss 변동이 크지 않습니다.

<br>

* 두번째 그림에서도 마찬가지로 큰 Step으로 기울기를 이동해도 많이 다르지 않습니다.   

<br>
<br>

<p align="center">
  <img src="/assets/Batch_Normalization/pic_07.png">
</p>

<br>
<br>

* 종합하면, Batch Normalization을 적용하면 현재의 기울기 방향으로 큰 Step만큼 이동을 해도 이동한 뒤의 기울기가 현재와 유사할 가능성이 높다는 의미입니다.   

---
title: "Flash Attention"
date: 2025-03-26 08:26:28 -0400
categories: Deep Learning
---

<br>
<br>

기존 Attention 메커니즘을 Flash Attention에 대해서 알아보도록 하겠습니다.

Flash Attention은 Stanford 연구진에 의해 제안되었으며, 기존 Transformer 모델의 핵심 구성 요소인 Attention 메커니즘을 개선한 기술입니다.

<br>

## 0. 기존 Attention 메커니즘의 문제점​
<br>
기존의 Attention 메커니즘인 Scaled Dot-Product Attention은 Transformer 모델의 핵심 요소로서, 입력 텍스트의 관련 부분에 집중하여 예측을 수행하는 데 효과가 있습니다.

​

이 메커니즘은 Query, Key, Value라는 세 가지 요소를 사용하여 계산하며, 수식은 비교적 간단하지만 실제 계산 과정에서는 큰 텐서들의 곱셈과 상당한 데이터 이동이 발생합니다. 

​

특히, 시퀀스 길이가 n일 때, Attention 메커니즘은 O(n²)의 추가 메모리와 O(n²)의 계산 시간 복잡도를 가지게 됩니다. 이는 긴 시퀀스를 처리할 때 심각한 병목 현상을 발생시키며, 모델이 효과적으로 처리할 수 있는 문맥 길이를 제한하고 필요한 컴퓨팅 자원을 크게 증가시키는 원인이 됩니다.

​

또한, Attention 메커니즘은 Key, Query 값을 저장, 읽기, 쓰기 위해 고대역폭 메모리(HBM)를 사용하는데, 이러한 데이터를 더 빠른 온칩 SRAM으로 반복적으로 전송하는 과정은 상당한 비용을 발생시키며, GPU가 실제 연산보다 데이터 전송에 더 많은 시간을 소비하게 만드는 메모리 바운드(memory-bound) 연산을 발생시킵니다.

<br>
<br>
<br>

## 1. Flash Attention의 주요 특징
<br>

### 1.1. 속도 향상

Flash Attention은 이전 버전보다 더 빠른 연산 속도를 제공하는데, 이는 GPU 메모리 계층 구조를 더 효율적으로 활용하고, 불필요한 메모리 액세스를 줄이는 방식으로 달성됩니다. 

​

구체적으로, 다음과 같은 기술을 사용합니다.

1) Tiling : 입력 데이터를 더 작은 타일로 나누어 처리하여, GPU의 공유 메모리를 최대한 활용합니다.

2) Kernel Fusion : 여러 연산을 하나의 커널로 융합하여 커널 실행 오버헤드를 줄입니다.

3) Parallel Reduction : 병렬 연산을 통해 Attention 가중치를 계산하고 정규화하는 과정을 가속화합니다.

​
<br>
​

### 1.2. 메모리 효율성

Flash Attention은 기존 Attention 연산에 필요한 메모리 사용량을 크게 줄입니다. 이를 통해 더 긴 시퀀스를 처리하거나, 더 큰 모델을 훈련하는 것이 가능해집니다.  메모리 사용량 감소를 시키는 기법은 다음과 같은 방법이 있습니다.

Attention 가중치 저장 최소화 : Attention 가중치를 완전히 저장하는 대신, 필요할 때마다 즉석에서 계산하여 메모리 공간을 절약합니다.

backward pass 최적화 : 역전파 과정에서 필요한 중간 결과를 효율적으로 계산하고 저장하여 메모리 사용량을 줄입니다.

​

### 1.3. 다양한 하드웨어 지원

Flash Attention은 다양한 GPU 환경에서 잘 작동하도록 설계되었습니다. NVIDIA GPU 뿐만 아니라 AMD GPU에서도 최적의 성능을 낼 수 있도록 지원합니다.

​<br>
<br>

## 2. Flash Attention의 동작 원리

Flash Attention의 성능 향상의 가장 큰 역할을 하는 것은 Tiling 기법과 계산시에 Global Memory(HBM, High Bandwidth Memory)를 사용하는 대신 Shared Memory를 사용하는 것, 이 2가지입니다.

​

​

### 2.1. Tiling

<br>

#### 2.1.1. 기존 Attention 메커니즘의 문제점

​

기존의 표준 Attention 메커니즘은 다음과 같은 단계를 거칩니다.
​

1) Query, Key, Value 생성 : 입력 시퀀스로부터 Query(Q), Key(K), Value(V) 행렬을 생성합니다.

2) Attention Score 계산 : Q와 K의 내적(dot product)을 계산하여 Attention Score 행렬을 얻습니다.

3) Softmax 적용 : Attention Score 행렬에 Softmax 함수를 적용하여 Attention Weight 행렬을 얻습니다.

4) 가중합 계산 : Attention Weight 행렬과 V 행렬을 곱하여 최종 Attention 출력값을 얻습니다.

​

이 과정에서 가장 큰 문제는 2단계와 3단계에서 발생하는 큰 중간 결과물(Attention Score 및 Weight 행렬)을 GPU의 고대역폭 메모리(HBM)에 저장해야 한다는 것입니다. 시퀀스 길이가 길어질수록 이 행렬들의 크기는 제곱으로 증가하여, 메모리 병목 현상이 발생하고 속도가 느려집니다.


좀 더 자세한 Transformer의 Attention 알고리즘에 대한 내용은 아래 글을 참고해 주시기 바랍니다.

[Transformer #1 - Attention Mechanism](https://moonlight314.github.io/deep/learning/Transformer_Attention_Mechanism/)

[Transformer #2 - Self Attention](https://moonlight314.github.io/deep/learning/Transformer_Self_Attention/)

[Transformer #3 - Overall](https://moonlight314.github.io/deep/learning/Transformer_Overall/)

[Transformer #4 - Encoder Detail](https://moonlight314.github.io/deep/learning/Transformer_Encoder_Detail/)

[Transformer #5 - Decoder Detail](https://moonlight314.github.io/deep/learning/Transformer_Decoder_Detail/)

<br>

#### 2.1.2. Tiling 기법

​FlashAttention-2의 핵심은 Tiling이라는 방법인데, Tiling은 아래 그림과 같이 Q, K, V 행렬을 각각 N/B개의 블록(타일)으로 나누어서 계산하는 것입니다. 각 블록의 크기는 B x d가 되고, B는 하이퍼파라미터로 조절 가능합니다.

![image](https://github.com/user-attachments/assets/461c6160-0f1f-4a88-af25-9950b402af83)



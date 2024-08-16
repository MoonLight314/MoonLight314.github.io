---
title: "Learning to (Learn at Test Time): RNNs with Expressive Hidden States"
date: 2024-08-15 08:26:28 -0400
categories: Deep Learning
---

<br>

# 0. 소개

<br>

안녕하세요. 현재 AI 분야를 평정하고 있는 생성형 AI(Generative AI)의 기본은 Self-Attention을 기반으로 하는 Transformer 구조입니다.

Transformer는 지금까지 발표된 어떤 구조보다도 우수한 성능을 가지지만, 유일한 단점이라면 Model의 표현력과 비례해서 Hidden State도 커져야 한다는 것입니다.

최근 발표된 Llama 3.1은 Model을 구성하는 Parameter의 개수가 4000억 개가 넘는다고 하죠.

이와 같은 Transformer의 구조는 한계를 극복하고자 발표된 Model이 TTT(Test-Time Training)이라는 구조로써, 핵심은 Hidden State 값들을 개별적으로 모두 저장하는 것이 아니라,

Hidden State 값을 표현하는 Machine Learning Model을 만들고 이를 TTT(Test-Time Training) Layer라고 부릅니다.

**이 TTT(Test-Time Training) Layer(Layer라고는 하지만 실제로는 Machine Learing Model입니다) 자체의 Weight는 Self-Supervised 방식으로 Update 하도록 한다는 것이 핵심입니다.**  
( We propose a new class of sequence modeling layers with linear complexity and an expressive hidden state. The key idea is to make the hidden state a machine learning model itself, and the update rule a step of self-supervised learning. )

논문 Link를 아래에 남겨두었으니, 참고해 주세요

Github : https://github.com/test-time-training/ttt-lm-pytorch?tab=readme-ov-file[https://github.com/test-time-training/ttt-lm-pytorch?tab=readme-ov-file]

Paper : https://arxiv.org/abs/2407.04620[https://arxiv.org/abs/2407.04620]

PDF : https://arxiv.org/pdf/2407.04620[https://arxiv.org/pdf/2407.04620]


<br>
<br>

# 1. 기존 방식

<br>
<br>

![image](https://github.com/user-attachments/assets/9e164342-d78f-4bc1-8d44-ecea3d8099ea)

<br>
<br>

Fig. 1. 은 지금까지 나온 대표적인 기법들에 대한 Hidden State Update 방식을 비교한 표입니다.

RNN & LSTM 구조와 같은 경우에는 Context를 고정된 길이로 압축을 해야 합니다. 

이렇게 하는 경우에는 일정 시간 안에 처리되어 빠르다는 장점이 있지만, 어떤 길이의 Context라도 일정 길이로 압축되기 때문에 긴 문장의 경우에는 성능 저하가 필연적입니다.

반대로, Self-Attention은 Key-Value Tuple를 List에 추가하는 방식을 택하고 있으므로 문장의 길이가 거의 성능에 영향을 미치지 않습니다.하지만, Key-Value를 저장하는 List의 크기도 선형적으로 증가하기 때문에 KV List를 Scan 하는 시간도 선형적으로 증가하게 된다는 단점이 있습니다.





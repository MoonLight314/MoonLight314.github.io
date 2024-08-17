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

<br>
<br>

# 2. 실험 결과

<br>
<br>

TTT Layer를 사용한 Model의 Test 결과를 살펴보도록 하겠습니다.

그전에 Test 결과에 자주 등장하는 Perplexity라는 것에 대해서 살짝 알아보도록 하겠습니다.

<br>

## 2.0. Perplexity

Perplexity는 Language Model의 성능을 평가하는 데 사용되는 지표로, Model이 예측하는 확률 분포의 불확실성을 측정합니다. 

Perplexity는 Model이 주어진 텍스트를 얼마나 잘 예측하는지 나타내며, 낮을수록 Model의 예측이 더 정확함을 의미합니다.

<br>

### 2.0.0. Perplexity의 의미

<br>

낮을수록 Model이 텍스트를 더 잘 예측하고, 따라서 Model의 성능이 더 좋음을 의미합니다. 

예를 들어, Perplexity 값이 1에 가까울수록 모델의 예측이 완벽하다는 것을 나타냅니다.

높을수록 Model의 예측이 부정확하고 불확실성이 높음을 의미합니다.

이는 Model이 텍스트의 다음 단어를 예측하는 데 어려움을 겪고 있음을 나타냅니다.

<br>

### 2.0.1. Example

<br>

Perplexity가 10이라면, Model이 다음 단어를 예측할 때 평균적으로 10개의 선택지 중 하나를 고르는 것과 같다는 의미입니다.

Perplexity가 100이라면, Model이 다음 단어를 예측할 때 평균적으로 100개의 선택지 중 하나를 고르는 것과 같다는 의미입니다.

이는 Model의 예측이 더 불확실함을 나타냅니다.

Perplexity는 Language Model의 성능을 비교하는 데 중요한 지표로 사용되며, 특히 Model이 얼마나 효율적으로 Language Pattern을 학습했는지를 평가하는 데 유용합니다.

<출처 : https://en.wikipedia.org/wiki/Perplexity[https://en.wikipedia.org/wiki/Perplexity]>

<br>

## 2.1. Short context: the Pile

<br>

![image](https://github.com/user-attachments/assets/5301373c-7798-4019-be4e-9c08354de82d)

<br>
<br>

Fig. 2.는 the Pile Dataset에서 Context Length 2k 및 8k에 대한 Test 결과를 나타냅니다. 

TTT-Linear는 2k에서 Mamba와 비슷한 성능을 보이며, 8k에서는 더 나은 성능을 보입니다.

### 2.1.0 2k Context Length

<br>

TTT-Linear (M), Mamba, 그리고 Transformer의 성능은 거의 동일합니다. TTT-MLP (M)은 큰 FLOP 예산에서 약간 더 나쁜 성능을 보입니다. TTT-MLP는 모든 모델 크기에서 TTT-Linear보다 더 낮은 perplexity를 가지고 있지만, 추가적인 FLOP 비용이 그 이점을 상쇄시킵니다.

<br>

### 2.1.1 8k Context Length

<br>

TTT-Linear (M)과 TTT-MLP (M)은 Mamba보다 훨씬 더 좋은 성능을 보이며, 이는 2k에서의 관찰과는 대조적입니다. 

Transformer 백본을 사용하는 TTT-MLP (T)조차도 약 1.3B 크기에서는 Mamba보다 약간 더 좋은 성능을 보입니다. 

이 논문 전반에서 관찰되는 강력한 현상은 문맥 길이가 길어질수록 TTT 레이어가 Mamba보다 더 큰 이점을 가진다는 것입니다.

또한, Transformer는 여전히 모든 모델 크기에서 좋은 (어쩌면 최고 수준의) perplexit를 가지지만, FLOP 비용 때문에 경쟁력 있는 성능을 보이지 않습니다.

<br>

### 2.1.2 Backbone의 효과

<br>

TTT Layer를 Mamba Backbone에서 Transformer Backbone 으로 전환하면 두 가지 효과가 있습니다. 

첫째, TTT Layer가 Mamba Backbone 에서 더 좋은 성능을 보입니다. 

둘째, Mamba Backbone 을 사용한 경우 TTT-MLP는 최대한 TTT-Linear와 동등한 수준이지만, Transformer Backbone 을 사용하면 TTT-MLP가 명확히 더 나은 성능을 보입니다. 

<br>

## 2.2. Long context: Books

<br>

![image](https://github.com/user-attachments/assets/9033ea27-6acf-4ee1-bf60-bc8e32065201)

<br>

Fig. 3.은 Books Dataset에서 Context Length 2k와 32k에 대한 Test결과입니다. 

<br>

긴 Context에서의 성능을 평가하기 위해, Pile의 Books3라는 인기 있는 하위 집합을 사용하여 문맥 길이를 1k에서 32k까지 2배씩 증가시키며 Test를 진행했습니다. 

Train Recipe는 Pile과 동일하며, 모든 TTT Layer에 대한 Test는 한 번의 Train 실행에서 수행되었습니다. 

<br>

### 2.2.0 Books의 2k Context Length

<br>

Pile 2k에서의 모든 관찰은 여전히 유효하나, 이제 Mamba가 TTT-Linear보다 약간 더 나은 성능을 보입니다.
(둘의 성능이 Pile 2k에서는 거의 동일했음).

<br>

### 2.2.1 Books의 32k Context Length

<br>

TTT-Linear (M)와 TTT-MLP (M)가 Mamba보다 더 나은 성능을 보이며, 이는 Pile 8k에서의 관찰과 유사합니다. 

심지어 Transformer Backbone을 사용하는 TTT-MLP (T)도 32k 문맥에서 Mamba보다 약간 더 나은 성능을 보입니다.

TTT-MLP (T)는 1.3B에서 TTT-MLP (M)보다 약간 더 낮은 성능을 보입니다. 

<br>

# 3. 마치며

<br>


최근 들어서 AI 분야에서는 기존의 영향력이 큰 기술들이나 기업들에서 탈출하고자 하는 움직임들이 많이 보이는 것 같습니다.

nVidia GPU 구조를 탈피하여 좀 더 가격 경쟁력이 있는 구조를 선보이는 노력을 한다거나, 이 Paper에서 제시한 TTT Layer와 같이 Transformer 구조의 한계를

극복하고자 하는 노력등이 대표적이라고 할 수 있겠습니다.


저는 이러한 방향성과 노력들이 매우 훌륭하다고 생각합니다.

이제 쓸만한 Model들은 너무 무거워져서 개인이나 작은 규모의 업체에서는 어지간한 Model은 돌릴 엄두조차 내지 못하는 상황에서 성능은 그대로 유지하면서 좀 더 경량화된

Model 구조나 효율이 좋은 HW에 대한 연구는 환영받을만 하다고 생각하고, 앞으로도 이런 방향의 연구가 많이 활성화 되었으면 합니다.

<br>

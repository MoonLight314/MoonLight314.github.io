---
title: "구글 BERT의 정석 ( Getting Started With Google BERT )"
date: 2021-10-29 08:26:28 -0400
categories: Deep Learning
---
# 구글 BERT의 정석 ( Getting Started With Google BERT )

### 한빛미디어 '나는 리뷰어다' 활동을 위해서 책을 제공받아 작성된 서평입니다.

<br>
<br>

<p align="center">
  <img src="/assets/Book_Review_Assets/Getting_Started_With_Google_BERT.png">
</p>

### 0. 소개

<br>

2012년에 CNN 구조를 사용한 AlexNet이 ImageNet에서 지난 대회 우승 Model보다 압도적인 성능으로 우승을 차지하면서 Deep Learning에 대한 관심은 비약적으로 높아졌습니다.

<br>

그 이후로 Image 분야에 Deep Learning을 응용하는 분야는 비교적(?) 접근이 쉽고 이해하기 쉬운 CNN구조를 바탕으로 널리 보급되고 누구나 쉽게 사용할 수 있었습니다.

<br>

하지만, NLP(Natural Language Processing)분야는 상대적으로 접근이 어려웠고 실제 업무 / 제품에 응요하기가 용이하지 않다는 분위기가 팽배했습니다.

<br>

NLP 초창기에는 다양한 기법(말뭉치 , Wordvec, Word Embedding , RNN , LSTM...)들에 대한 기본적인 상식이 풍부해야 이 분야에 대해서 어느 정도 이해를 할 수 있었습니다.

<br>

하지만, Image 분야에 혁신을 이끌어준 CNN이 있었던것과 같이, 2018년에는 NLP 분야에 혁신을 이끌어준 BERT(Bidirectional Encoder Representations from Transformers)가 Google에 의해서 세상에 나오게 됩니다.

<br>

이후 BERT는 NLP 분야의 모든 영역에서 기존 Model의 성능을 압도해 버리며, 다양한 파생 분야에서도 훌륭한 성능을 나타나게 됩니다.

<br>

BERT는 한마디로 Transformer 구조를 Encoding에 적용한 Language Model입니다.
( Transformer 구조를 Decoding 분야에 적용한 Language Model이 GPT이죠)

<br>

검색해 보면 매우 다양한 BERT 파생 Model에 대한 정보가 있지만, 각 분야에 어떤 방식으로 응용이 되었고 실제로 어떻게 사용하는지에 대해서는 일목 요연하게 Code와 함께 설명해 놓은 책이 사실상 없었습니다.

<br>

이 책은 BERT의 기본인 Transformer의 기본적인 Architecture부터 시작해서 BERT가 사전 학습되는 방법과 더불어 BERT를 Fine-Tuning하여 다양한 Task에 활용하는 방법까지 설명합니다.

<br>

또한, 매우 다양한 BERT 변형 Model에 대해서 설명하고, 마지막으로 한국어 BERT의 소개 및 활용 방안까지 다루며 마무리 됩니다.

<br>

BERT를 이용한 연구 분야와 활용 분야를 하나의 책을 엮은 유일한 전문서적이며, NLP에 관심이 있는 그 누구나 쉽게 BERT를 활용해 NLP 연구를 시작할 수 있는 디딤돌 역할을 할 것이라고 생각합니다.

<br>
<br>
<br>

### 1. 구성

<br>
<br>

**1장**

BERT의 기본인 Transformer에 대해서 자세히 설명합니다.
Transformer가 어떻게 동작하는지에 대해서 구체적으로 설명하고 기존 방식에 비해서 어떻게 다르며 어떤 점이 향상되었는지 살펴봅니다.

<br>

**2장**

BERT의 구조를 알아보았으니, 이제는 BERT를 어떻게 Train 시키는지에 대해서 알아봅니다.
어떤 Dataset을 사용하였는지, MLM(Masked Language Modeling) / NSP(Next Sentence Prediction)이 실제로 BERT를 학습시키는 방법에 대해서 구체적으로 알게됩니다.
그리고, NLP Model에 많이 사용되는 몇 가지 Tokenization 방식들에 대해서도 살펴봅니다.

<br>

**3장**

이번 장에서는 사전 학습된 BERT를 가지고 실제로 어떤 분야에 어떻게 사용하는지에 대해서 알아봅니다.
기본적으로 Sentence의 Embedding을 추출하는 방법, Quesntion-Answer, Text Classification 등을 어떻게 수행하는지에 대해서 알아봅니다.

<br>

**4장**

BERT는 매우 다양한 파생 Model이 존재하는데, 그 중에서도 많이 사용되는 몇 가지 파생 Model(ALBERT / RoBERTa , ELECTRA , SpanBERT)에 대해서 알아봅니다.
BERT와 어떤 점이 달라졌으며 어떤 용도에 주로 사용되는지 등에 대해서 다루게 됩니다.

<br>

**5장**

지식 증류(Distillation) 기반의 BERT Model인 DistilBERT / TinyBERT에 대해서 다룹니다.
Distillation 기법은 한 마디로 Model의 압축입니다. 덩치가 큰 BERT Model을 거의 같은 성능을 내면서도 작고 빠른 Model로 변환하는 방법에 대해서 알아봅니다.

<br>

**6장**

text summarization에 BERT를 사용해 봅니다.
참고로 text summarization에는 크게 extractive 방식과 abstractive 방식이 있는데 이에 대해서도 알아봅니다.

<br>

**7장**

multilingual BERT에 대해서 알아봅니다. BERT는 영어뿐만 아니라 매우 다양한 언어 Model도 지원해 주는데 이를 이용하는 방법에 대해서 알아봅니다.
유감스럽게도 한국어도 multilingual BERT에 포함되어 있지만, 성능이 썩 좋지는 않다고 하네요.

<br>

**8장**
문장 표현에 사용되는 Sentence BERT와 특정 Domain에 특화된 ClinicalBERT / BioBERT도 알아봅니다.

<br>

**9장**

VideoBERT , BART등과 같은 흥미로운 주제들에 대해서 알아봅니다.
이런 Model들이 어떤 일을 할 수 있는지 알아보는 것보다 개인적으로 이런 Model들을 Train 시키는 방법이 더욱 흥미로웠습니다.

<br>

**10장**

마지막으로, 앞에서 잠깐 말씀렸듯이 multilingual BERT에 포함된 한국어 BERT는 성능이 좋지 않습니다.
하지만, SK에서 공개한 한국어 언어 Model BERT인 KoBERT , KoGPT2 , KoBART등에 대해서 알아보고 실제 사용법에 대해서도 알아봅니다.

<br>
<br>


### 1. 대상독자

<br>

Deep Learning에 대한 기본적인 이해도 했고, Image 분류 등과 같은 Image에 Deep Learning을 적용하는 다양한 시도도 해 본 상태에서, 이제 자연어 처리도 한 번 해 보고는 싶은데 NLP는 뭔가 Image와 많이 달라서 고통받고 있는 분들에게 이 책을 권하고 싶습니다.

<br>

물론 NLP , Transformer에 대해서 이해하기 위해서는 NLP 분야에 대한 기본적인 지식은 가지고 있어야 합니다.

<br>

주위에서 현재 NLP의 SOTA는 BERT라고들 하지만 처음부터 무지막지한 수식 및 어려운 개념들에 치여서 포기해버린 기억이 있거나 혹은 당장 NLP 관련 업무를 진행해야 하는데 뭐부터 해야 할지 막막한 분들에게도 좋은 출발점이 될 수 있을거라고 믿습니다.

<br>

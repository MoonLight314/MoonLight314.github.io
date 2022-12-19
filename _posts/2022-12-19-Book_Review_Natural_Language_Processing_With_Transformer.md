---
title: "트랜스포머를 활용한 자연어 처리"
date: 2022-12-15 08:26:28 -0400
categories: Deep Learning
---
# 트랜스포머를 활용한 자연어 처리

### 한빛미디어 '나는 리뷰어다' 활동을 위해서 책을 제공받아 작성된 서평입니다.

<br>
<br>

<p align="center">
  <img src="/assets/Book_Review_Assets/Natural_Language_Processing_With_Transformer.png">
</p>

<br>
<br>

## 0. 소개 ##

<br>

Image 분야의 Deep Learning에 CNN 구조가 있다면, Text 분야의 Deep Learning에서의 대세는 역시 Transformer입니다.

<br>

2017년에 Google에 의해 'Attention Is All You Need' Paper가 발표되고 난 후, NLP 분야는 Transformer로 인해 그야말로 비약적인 발전을 이루게 됩니다.

<br>

Paper 발표 이후, Text Classification , Summarization , Translation , QA, NLU , NER 등 모든 Text 관련 부분에서 SOTA를 이루어 냈고, 지금도 그 위력은 여전합니다.

<br>

Transformer 구조 자체가 약간 어려운 느낌은 있지만, 대중화시킨 것은 HuggingFace의 역할이 큽니다.

<br>

누구나 Transformer 구조를 Base로 한 Pre-Trained Model을 검색하고 다운로드할 수 있으며,

<br>

Fine Tuning 하여 자신의 Project에 적용할 수 있도록 도와줍니다.

<br>

이 책은 Transformer 개발자와 HuggingFace 개발자들이 공동 집필, 이를 둘러싼 생태계까지, 전 영역을 다룹니다.

<br>

모든 주제가 실습 위주 방식이고, 주피터 노트북을 제공하기 때문에 직관적이며 이해하기 쉽습니다.

<br>

또한, 통찰력 있는 설명과 실질적 애플리케이션에 대해서 다루기 때문에 매우 실용적입니다.

<br>

Transformer를 기초부터 설명하고 실전에 사용할 수 있도록 Guide를 제시하고 있기 때문에, NLP를 처음 배우는 사람이든 전문가이든 Model을 빠르게 개발하고 배포할 수 있도록 도와줍니다.

<br>

책에는 실전에서 만날 수 있는 난관(처음부터 Model을 훈련시켜야 한다든지 Label이 있는 Dataset이 없다든지...)에 대응할 수 있는 실용적이고 즉시 사용 가능한 Solution도 제공합니다.

<br>

번역가(박해선 님)의 전문성에서 우러나오는 물 흐르는 듯한 부드러운 문장이 읽기 편하고 이해가 쉽다는 것도 장점입니다.

<br>

이 책은 현재 구할 수 있는 Transformer의 Guide 중 최고라고 생각합니다.

<br>
<br>

## 1. 구성 ##

<br>

**CHAPTER 1 트랜스포머 소개**

<br>

**1.1 인코더-디코더 프레임워크**

- 흔히 Seq-To-Seq 방식이라고도 알려진 Encoder - Decoder Framework에 대한 소개
- 이 방식들이 안고 있는 고질적인 문제점 소개

<br>

**1.2 어텐션 메커니즘**

- Attention의 장점과 Attention이 어떤 방식으로 기존 방식들의 문제점들을 개선하는지 소개
- Transformer : Self-Attention 방식 소개

<br>

**1.3 NLP의 전이 학습**

- Image 분야에서는 전이학습이 흔한 일
- NLP 분야에서 전이학습을 어떻게 정의하고 적용하는가에 대한 고찰

<br>

**1.4 허깅페이스 트랜스포머스**

- HuggingFace 소개 및 장점

<br>

**1.5 트랜스포머 애플리케이션 둘러보기**

- Transformer를 사용한 NLP 작업의 종류

<br>

**1.6 허깅페이스 생태계**

- HuggingFace는 어떻게 구성되어 있는지에 대해서 알아본다.
- Hub , Tokenizer , Dataset , Accelerator

<br>

**1.7 트랜스포머의 주요 도전 과제**

- 앞으로 Transformer가 풀어야할 주요 과제들에 대한 소개

<br>

**CHAPTER 2 텍스트 분류**
- DistilBERT를 이용한 텍스트 분류 작업을 해보면서 HuggingFace 사용법을 익힌다.

<br>

**2.1 데이터셋**

- HuggingFace에서 Dataset을 검색하고 다운로드 / Pre-Processing / EDA하는 방법 소개

<br>

**2.2 텍스트에서 토큰으로**

- Tokenizer 사용법 , 부분단어(Sub Word) 토큰화
- 특정 Model의 Tokenizer Load 하기

<br>

**2.3 텍스트 분류 모델 훈련하기**

- Feature Extractor로 사용하기
- Pre-Trained Model 사용법
- Fine Tuning 방법

<br>

**2.4 결론**

- 실제 일어날 수 있는 다양한 상황에 대한 의견

<br>

**CHAPTER 3 트랜스포머 파헤치기**

- Transformer의 실질적인 구현을 알아본다.

<br>

**3.1 트랜스포머 아키텍처**

- Encoder - Decoder 기반 구조
- Transformer Model의 구분 : Encoder 유형 / Decoder 유형 / Encoder - Decoder 유형

<br>

**3.2 인코더**

- Encoder 구조를 분석
- 실제 Torch / Tensorflow 구현 소개

<br>

**3.3 디코더**

- Decoder 구현 분석 및 Torch 구현

<br>

**3.4 트랜스포머 유니버스**

- 다양한 Transformer Base Model들의 분류 및 소개

<br>

**CHAPTER 4 다중 언어 개체명 인식**
- 영어 이외의 다른 언어에서 Transformer 적용 방법을 NER을 통해서 알아본다.

<br>

**4.1 데이터셋**

- NER에 사용할 Dataset 소개

<br>

**4.2 다중 언어 트랜스포머**

- 다중 언어 NER에 사용할 Dataset 소개

<br>

**4.3 XLM-R 토큰화**

- NER에 사용할 Tokenizer 소개
- 단계별 Tokenizer가 하는 일 소개

<br>

**4.4 개체명 인식을 위한 트랜스포머**

- Transformer가 어떻게 Token을 인식하는지 확인

<br>

**4.5 트랜스포머 모델 클래스**

- Transformer를 임의의 NLP 작업에 적용해 보기
- Body - Head 구조로 인해서 쉽게 어디든지 적용가능하다는 것을 확인

<br>

**4.6 NER 작업을 위해 텍스트 토큰화하기**

- Tokenizer 과정 소개

<br>

**4.7 성능 측정**

- NER의 성능을 측정하기 위한 seqeval 제작

<br>

**4.8 XLM-RoBERTa 미세 튜닝하기**

- Aug.를 설정하고 Train 하는 과정

<br>

**4.9 오류 분석**

- 오류를 분석하고 몇 가지 Tip과 검증 방법 확인

<br>

**4.10 교차 언어 전이**

- 하나의 언어에서 학습한 Model을 다른 언어에도 적용해보는 방법을 알아보자

<br>

**4.11 모델 위젯 사용하기**

<br>

**CHAPTER 5 텍스트 생성**

GPT-2를 이용한 Text Generation 예제를 살펴본다.

<br>

**5.1 일관성 있는 텍스트 생성의 어려움**

- GPT가 입력 Seq.을 보고 다른 Seq.에 등장하는 확률을 예측하도록 Train
- 충분한 학습 Data 얻기가 불가능하므로 조건부 확률을 이용한다.

<br>

**5.2 그리디 서치 디코딩**

- Greedy Search 방식의 작동 방법과 문제점 그리고 그 대안과 Beam Search Decoding

<br>

**5.3 빔 서치 디코딩**

- Beam Search Decoding에 대해서 알아본다.

<br>

**5.4 샘플링 방법**

<br>

**5.5 탑-k 및 뉴클리어스 샘플링**

- 탑-k 및 뉴클리어스 샘플링에 대한 설명

<br>

**5.6 어떤 디코딩 방법이 최선일까요?**

- Decoding 방법과 선택에 대한 가이드

<br>

**CHAPTER 6 요약**

- Seq. To Seq. Model(Encoder - Decoder)과 CNN/DailyMail Dataset을 이용하여 문장을 요약하는 방법을 살펴보자

<br>

**6.1 CNN/DailyMail 데이터셋**

- CNN/DailyMail Dataset에 대해서 알아보자

<br>

**6.2 텍스트 요약 파이프라인**

- Data Pre-Processing 방법
- Baseline Model 선정 및 Model을 이용하는 방법 ( BART / GPT-2 / T5 / PEGASUS )

<br>

**6.3 요약 결과 비교하기**

- 4개 Model 별로 결과 비교

<br>

**6.4 생성된 텍스트 품질 평가하기**

- 다양한 Text Summarization 품질 평가 / 방법 소개 ( BLEU / ROUGE )

<br>

**6.5 CNN/DailyMail 데이터셋에서 PEGASUS 평가하기**

- PEGASUS BLEU 점수확인 하기

<br>

**6.6 요약 모델 훈련하기**

- 직접 Model을 특정 Dataset에서 훈련하는 과정
- SAMSum( 삼성이 만든 Dataset) 소개
- PEGASUS로 SAMSum 평가하기
- PEGASUS의 Fine Tuning 방법 소개
- 실제 대화를 요약해 보기

<br>

**CHAPTER 7 질문 답변**

- Text 일부를 답변으로 추출하는 방법. 
- 문서(Web Page , 계약서 , News)에서 원하는 답을 찾는 방법 확인

<br>

**7.1 리뷰 기반 QA 시스템 구축하기**

- 수많은 Review 중 특정 질문에 대한 답을 찾는 방법 소개
- SubjQA 소개 및 특징 확인 & 사용법
- QA를 NLP 문제로 해결하기 위한 방안 모색 ( 문장에서 어떻게 답변 부분을 골라내게 하느냐 )
- Base Model을 어떤 Model로 선택할지 Guide
- Tokenizer for QA Model
- 긴 문장 처리 방법
- 질문만 있는 Dataset 처리 방법

<br>

**7.2 QA 파이프라인 개선하기**

- 리트리버 / 리더 평가 방법
- 특정 Domain에 적용하기

<br>

**7.3 추출적 QA를 넘어서**

- 추상적(Abstractive) or 생성적(Generative) QA 소개
- RAG 소개

<br>

**CHAPTER 8 효율적인 트랜스포머 구축**

- Transformer 성능 향상을 위한 기법들 소개

<br>

**8.1 의도 탐지 예제**

- 질문의 의도를 파악하는 Model

<br>

**8.2 벤치마크 클래스 만들기**

- Model 배포시에 염두해야할 사항들 ( 성능 , Latency , Memory 등 )을 확인할 수 있는 방법

<br>

**8.3 지식 정제로 모델 크기 줄이기**

- 지식 정제 : 느리고 크지만 성능이 더 좋은 Teacher의 동작을 모방하도록 작은 Student Model을 훈련하는 전반적인 방법 소개

<br>

**8.4 양자화로 모델 속도 높이기**

- Weight & Output Format으로 Int로 바꾸는 양자화를 통해서 속도를 비약적으로 빠르게 만드는 방법을 소개

<br>

**8.5 양자화된 모델의 벤치마크 수행하기**

- 성능 향상 확인

<br>

**8.6 ONNX와 ONNX 런타임으로 추론 최적화하기**

- ONNX로 성능 향상 시키기

<br>

**8.7 가중치 가지치기로 희소한 모델 만들기**

- Memory가 부족한 경우에 속도 향상을 이룰 수 있는 방법
- 중요하지 않는 가중치를 줄이는 방법에 대해서 알아본다.

<br>

**CHAPTER 9 레이블 부족 문제 다루기**

- Labeled Data가 없거나 매우 적은 경우에 대한 Guide

<br>

**9.1 깃허브 이슈 태거 만들기**

- Github Issue를 Dataset으로 삼아서 지도학습(Multi Label Text Classification) 문제로 만들 수 있다.
- Data Pre-Processing / Dataset 나누기 등의 준비 작업을 한다.

<br>

**9.2 나이브 베이즈 모델 만들기**

- Baseline Model로 Naive Bayes을 준비한다.
- 이를 훈련시키고 성능을 알아본다.

<br>

**9.3 레이블링된 데이터가 없는 경우**

- Labeled Data가 없는 경우 처리 방법 소개

<br>

**9.4 레이블링된 데이터가 적은 경우**

- Data Augmentation 방법 소개

<br>

**9.5 레이블링되지 않은 데이터 활용하기**

- Labeling 되지 않는 Data를 활용하는 방법 소개

<br>

**CHAPTER 10 대규모 데이터셋 수집하기**

- 원하는 Data가 모두 있는 경우에 어떤 일을 할 수 있는지 알아보자

<br>

**10.1 대규모 데이터셋 수집하기**

- 사전 훈련을 위해서 대규모 말뭉치부처 만드는 다양한 방법들을 소개한다.
- 대용량 Dataset을 만드는 방법을 소개한다.

<br>

**10.2 토크나이저 구축하기**

- Customized Tokenizer를 구축하는 방법 , 성능 측정 방법 , 훈련 방법 소개

<br>

**10.3 밑바닥부터 모델을 훈련하기**

- 목적에 맞게 모델을 훈련시키는 방법 찾기
- Model 초기화 / Data Loader 구축하기 / Train Arguments 설정 방법

<br>

**10.4 결과 및 분석**

- 결과 확인

<br>

**CHAPTER 11 향후 방향**

- Transformer가 강력하지만 현재 당면한 과제들이 어떤 것이 있는지 알아보고 연구동향을 살펴보자

<br>

**11.1 트랜스포머 확장**

- 기존의 성능이 우수한 Model들을 성능을 향상시키는 방향으로 발전시키는 방안에 대해서 모색한다.
( 규모의 면 : GPU , Dataset의 크기 등
  Model Algorithm 자체 : Attention 자체를 향상시키는 방법 )

<br>

**11.2 텍스트를 넘어서**

- Text Data 뿐만 아니라, Image , Table Data 등과 같은 Data를 활용하는 방안

<br>

**11.3 멀티모달 트랜스포머**

- 일반적인 Text Data 뿐만 아니라 Audio Text , Vision Text 등과 같이 두 개의 Data 형식을 합치는 방안에 대한 검토

<br>

**11.4 다음 목적지는?**

- Transformer를 활용한 자신만의 Tech. / Project / 글을 작성해 보자

<br>
<br>

## 2. 대상 독자 ##

<br>

저는 이 책을 Transformer라는 Model은 알고 있지만, 이 Model을 실제 Application에 적용할 자세한 Guide가 필요한 개발자 혹은

<br>

어느 정도 Deep Learning Framework에 경험이 있지만, 자신만의 Project에 Transformer를 적용하려고 하는 개발자들에게 추천합니다.

<br>

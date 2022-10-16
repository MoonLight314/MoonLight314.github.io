---
title: "소프트웨어 아키첵처 The Hard Parts"
date: 2022-10-16 08:26:28 -0400
categories: Deep Learning
---
# 소프트웨어 아키첵처 The Hard Parts

### 한빛미디어 '나는 리뷰어다' 활동을 위해서 책을 제공받아 작성된 서평입니다.

<br>
<br>

<p align="center">
  <img src="/assets/Book_Review_Assets/Software Architect - The Hard Part.png">
</p>

<br>

## 0. 소개 ##

<br>

Software Architect는 끊임없는 결정의 연속입니다.  

<br>

복잡하고 다양한 구성 요소들의 상충관계를 파악하고 그에 맞는 의사 결정을 내려야 하는 선택들만이 남아 있는 'Hard Parts'입니다.

<br>

이 책은 분산 Architecture를 구성할 때 Architect가 각 결정이 가지고 Tradeoff를 분석하여 객관적으로 의사 결정을 내리는 모든 과정에 대한 Know-How가 녹아있습니다.

<br>

Architecture 결정에 필요한 지식 , 각 Architecture 장단점 , 다양한 Pattern 설명 , 큰 그림을 보는 안목 등을 구체적인 사례를 통해서 공유합니다.

<br>

또한, 최신 Software Architecture의 다양한 난제들에 대한 결정을 어렵게 만드는 이유에 대해서 분석과 소개를 하고 자신이 처한 문제에 대해서 적용하는 방법을 소개해 줍니다.

<br>

모든 방법에 통용되는 만능 Solution이라는 것은 세상에 존재하지 않기 때문에 각 방법론과 Pattern의 장단점을 모아놓았습니다.

<br>
<br>

주요하게 다루는 내용으로는 Tradeoff 분석을 하여 의사 결정을 내리기 위한 자료를 효과적으로 정리하는 방법, Service를 필요에 따라 다양하게 세분화하여 결과적으로 더 나은 결정을 내리는 방법론, Service들 사이의 계약 관리 및 각 Service를 어떻게 분리할 것인가에 대한 고찰 , 복잡하게 분산된 Architecture 구조에서 Data를 관리하고 처리하는 방법론 , Application을 쪼갤 때 Workflow와 Transaction을 관리하는 Pattern들의 방법들에 대해서 알아봅니다.

<br>

이러한 내용들을 가상의 팀인 '한빛가이버 사가'의 팀원들이 좌충우돌하는 이야기와 함께 풀어나가고 있어서 자칫 지겨울 수 있는 이야기를 재미있게 풀어나갑니다.

<br>
<br>
<br>

## 1. 구성 ##

<br>

## chapter 1 ‘Best Practice’가 없다면? ##
- 모든 Architecture 문제에서 하나의 만병통치약같은 Solution은 없으며, 제목이 Hard Part인 이유에 대해서 알아보고, Data의 중요성 , Architecture 결정 Record , Architecture와 설계의 차이와 같은 기본적인 개념들을 설명합니다.

<br>

## PART 1 따로 떼어놓기 ##
- 이 책의 목표는 분산 Architecture에서 Tradeoff를 분석하는 것이므로 Architecture의 각 부분을 따로 떼어놓는 것부터 시작합니다. 그 후에 각 요소가 정적으로 결합하는 방식에 대한 설명을 중점적으로 합니다.

<br>

## chapter 2 Architecture 퀀텀 ##
- Architecture의 정적 커플링 & 동적 커플리의 범위 정의 문제에 대해서 다룬다

<br>

## chapter 3 Architecture 모듈성 ##
- Architecture 모듈성이 무엇인지를 정의하고 실제 분해 프로세스를 시작합니다.

<br>

## chapter 4 Architecture 분해 ##
- Codebase를 평가하고 해제하는 도구에 대한 설명을 합니다.

<br>

## chapter 5 컴포넌트 기반 분해 Pattern ##
- Architecture 분해 프로세스에서 사용할 수 있는 유용한 다양한 Pattern에 대한 설명

<br>

## chapter 6 운영 Data 분리 ##
- Service와 Data를 조정하는 방법 등을 사용해 보면서, Data가 Architecture에 어떤 영향을 끼치는지 확인해 봅니다.

<br>

## chapter 7 Service 세분도 ##
- Architecture 커플링과 Data 관심사를 통합해서 통합인과 분해인에 대해서 이야기 합니다.

<br>

## PART 2 다시 합치기 ##
- Service , 통신 , 계약 , 분석 Workflow, 분산 Transaction, Data Ownership, Data 액세스 , 분석 Data 관리 등과 같이 분산 Architecture에서 Hard Part를 어떻게 극복할지에 대한 다양한 예를 보여줍니다.
- PART 2의 핵심은 각 분산 개체들간의 '통신(Communication)'입니다.

<br>

## chapter 8 재사용 Pattern ##
- 분산된 Architecture에서 Code 재사용( Code , Library , Service )은 어떻게 적용해야 하고 어떠한 경우에 가치가 있는가에 대해서 다룹니다.

<br>

## chapter 9 Data 오너십과 분산 Transaction ##
- Data(DB)는 누가 어떤식으로 관리하고 어떤 식으로 사용하는지에 대한 다양한 기법들을 알아봅니다.

<br>

## chapter 10 분산 Data 액세스 ##
- 분산 시스템 상에서 Service 간에 Data 통신을 어떻게 할 것인가에 대해서 알아봅니다.

<br>

## chapter 11 분산 Workflow 관리 ##
- 분산 Architecture에서 Domain에 특정한 작업을 지시하고 그와 관련된 다양한 문제를 해결하기 위해 복수개의 Service를 조합하는 것을 말합니다.
이를 조정이라고 하는데, 조정 Pattern에는 오케스트레이션(orchestration)과 코레오그래피(choreography)이 있습니다. 이 2가지 Pattern에 대해서 알아봅니다.

<br>

## chapter 12 트랜잭셔널 사가 ##
- 이 Chapter에서는 Transactional Saga의 내부 작동원리와 관리 방법에 대해서 자세히 알아봅니다.

<br>

## chapter 13 계약 ##
- 계약이란 종류가 다른 Architecture간의 서로 어떻게 연결되는지 정의한 것으로 이 Chapter에서는 이러한 계약의 종류와 Tradeoff에 대해서 알아봅니다.

<br>

## chapter 14 분석 Data 관리 ##
- 운영 Data와 분석 Data를 어떻게 분리하여 관리할 것인가에 논의합니다.

<br>

## chapter 15 자신만의 Tradeoff 분석 ##
- 일반적인 상황에서도 적용할 수 있는 Tradeoff 분석 절차와 각 단계마다 적용할 수 있는 기법들에 대해서 알아봅니다.

<br>
<br>
<br>

## 2. 대상 독자 ##

<br>

Software Architecture에 관해서는 아무리 구글링을 해봐도 정해진 답이 없는 당연합니다. 각각의 개별 Case에 대해서 모두 다른 방법론을 적용하고 Tradeoff 분석을 해야 하기 때문입니다.

<br>

이 책은 Software Architecture에 관심을 가진 모든 이들에게 필독서라고 생각합니다.

<br>

Architecture에 관심이 많은 기술자. 초보 Architect , 리더급 Architect , 분산 시스템을 구성해야하는 Architect, 누구든지 이 책에서 훌륭한 영감을 받을 수 있다고 생각합니다.

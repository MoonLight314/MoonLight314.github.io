---
title: "고객을 끌어오는 구글 애널리틱스4"
date: 2021-10-29 08:26:28 -0400
categories: Deep Learning
---
# 고객을 끌어오는 구글 애널리틱스4

### 한빛미디어 '나는 리뷰어다' 활동을 위해서 책을 제공받아 작성된 서평입니다.

<br>
<br>

<p align="center">
  <img src="/assets/Book_Review_Assets/Business_Data_Science.png">
</p>

## 0. 소개 ##

<br>

Digital Marketing 영역에서 고객의 Data를 분석해서 다양한 Insight를 뽑아내고 이를 Business 전략에 반영하는 Data 분석 능력은 Marketer 뿐만 아니라 Data Scientist가 
갖추어여할 필수 역량이 된지 오래입니다.

<br>

다양한 고객의 Data 중에서 가장 기본이 되는 고객 Data가 Web Log 분석인데, 많은 기업에서 고객 Web Log 분석을 중요하게 생각하는 이유가 Website가 Digital Marketing 활동의 거점 역할을 한다는 점과 쉬운 설치만으로 방대한 양의 고객 Data 수집이 가능하다는 이유입니다.

<br>

많은 기업들이 Web Log 분석을 실시하고 있다고 생각되며, 일부 업체에서는 유료 서비스를 이용하기도 하겠지만, 이 책에서 소개하고 있는 Google Analytics를 사용하고 있다고 여겨집니다.

<br>

**0.1. Google Analytics란?**

<br>

Google Analytics는 Web Site 방문자의 Data를 수집해서 분석함으로써 Online Business의 성과를 측정하고 개선하는 데 사용하는 Web Log 분석 도구입니다.

Google은 2005년 3월 Web 분석 전문 업체인 어친 소프트웨어(Urchin Software)를 인수한 후 그해 11월 Google Analytics Service를 출시했습니다. 무료 서비스임에도 매우 강력한 기능을 제공함으로써 지금은 전세계적으로 가장 널리 사용되는 Web 분석 툴이 되었습니다.

Google Analytics는 다른 Web Log 분석 Service에 비해서 다음과 같은 장점을 가지고 있습니다.

<br>

**1) 무료**
   - Google Analytics의 가장 큰 장점입니다. Google 계정만 있으면 누구나 무료로 사용할 수 있습니다.

<br>

**2) 강력한 기능**
   - Google Analytics는 무료임에도 유료 서비스 못지않은 다양한 디지털 분석 기능을 제공합니다.
     * 다양한 Standard Reports Template 제공
     * Standard Reports Template뿐만 아니라, Custome Report Template 작성 기능도 제공하며, 작성하고자 하는 항목을 선택하여 Report를 작성할 수 있습니다.
     * Segment 분석 기능을 이용하여 심층적인 Data 분석이 가능합니다.
     * Google에서 제공하는 Cloud Infra와 연동하여 빠른 Data 처리가 가능합니다.

<br>

**3) 우수한 UI**
   - 간결하고 직관적인 UI를 사용하고 있어서 사용하기 쉽고, 표 뿐만 아니라 다양한 그래프, 차트 등 많은 시각화 방식을 제공해 주어 Data 이해를 쉽게 해줍니다.

<br>

**4) 확장성 / 통합성**
   - Google에서 제공하는 디지털 광고 솔루션인 Google AdWords, Google Tag Manager, Google Data Studio, Google Optimize 등 디지털 분석시 함께 활용하면 좋은 다른 구글 도구들과 쉽게 연동이 가능합니다.

<br>

**5) 지속적인 기능 추가**
   - 초반 부실한 메뉴 화면에 비해 점점 기능이 추가되고 기능이 Update되고 있습니다.

<br>

이 책은 Google Analytics을 Coding 지식없이 실제 예제를 따라해 보면서 익힐 수 있도록 구성되어 있습니다. 

<br>

쇼핑몰 예제를 사용하여 고객 분석을 수행할 수 있도록 구성되어 있습니다. 고객이 어떻게 사이트에 접속하는지, 어느 부분을 클릭했고 무엇을 입력하고 어떻게 사이트를 탐색하는지에
대한 분석을 Google Analytics로 직접해 보면서 익힐 수 있도록 구성되어 있습니다.

<br>
<br>

## 1. 구성 ##

<br>

**Chapter 00 시작하기 전에**
<br>
Google Analytics를 시작하기 전에 Data 분석이란 무엇이며, Google Analytics은 무엇이며 누구에게 필요한 것인지 간단히 알아봅니다.

<br>

**Chapter 01 구글 애널리틱스 시작하기**
<br>
Google Analytics 실습을 하기 전에 Google Analytics를 사용하기 위한 환경을 만듭니다.

<br>

**Chapter 02 보고서 조작 방법 따라 배우기**
<br>
기본적인 보고서 사용방법을 익힙니다.

<br>

**Chapter 03 기본 분석 따라 배우기**
<br>
우리에게 필요한 보고서가 무엇인지 확인해 보고 Google Analytics가 생성한 보고서를 어떻게 해석할지에 대해서 알아봅니다.

<br>

**Chapter 04 데이터 더 상세하게 파악하기**
<br>
Google Analytics가 생성한 보고서에서 비교군 / 잠재고객 / Segment에 대해서 알아봅니다.

<br>

**Chapter 05 이벤트란 무엇인가?**
<br>
Google Analytics에서 말하는 'Event'란 무엇인지에 대해서 자세하게 알아봅니다.

<br>

**Chapter 06 이벤트 분석 따라 배우기**
<br>
앞에서 알아본 'Event'에 대해서 좀 더 자세히 살펴봅니다. 직접 수집 Event를 설정하는 방법에 대해서도 알아봅니다.

<br>

**Chapter 07 탐색 분석 따라 배우기**
<br>
Ch. 00~06까지의 내용을 정리하고, 탐색 분석에 대해서 알아봅니다.

<br>

**Chapter 08 획득 보고서 따라 배우기**
<br>
사용자가 사이트에 접속하는 방법에 대한 분석 방법을 알아봅니다.

<br>

**Chapter 09 캠페인 링크 분석 따라 배우기**
<br>
사용자가 캠페인 링크로 들어온 경로에 대한 분석에 대해서 알아봅니다.

<br>

**Chapter 10 기술과 인구통계 따라 배우기**
<br>
기술과 인구통계 분석 방법을 알아봅니다.

<br>

**Chapter 11 사용자 속성 분석 따라 배우기**
<br>
사용자의 속성(성별 / 연령 / 관심사 등) 정보를 분석할 수 있는 방법에 대해서 알아봅니다.

<br>

**Chapter 12 잠재고객 따라 배우기**
<br>
Ch. 08~11까지의 내용을 정리하고, 잠재고객을 만드는 방법을 알아봅니다.

<br>

**Chapter 13 사용자 ID 설정하기**
<br>
특정 사용자를 한정해서 Event를 분석하는 방법을 알아봅니다.

<br>

**Chapter 14 맞춤 이벤트 따라 배우기**
<br>
'사전 정의 Event'와 '사용자 정의 Event'에 대해서 알아보고 사용하는 방법에 대해서 배웁니다.

<br>

**Chapter 15 잠재고객과 잠재고객 이벤트 트리거 따라 배우기**
<br>
앞서 배운 잠재 고객 따라 배우기 보다 좀 더 고도화된 잠재 고객 이벤트 트리거에 대해서 알아봅니다.

<br>

**Chapter 16 전자상거래 이벤트 수집하기**
<br>
온라인 전자상거래 상에서 이루어지는 다양한 거래 형태에 대해서 분석하는 방법에 대해서 배웁니다.

<br>

**Chapter 17 사용자 행동 순서 분석 따라 배우기**
<br>
시용자가 어떤 순서로 사이트에서 활동하는지에 대한 Log를 수집하고 분석하는 방법에 대해서 알아봅니다.

<br>

**Chapter 18 전자상거래 잠재고객 활용하기**
<br>
잠재 고객을 특징 짓는 방법과 잠재 고객을 어떤 식으로 활용할 수 있는지에 대해서 학습합니다.

<br>

**Chapter 19 앱 데이터 분석하기**
<br>
'App.' 데이터 분석 예시를 살펴보고, 'App.'에서 발생하는 몇 가지 중요한 Event를 배우고 활용해 보도록 하겠습니다.

<br>

**Chapter 20 파이어베이스로 사용 참여 유도하기**
<br>
Firebase는 Google에서 제공하는 앱 개발 도구로써, Firebase Analytics, Firebase Cloud Messaging 등을 활용하는 방법에 대해서 알아봅니다.

<br>
<br>

## 2. 대상 독자 ##

<br>

제가 생각하는 이 책은 가장 큰 장점 중의 하나는 Google Analytics를 사용해 본 적 없거나 Coding 능력이 부족한 독자들도 차근차근 따라해 보면서 잘 익힐 수 있도록 설명이 잘 되어 있다는 점입니다.

<br>

이러한 책의 특징 때문인지도 몰라도 Google Analytics를 처음 배우는 독자들이 쉽게 읽을 수 있다고 생각합니다.

<br>

Google Analytics의 목적이 고객 분석을 잘 해서 매출을 올리는 것이므로 상품 구매를 더 많이 유도하고 싶은 마케터들이나 사용자 친화적인 홈페이지를 만들고 싶은 기획자, 디자이너들에게도 좋은 가이드가 될 것입니다.

<br>

또한, 고객을 좀 더 잘 분석하여 좀 더 효율적인 결정을 하고 싶어하는 관리자들에게도 좋은 조언자가 될 것입니다.

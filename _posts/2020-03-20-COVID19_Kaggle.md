---
title: "Kaggle Competition Related COVID19"
date: 2020-03-20 08:26:28 -0400
categories: Kaggle COVID19
---

# COVID19 & Pandemic
코로나 19로 인해 WHO가 Pandemic을 선언한 가운데, 최근 Kaggle에서 코로나 관련 Dataset과 Competition이 등장하고 있어
이번 Post에서 소개해 볼까 합니다.
<br>
<br>
<br>
<br>

# Dataset

먼저 Kaggle에 등록된 코로나 19 관련 Dataset들을 살펴보도록 하겠습니다.

[https://www.kaggle.com/datasets?search=covid19&sort=usability](https://www.kaggle.com/datasets?search=covid19&sort=usability)

<p align="center">
  <img src="/assets/kaggle_COVID19/COVID_19_00.png">
</p>
<br>
<br>
<br>
<br>

매우 다양한 Dataset이 공개되어 있습니다. 
또한, 대한민국에서 공개한 Dataset도 꽤 많이 눈이 띄네요.

그 중에서도 초기에 공개되어 인기를 끌었던, 한양대 대학원생 김지후님이 공개하신 Dataset도 있습니다. 

* 관련뉴스 : [https://n.news.naver.com/article/030/0002871312](https://n.news.naver.com/article/030/0002871312)
* Kaggle에 공개된 김지후님의 Dataset : [https://www.kaggle.com/kimjihoo/coronavirusdataset](https://www.kaggle.com/kimjihoo/coronavirusdataset)

국격 높아지는 소리가 들리네요~!

<br>
<br>
<br>
<br>
<br>
<br>
<br>
<br>

# Task
  
* 이번에는 코로나 19와 관련된 Kaggle Task를 알아보도록 하겠습니다.
* 관련 Dataset은 아래의 Link에서 확인하실 수 있습니다.
  - COVID-19 Open Research Dataset Challenge (CORD-19) : https://www.kaggle.com/allen-institute-for-ai/CORD-19-research-challenge

* 이 Dataset은 백악관과 몇몇의 연구 그룹에서 배포한 것입니다.
* 이 Dataset은 COVID-19, SARS-CoV-2 및 코로나 바이러스에 관한 44,000 개가 넘는 학술 자료를 제공하며, 그 중에 29,000 개는 전문으로 제공됩니다.
* Text 형태의 학술 자료이기 때문에 자연어 처리 혹은 기타 AI 기술을 이용하여 다양한 작업을 할 수 있을 것이라고 기대됩니다.
* 이 Datset은 주기적으로 Update되며, 최신 Dataset은 아래 Link에서 확인하시면 됩니다.
  - [https://pages.semanticscholar.org/coronavirus-research](https://pages.semanticscholar.org/coronavirus-research)
  
* 저는 3월 13일자 Dataset을 Download하였습니다. Directory 구조는 아래와 같았습니다.
<br>
<p align="center">
  <img src="/assets/kaggle_COVID19/COVID_19_01.png">
</p>
<br>

* 각 Folder는 아래와 같은 문서를 제공합니다.
  - Commercial use subset (includes PMC content)
  - Non-commercial use subset (includes PMC content)
  - Custom license subset
  - bioRxiv/medRxiv subset (pre-prints that are not peer reviewed)

* 개별 문서는 JSON 형태의 File로 제공되며, Schema는 다음과 같습니다.
  - [JSON Schema](https://ai2-semanticscholar-cord-19.s3-us-west-2.amazonaws.com/2020-03-13/json_schema.txt)

* CSV 형태로 Metadata가 제공되며, 개별 문서의 제목은 각 문서의 Hash값( SHA )값과 Match 됩니다.

<br>
<br>

## Task List
* 이 Dataset과 연결된 Task List는 아래 Link를 참고해 주시기 바랍니다.
  - [Task List](https://www.kaggle.com/allen-institute-for-ai/CORD-19-research-challenge/tasks)

* 다양한 Task 들이 있네요.
<br>
<p align="center">
  <img src="/assets/kaggle_COVID19/COVID_19_02.png">
</p>
<br>

* Task들의 제목들을 보아하니, 공개된 Dataset에서 적절한 답변을 찾는 Model을 만드는 것이 이 Task들의 목표 같습니다.

* 아무 Task 하나를 살표보겠습니다. 두번째 보이는 Task에 들어가 보겠습니다.
<br>
<p align="center">
  <img src="/assets/kaggle_COVID19/COVID_19_03.png">
</p>
<br>

* Dataset에서 코로나 19의 Risk Factor를 조사해 보라는 Task네요.

* 제출된 Submission 중 하나를 살펴보도록 하겠습니다.
https://www.kaggle.com/shiromiyuki/covid-19-risk-factors-using-tf-idf



# Competition
COVID-19 Global Forecasting Challenge : https://www.kaggle.com/c/covid19-global-forecasting-week-1/overview

COVID-19 California Forecasting Challenge : https://www.kaggle.com/c/covid19-local-us-ca-forecasting-week-1/overview


<br>
<br>
<br>
<br>
<br>
<br>
<br>
<br>


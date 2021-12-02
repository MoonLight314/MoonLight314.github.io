---
title: "Tensorflow_Certificate"
date: 2021-10-29 08:26:28 -0400
categories: Deep Learning
---
# Tensorflow_Certificate

<br>
<br>
<br>
<br>

<p align="center">
  <img src="/assets/Tensorflow_Certificate/pic_00.png">
</p>

<br>
<br>

## 0. About Tensorflow Certificate

<br>

* Tensorflow Certiciate는 Google에서 운영하는 Tensorflow Developer 인증 시험입니다.

<br>

* 저는 Tensorflow를 주로 사용하고 있는데 이왕 사용하는 김에 자격증은 없나 찾아보다 알게되었고 이번에 취득하게 되었습니다.

<br>

* Tensorflow Certiciate는 AI Framework중의 하나인 Tensorflow를 활용하여 다양한 Model을 구축하는 능력을 확인하는 시험입니다.

<br>

* 간단하게 Tensorflow Certificate에 대해서 알아보면, 시험 주최는 Google이며 개인적으로 응시합니다.

<br>

* 시험은 Online환경에서 PyCharm의 Plug-In을 이용해서 치뤄집니다. **( VS Code는 안됩니다. Only PyCharm )**

<br>

* 시간은 **5시간**이며 직접 치뤄본 경험상으로는 적절한 시간입니다.

<br>

* 문제는 **총 5문제**이며, **Tensorflow 2.x 으로 총 5개의 Model(.h5 형식)** 을 만들어서 제출하면 Model 성능을 평가해서 5점 만점 기준으로 실시간으로 점수를 알려줍니다.

<br>

* 이렇게 5개의 Model을 만들어서 제출해서 모두 Pass(5/5)를 받으면 합격하게 됩니다.

<br>

*  **Coursera에 이와 관련된 강좌**가 있으니 해당 과정을 수료한 후 시험을 치르면 됩니다. Coursera 강의와 매우 매우 유사한 형식으로 시험이 나옵니다.

<br>

* 응시료는 **100 US**이며, 결제 후 6개월이던가...1개월이던가..내에 언제든지 온라인으로 시험을 치르면 됩니다.

<br>
<br>
<br>
<br>
<br>

## 1. Environment

<br>

시험에 대한 좀 더 구체적인 내용들( PyCharm에 Plug-In 설치하고, Coursera 어떤 강의를 들어야 하며, 어떻게 결제하고, 어떻게 제출하는 것)은

검색하면 아주 잘 나오니 여기서는 하지 않도록 하겠습니다.

<br>
<br>

#### 1.1. PyCharm
PyCharm에 조금 익숙해 질 필요가 있습니다.

저는 주로 Jupyter Notebook이나 Visual Studio Code을 사용하고 있었으나, 시험을 위해 PyCharm 사용법을 익혔습니다.

잘 다룰 필요는 없고, Interpreter 연결 / 실행 방법 / Debugging 방법등과 같은 간단한 사용법만 알면 됩니다.

<br>

#### 1.2. Plug-In
'Start Exam' Button을 누르면 Plug-In이 무언가 작업을 합니다. 문제 파일을 Download해서 Local에 복사해 놓습니다.

5개 문제가 모두 동일한 'starter.py'라는 File명으로 되어 있고, Folder Name이 Category1 ~ Category5으로 다릅니다.

<br>

#### 1.3. 문제 확인
내가 지금 실행중인 문제가 몇 번인지 꼭 확인하면서 문제를 풀기 바랍니다.

지금 수정하는 File과 다른 File을 실행하는 우를 범하지 않도록 하십시오.

<br>

#### 1.4. 채점
기본 Skeleton Code와 해당 문제에서 필요한 Dataset은 이미 Folder에 있습니다.

모든 문제에는 solution_model()이라는 함수가 있고, 이 함수는 학습이 완료된 Model을 return하도록 되어있습니다.

이 solution_model()이 return하는 Model을 mymodel.h5로 저장되고, 이 파일을 Test Server로 보내서 실시간으로 채점하는 방식입니다.

Test Server에서 제출한 Model의 성능을 평가하고,결과가 5/5가 되어야 통과입니다.

이런 방식으로 5개의 문제를 모두 풀면 Pass입니다.

<br>

#### 1.5. 문제 설명
오른쪽 'Submit and Test model' Button 밑에 각 문제에 대해서 설명이나 Tip 같은 것들이 자잘하게 적혀있습니다.

예를 들면, 어떤 Layer는 쓰지 말라, Input Shape은 어떻게 하라 등등..

감이 잘 안 잡히면 주의깊게 읽어보는 것도 좋습니다.

<br>

#### 1.6. 오류
아직까지 Plug-In에 아직 문제가 좀 있는 것 같습니다. 'Start Exam'이나 'Submit and Test model'같은 Button이 보이지 않거나

Model을 제출했는데 Plug-In 오류가 발생하는 경우가 있습니다.

이럴때는 당황하지 말고 PyCharm 종료 후 다시 실행하면 되더군요

<br>

#### 1.7. 가상 환경
안내서에는 가상환경을 virtualenv를 사용하라고 나와있습니다.

하지만, 반드시 virtualenv 사용할 필요는 없고, 어떤 가상환경을 사용하든 상관없습니다.

우리가 제출해야 하는 것은 Training된 h5 File이고, 이것을 만들고 Training시키는 Local 환경은 어떤 것이든 상관없습니다.

<br>

#### 1.8. GPU
GPU가 필수는 아닙니다.  GPU가 있으면 Training이 훨씬 더 빠르겠지만, GPU가 필요할 만큼 큰 Dataset이나 복잡한 Model은 없습니다.

앞서 말씀드렸듯이, Train을 마친 h5 File만 제출하면 되기 때문에 Colab에서 Train하고 결과를 Local에 받아서 제출해도 전혀 상관없습니다.       

<br>
<br>
<br>
<br>
<br>

## 2. Tips   

<br>
<br>

* 첫번째 문제는 가격 예측 Regression 문제입니다. 적절하게 Dense Layer 추가하면 쉽게 풀립니다.

<br>

* 두번째 문제는 MNIST Dataset을 이용한 CNN Model 구현 문제입니다. Coursera 강의의 Code를 보면서 풀면 됩니다.

<br>

* 세번째 문제는 Horse / Human 구분 문제입니다. 

ImageDataGenerator를 사용해야 하는 문제이니, 이 함수 사용법 숙지하셔야 합니다.

적절하게 CNN Model 구성하셔서 Accuracy 높은 Model 저장후 제출하시면 됩니다.

Training 결과가 Overfitting 된 것 같아 보여도 불안해 하실 필요없어요.

<br>

* NLP 문제입니다. 문장을 보고 Binary Classification하는 문제입니다.

Model 구성이나 Data Input은 Coursera 강의에서 사용한 Code 그대로 사용하면 됩니다.

Training을 시작하면 어떤 방법(Conv1D , LSTM , Bidirectional LSTM , Dropout 등등)을 써도 Val. Accuracy가 0.83을 넘지 못하더군요

3번 문제처럼 Overfitting 신경 안쓰고 제출하면 4/5처럼 뭔가 약간 부족하다는 결과가 나와서 고생을 좀 했습니다.

다양한 방법으로 여러번 시도해서 제출하다 보니 5/5가 되더라구요.

이런 부분이 좀 힘들었습니다.

<br>

* Time Series 문제입니다.
앞선 4개의 문제는 모두 Coursera 강의에서 다루었던 문제와 동일한 유형의 문제들만 나왔습니다.

하지만, 5번 문제는 Dataset도 처음 다루는 것이었고, Prediction도 하나의 값이 아닌 여러개의 값을 출력해야 하는 문제였습니다.

( Coursera강의에서 Time Series 문제는 Multi Feature를 보고 특정 하나의 값을 예측하는 Regression 문제였습니다. )

Coursera강의에서 Single Output을 변형해서 Multi Output Model을 만들면 해결 가능합니다.

<br>

* 전체적으로 Coursera 강의의 Code를 참고하면 쉽게 풀 수 있는 문제들이며, Classification 문제들이 많았습니다.

<br>

* 점수가 3/5 혹은 4/5가 나올때는 Model 구현은 맞지만, 성능이 원하는 만큼 나오지 않는다는 의미로 생각됩니다.

이런 경우에는 Epoch을 늘리거나 Learning Rate를 조절하는 등의 Tuning을 통해서 5/5를 만들 수 있습니다.                  

<br>
<br>
<br>
<br>
<br>

## 3. Opinion

전반적인 내용은 매우 기초적인 내용이며 또한 강좌와 동일한 문제가 출제됩니다.

<br>

이 자격증을 따기위한 교육이나 강좌도 개설되어 있던데, 이 자격증이 그 학원까지 다니면서 따야할 정도인지는 솔직히 좀 의문입니다.

<br>

제가 생각하는 이 자격증의 의미는 'Tensorflow / Keras를 사용할 줄 아는구나'라는 정도라고 생각합니다.

<br>

자격증 취득을 준비하시는 분들께서 이 글이 조금이나마 도움이 되었으면 좋겠습니다.   

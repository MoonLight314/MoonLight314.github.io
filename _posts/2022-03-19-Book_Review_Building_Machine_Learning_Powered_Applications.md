---
title: "Building Machine Learning Powered Applications"
date: 2021-10-29 08:26:28 -0400
categories: Deep Learning
---
# Building Machine Learning Powered Applications

### 한빛미디어 '나는 리뷰어다' 활동을 위해서 책을 제공받아 작성된 서평입니다.

<br>
<br>

<p align="center">
  <img src="/assets/Book_Review_Assets/Book_Review_Building_Machine_Learning_Powered_Applications.png">
</p>

<br>

### 0. 소개

만약 여러분이 직접 만든 추천 시스템 Machine Learning Model을 여러 사람들에게 서비스하고 싶다면 ?
<br>
<br>
여러분들은 다양한 Dataset으로 훌륭한 Data Preprocessing을 할 수 있고, 훌륭한 직감을 가지고 있으며 훌륭하게 여러분의 Model을 Tuning 할 수 습니다.
<br>
<br>
수 없이 많은 .fit()을 호출하면서 Accuracy / ROC-AUC 등의 지표를 보면서 더욱 나은 성능이 나올 수 있도록 Model을 만들 수 있는 능력이 있습니다.
<br>
<br>
하지만, 여러분들이 훌륭하게 동작하는 Model을 만들 수 있는 능력이 있다고 하더라도, 그 Model을 이용하여 실제 서비스를 구축하는 것은 완전히 다른 문제입니다.
<br>
<br>
<br>

거대한 서비스에 Machine Learning Model을 적용하는 것은 다양한 능력이 필요합니다. Machine Learning 자체에 대한 능력뿐만 아니라, 전체적인 서비스가 작동하는 방식에 대한 이해도 요구됩니다. 
<br>
<br>
서비스가 필요로 하는 Machine Learning Model이 무엇인지에 따라서 어떤 방식의 Machine Learning 기법을 선택할 지, 
실제 적용시에 오류와 실제로 수집되는 Data를 바탕으로 Model을 어떻게 Update할 지와 같은 다양한 문제는 매우 어려운 문제들입니다.
<br>
<br>

여러분들이 이런 문제를 고민하고 있거나, 실제로 Machine Learning Model을 서비스에 적용해야 한다면 이 책을 추천드립니다.
<br>
<br>
시중에는 Machine Learning / Deep Learning 자체를 다루는 훌륭한 책은 매우 많이 있습니다.
<br>
<br>
하지만, Machine Learning / Deep Learning을 이용해서 만든 Model을 실제로 서비스에 적용할 수 있도록 도움을 줄 수 있는 책들은 거의 없습니다.
<br>
<br>

이 책은 Machine Learning Model을 서비스에 적용하기 위한 모든 과정을 설명해 줍니다.
<br>
<br>
관련된 예제 Code들과 오랜 경험을 가진 저자의 조언을 함께 할 수 있기 때문에 목표 달성에 큰 도움이 될 것입니다.
<br>
<br>

필자는 이 책을 통하여, 다양한 실제 경험을 바탕으로 실질적인 서비스 구축에 도움이 되는 Know-How를 전수하고 있습니다.
<br>
<br>
또한, 실제로 Model을 서비스에 적용해 가는 과정을 예제 Project를 통해서 설명하고 있습니다.
<br>
<br>
최근 YouTube를 통해 광고를 하고 있는 'Grammarly'와 유사한 구현해 가는 과정을 설명해 줍니다.
<br>
<br>

이 책은 다음과 같은 독자분들에게 추천드립니다.
 - Python 및 다양한 Machine Learning & Deep Learning Open Source Framework 사용 가능한 분
 - 기초적인 Web 지식을 가지신 분
 - Data Scientist / Data Analyst
 - Machine Learning & Deep Learning 관련 지식은 없지만, 해당 서비스를 실제 적용해야 하는 일을 하시는 분\

<br>
<br>
<br>

### 1. PART I

 * Part I에서는 우리가 구현하고 실제 적용하고자 하는 서비스에 대해서 구체적으로 어떤 기준으로 성공/실패를 판별할지와 최초의 Dataset은 어떻게 선택하고 모을지에 대한 초기 계획을 결정하는데 도움을 줄 수 있는 여러 조언들이 실려있습니다.
<br>

 * 서비스에 처음 Machine Learning을 적용하려고 한다면, 과연 이 작업이 Machine Learning 기법 적용이 적합한지부터 판단해야 할 것입니다.
<br>

 * 기존의 전통적 Programming 방식을 사용중이고 이미 훌륭한 성능을 내고 있는데, Machine Learning을 적용해야 하는지에 대한 고민부터 Machine Learning 적용한다면 분류(Classification) 문제인지 회귀(Regression) 문제인지 등을 판단해야 합니다.
<br>

 * 또한, Machine Learning을 적용하려면 Data는 어떻게 수집하며 어떤 종류의 Data가 필요한지에 대한 고민을 반드시 해야할 것 입니다.
<br>

 * Baseline Model의 성능 평가 방법에 관한 고민도 하여야 할 것입니다.
<br>

 * Part I에서는 위와 같이, Machine Learning을 서비스에 적용하기 위한 최초의 고민들에 대한 실질적이고 유용한 정보를 얻을 수 있습니다.
<br>

<br>
<br>

### 2. PART II

 * Part II에서는 실제 Machine Learning을 사용하지 않고 Prototyping을 하는 방법에 대해서 이야기합니다.
 <br>
 
 * Prototype에서 Machine Learning을 사용하지 않는 이유는 Prototype때 Machine Learning을 사용하지 않는 것이 오히려 가장 적합한 Machine Learning Model을 선택하는데 도움이 되기 때문입니다.
 <br>
 
 * 이번 Part에서는 Prototyping에 대한 기본적인 규칙 및 기본적인 Data Preprocessing 작업에 대한 Know-How를 공유합니다.
 <br>
 
 * 대부분의 Machine Learning / Deep Learning 관련 교육이나 책에서는 최초부터 깔끔하게 Preprocessing을 거친 Dataset이 제공이 되고, Model 그 자체에 교육이 집중됩니다.
 <br>
 
 * 하지만, 현실적으로 Data 준비에 굉장히 많은 시간과 노력이 들어갑니다. 이 책은 다른 교육이나 책에서 간과한 이런 부분들에 대해서 현실적인 도움을 줍니다.
 <br>
 





3. PART III
 - Part III에서는 앞에서 Prototype으로 모은 Data를 바탕으로 실제 적용할 Model을 Train시키고 성능을 높이는 과정을 설명합니다.
 - Machine Learning / Deep Learning Model의 선택 / 구현 / 측정 / 분석의 반복에 대해여 설명하고 Know-How를 공유합니다.
 - 최초 Model 선택시에 어떤 사항들을 고려해야 하는지, 최초 시도 Model 구현시에 어떤 open source library or framework을 이용하여 빠르게 구현해야 하는지와 이런 선택에서 실제 배포시에 고려해야 할 것이 있는지 등과 같이 Model 관련 중요 사항에 대해서 이야기 합니다.
 - 또한, Model 뿐만 아니라 Model 만큼 중요한 Data 생성 방법 및 주의 사항들에 대해서도 이야기합니다.
 - 이 Part에서 가장 중요하다고 할 수 있는 부분은 Model의 Debugging입니다.
 - 사실 이 부분은 Machine Learning / Deep Learning Model 책이나 교육에서도 잘 다루지 않는 부분이라고 생각합니다. 그 이유는 고려해야할 사항들이 워낙 많고 영향을 미치는 변수들이 많기 때문에 어떤 것이 정답이라고 단정적으로 말하기 어렵기 때문일 것입니다.
 - 이 책은 성능에 영향을 줄 수 있는 다양한 요소들을 설명해주며 실질적 Debugging 방법을 공유합니다.






4. PART IV
 - Part IV에서는 Train된 Model을 실제 서비스에 적용하는 방법에 대해서 다룹니다.
 - Model 자체는 훌륭할지 모르지만, 적용 방법에 따라서 실패하는 경우도 있으니, 이 Part에서는 실무에서의 범할 수 있는 오류를 줄이고, Model이 잘 작동하는지 Monitoring하는 방법에 대해서 다룹니다.
 - 배포 방식에 따른 고려사항들, Server / Client 고려 사항
 - Data 수집시에 고려해야할 사항 ( 윤리적 문제 / 소유권 문제 )
 - Model이 잘못된 동작을 하고 있을 때 대처 방안
 - 모니터링에 관련해서도 어떤 요소를 모니터링 할 것인가와 같은 이야기를 하게 된다.
 - 또한, Update된 Data로 성능이 좋아진 Model을 어느 시점에 재배포를 할 것인지 그리고 어떤 방식으로 재배포를 할 것인가에 대한 이야기를 하게 된다.
 
 


5. 총평
 - 이 책은 Machine Learning Model을 실제 서비스에 적용하고는 싶은데, 어디서부터 무엇을 어떻게 해야 할 지 막막한 분들에게 단비같은 책입니다.
 - 다만, 이 책에 담겨있는 모든 내용들을 모두 자신의 것으로 만들기 위해서는 다방면에 대한 지식이 필요하다는 것이 조금 부담스러울수는 있지만, 저자가 다년간 쌓은 Know-How를 얻을 수 있다면 충분히 도전해 볼 가치가 있다고 본다.

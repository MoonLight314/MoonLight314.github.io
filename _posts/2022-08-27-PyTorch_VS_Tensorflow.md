---
title: "PyTorch vs Tensorflow"
date: 2021-10-29 08:26:28 -0400
categories: Deep Learning
---
# PyTorch vs TensorFlow in 2022

<br>
<br>
<br>

PyTorch와 TensorFlow는 오늘날 가장 인기 있는 두 가지 Deep Learning Framework입니다. 

각 진영에는 열렬한 지지자들이 있고, 어떤 Framework가 더 우월한가에 대해서 오랫동안 논쟁이 되어 왔습니다.

PyTorch와 TensorFlow는 비교적 짧은 시간 동안 빠르게 발전하여서 논쟁이 계속되고 있습니다.

TensorFlow는 산업 중심 Framework로 그리고, PyTorch는 연구 중심 Framework로 유명하지만 이러한 개념은 부분적으로 오래된 정보에서 비롯되었습니다.

어떤 Framework가 더 나은지에 대한 대화는 2022년이 되면서 더 복잡해집니다.

이제 두 Framework의 차이점을 살펴보겠습니다.

<br>
<br>

# 1. Practical Considerations

<br>
<br>

PyTorch와 TensorFlow는 모두 고유한 개발 배경과 복잡한 Design-Decision History를 가지고 있습니다. 

이전에는 이로 인해 현재 기능과 예상되는 미래 기능에 대한 복잡한 기술 비교 논의가 있었습니다.

두 Framework가 시작 이후 기하급수적으로 성숙했다는 점을 감안할 때 이러한 기술적 차이점 중 많은 부분이 이 때의 흔적으로 남아있습니다.

PyTorch와 TensorFlow 논쟁은 현재 세 가지 실질적인 고려 사항을 중심으로 살펴볼 수 있습니다.

<br>

## 1) Model Availability(Model Availability)
 - Deep Learning의 영역이 매년 확장되고 Model이 차례로 커지면서 처음부터 SOTA Model을 Train하는 것은 더 이상 불가능합니다.
  다행히 공개적으로 사용할 수 있는 SOTA Model이 많이 있으며 가능한 한 이를 활용하는 것이 중요합니다.

<br>

## 2) 배포 인프라(Deployment Infrastructure)
 - 잘 학습된 Model을 실제로 사용할 수 없다면 아무 의미가 없습니다.
특히 Micro Service Business Model의 인기가 높아짐에 따라 배포 시간을 줄이는 것이 무엇보다 중요합니다. 
효율적인 배포는 Machine Learning을 중심으로 하는 많은 Business의 성패를 가늠할 수 있습니다.

<br>

## 3) Ecosystem(Ecosystems)
 - 더 이상 Deep Learning은 고도로 통제된 환경의 특정 사용 사례와 관련이 없습니다. 
AI는 수많은 산업에 새로운 힘을 불어넣고 있으므로 Mobile, Local 및 Server Application 개발을 용이하게 하는 더 큰 Ecosystem 내에 있는 Framework가 중요합니다. 
또한 Google의 Edge TPU와 같은 특수 Machine Learning HW의 등장으로 개발자는 HW와 잘 통합될 수 있는 Framework를 사용해야 합니다.

<br>

우리는 앞에서 언급한 3가지를 차례로 살펴보고 우리들이 결론 내린 추천 Framework을 알려주겠습니다.

<br>

이 세 가지 실용적인 고려 사항을 차례로 살펴보고 다른 영역에서 사용할 Framework에 대한 권장 사항을 제공하려고 합니다.

<br>
<br>

# 2. PyTorch vs TensorFlow - Model Availability

<br>

처음부터 성공적인 Deep Learning Model을 구현하는 것은 특히 Engineering 및 최적화가 어려운 NLP와 같은 Application의 경우 매우 까다로운 작업이 될 수 있습니다. 

SOTA Model의 복잡성이 증가함에 따라 소규모 기업에서는 Train 및 조정이 비현실적이고 거의 불가능에 가까워졌습니다. 

OpenAI의 GPT-3에는 1,750억 개 이상의 Parameter가 있고 GPT-4에는 100조 개 이상의 Parameter가 있습니다. 

스타트업과 연구원 모두 스스로 이러한 Model을 활용하고 탐색할 수 있는 Computing Resource가 없기 때문에 이전 학습, Fine-Tuning 또는 즉시 사용 가능한 Inference을 위해 Pre-Trained Model에 Access하는 것이 매우 중요합니다.

<br>

Model Availability의 영역에서 PyTorch가 TensorFlow는 크게 다릅니다. 

PyTorch와 TensorFlow에는 모두 자체 공식 Model Repository가 있습니다. 

아래 Ecosystem Section에서 살펴보겠지만 실무자는 다른 Source의 Model을 활용하고 싶을 수 있습니다. 

각 Framework에 대한 Model Availability을 정량적으로 살펴보겠습니다.

<br>

## 1) HuggingFace

[HuggingFace](https://huggingface.co/)를 사용하면 단 몇 줄의 Code로 Train되고 조정된 SOTA Model을 Pipeline에 통합할 수 있습니다.

PyTorch와 TensorFlow에 대한 HuggingFace Model Availability을 비교해 보면 결과는 놀랍습니다.

아래에서 우리는 PyTorch 또는 TensorFlow 독점이거나 두 Framework 모두에서 사용할 수 있는 HuggingFace에서 사용할 수 있는 총 Model수에 대한 정보를 알 수 있습니다.

보시다시피 PyTorch에서만 사용할 수 있는 Model의 수가 압도적으로 많습니다.

거의 85%의 Model이 PyTorch 독점이며 독점이 아닌 Model도 PyTorch에서 사용할 수 있는 확률이 약 50%입니다. 

<br>

대조적으로 모든 Model의 약 16%만 TensorFlow에서 사용할 수 있으며 약 8%만 TensorFlow 전용입니다.

<p align="center">
  <img src="/assets/PyTorch_vs_Tensorflow/00.png">
</p>

<br>

HuggingFace에서 가장 인기 있는 30개 Model로만 분류하면 비슷한 결과가 나타납니다. 

상위 30개 Model 중 2/3도 TensorFlow에서 사용할 수 없지만 모두 PyTorch에서 사용할 수 있습니다. 

TensorFlow 독점인 상위 30개 Model은 없습니다.

<br>

<p align="center">
  <img src="/assets/PyTorch_vs_Tensorflow/01.png">
</p>

<br>

## 2) Research Papers

특히 연구 실무자에게는 최근에 발표된 Paper의 Model에 Access하는 것이 중요합니다. 

다른 Framework에서 탐색하려는 새 Model을 다시 만들려고 시도하는 것은 귀중한 시간을 낭비하므로 저장소를 복제하고 즉시 실험을 시작할 수 있다는 것은 중요한 작업에 집중할 수 있다는 것을 의미합니다.

PyTorch가 사실상의 연구 Framework라는 점을 감안할 때 HuggingFace에서 관찰한 추세가 연구 커뮤니티 전체로 계속 이어질 것으로 예상합니다. 

그리고 우리의 직관은 정확합니다.

<br>

PyTorch 또는 TensorFlow를 사용하는 출판물의 상대적 비율을 보여주는 아래 그래프는 지난 몇 년 동안 8개 상위 연구 저널의 Data를 집계했습니다. 

보시다시피 PyTorch의 채택이 매우 빠르게 중가하는 것을 볼 수 있습니다.

<br>

<p align="center">
  <img src="/assets/PyTorch_vs_Tensorflow/02.png">
</p>

<br>

이러한 변화의 이유는 연구 목적에서 Tensorflow 1.x 사용이 어렵기 때문에 사용자들이 PyTorch로 옮겨가게 되었다고 생각합니다.

TensorFlow의 많은 문제가 2019년 TensorFlow 2의 출시와 함께 해결되었지만 PyTorch의 추진력은 최소한 커뮤니티 관점에서 확립된 연구 중심 Framework로 유지될 만큼 충분히 컸습니다.

<br>

Framework를 Migration한 연구자의 비율을 보면 동일한 패턴을 볼 수 있습니다. 

2018년과 2019년에 PyTorch 또는 TensorFlow를 사용했던 저자의 출판물을 살펴보면 2018년에 TensorFlow를 사용한 저자의 대다수가 2019년에 PyTorch로 Migration한 반면(55%) PyTorch를 사용한 저자의 대다수는 2018년에는 PyTorch 2019(85%)를 유지했습니다. 

이 Data는 아래 Sankey Diagram에 시각화되어 있으며 왼쪽은 2018년, 오른쪽은 2019년에 해당합니다. 

Data는 총 수가 아닌 2018년 각 Framework의 사용자 비율을 나타냅니다.

<br>

<p align="center">
  <img src="/assets/PyTorch_vs_Tensorflow/03.png">
</p>

<br>

주의 깊은 독자는 이 Data가 TensorFlow 2 Release 이전의 Data라는 것을 알 수 있지만 다음 Section에서 볼 수 있듯이 이 사실은 연구 커뮤니티와 관련이 없습니다.

<br>

## 3) Papers with Code

Machine Learning Paper, Code, Data Set 등으로 무료 공개 Resource를 만드는 것이 임무인 Web Site인 [Papers with Code](https://paperswithcode.com/)의 Data를 살펴봅니다. 

2017년 말부터 현재 분기까지 PyTorch , Tensorflow 그리고 다른 Framework을 이용한 Paper의 비율을 그려보았습니다.

PyTorch를 활용하는 Paper의 꾸준한 성장을 보고 있고, 이번 분기에 생성된 4,500개의 Repository 중 60%는 PyTorch에서 구현되고 11%는 TensorFlow에서 구현되었습니다.

<br>

<p align="center">
  <img src="/assets/PyTorch_vs_Tensorflow/04.png">
</p>

<br>

반대로 TensorFlow 사용은 꾸준히 감소하고 있습니다.

2019년 TensorFlow 1을 사용하여 연구를 어렵게 만든 많은 문제를 해결한 TensorFlow 2의 출시조차도 이러한 추세를 뒤집기에 충분하지 않았습니다.

TensorFlow 2가 출시된 후에도 TensorFlow의 인기가 거의 단조롭게 감소했습니다.

<br>

## 4) Model Availability - Final Words

<br>

위의 Data에서 PyTorch가 현재 연구 환경을 지배하고 있음이 분명합니다.

TensorFlow 2는 연구를 위해 TensorFlow를 훨씬 쉽게 활용할 수 있게 해주었지만, PyTorch는 연구자들이 TensorFlow로 돌아가서 다시 시도해 볼 이유가 없습니다.

또한 TensorFlow 1의 이전 연구와 TensorFlow 2의 새로운 연구 사이의 하위 호환성 문제는 이 문제를 악화시킬 뿐입니다.


현재로서는 PyTorch가 커뮤니티에서 널리 채택되었고 대부분의 출판물/사용 가능한 Model이 PyTorch를 사용하기 때문에 연구 분야에서 확실한 승자입니다.

<br>

몇 가지 주목할만한 예외/참고 사항이 있습니다.

<br>

* **Google AI**

  분명히 Google에서 발표한 연구는 주로 TensorFlow를 사용합니다. 

  Google이 Facebook(2020년 NeurIPS 또는 ICML에 게시된 기사 292개 대 92개([관련자료 Link](https://www.cnbc.com/2021/01/21/deepmind-openai-fair-ai-researchers-rank-the-top-ai-labs-worldwide.html))보다 
 
  훨씬 더 많다는 점을 감안할 때 일부 연구자는 TensorFlow를 사용하는 것이 유용하거나 최소한 능숙할 수 있습니다. 
 
  또한, Google Brain은 JAX를 위한 Google의 신경망 Library인 [Flax](https://github.com/google/flax)와 함께 JAX를 사용합니다.

<br>

* **DeepMind**

  DeepMind는 2016년에 TensorFlow 사용을 표준화했지만 2020년에 연구를 가속화하기 위해 JAX를 사용한다고 발표했습니다. 
  
  또한 이 발표에서 그들은 JAX Ecosystem, 특히 JAX 기반 신경망 Library인 [Haiku](https://github.com/deepmind/dm-haiku)에 대한 개요도 제공합니다.
  
  DeepMind는 Facebook보다 더 많은 작업을 수행합니다(2020년 NeurIPS 또는 ICML에 게시된 기사 110 대 92([관련 자료 Link](https://www.deepmind.com/blog/using-jax-to-accelerate-our-research)). 
  
  DeepMind는 연구용으로 맞춤화되고 때때로 "Keras의 연구 버전"이라고 불리는 TensorFlow용 고급 API인 [Sonnet](https://github.com/deepmind/sonnet)을 만들었습니다. 
  
  이는 연구에 TensorFlow 사용을 고려하는 사람들에게 유용할 수 있습니다. 
  
  또한 DeepMind의 [Acme](https://github.com/deepmind/acme) Framework는 Reinforcement Learning 실무자에게 필수적일 수 있습니다.

<br>

* **OpenAI**

  반면에 OpenAI는 2020년에 내부적으로 PyTorch의 사용을 표준화했습니다. 
  
  그러나 Reinforcement Learning에 있는 사람들의 경우 이전 기준 저장소인 [Baselines](https://github.com/openai/baselines)은 Reinforcement Learning Algorithm의 고품질 구현을 제공하므로 TensorFlow는 Reinforcement Learning 실무자에게 최고의 선택이 될 수 있습니다.

<br>

* **JAX**
  
  Google에는 연구 커뮤니티에서 인기를 얻고 있는 [JAX](https://github.com/google/jax)라는 또 다른 Project가 있습니다. 
  
  어떤 의미에서는 PyTorch 또는 TensorFlow에 비해 JAX의 Overhead가 훨씬 적습니다. 
  
  그러나 기본 철학은 PyTorch 및 TensorFlow와 다르며 이러한 이유로 JAX로 Migration하는 것은 대부분의 경우 좋은 선택이 아닐 수 있습니다. 
  
  JAX를 활용하는 Model/Paper가 증가하고 있지만 현재로서는 PyTorch 및 TensorFlow와 비교하여 향후 연구 커뮤니티에서 JAX가 얼마나 널리 퍼질지 확실하지 않습니다.
 
<br>

TensorFlow가 지배적인 연구 Framework로 재건되기를 원한다면 불가능하지는 않더라도 길고 험난한 여정을 거쳐야 합니다.

PyTorch vs TensorFlow 토론의 1라운드는 PyTorch의 승리입니다.

<br>
<br>
<br>

# 3. PyTorch vs TensorFlow - Deployment

<br>

최신 결과를 위해 SOTA Model을 사용하는 것은 Inference 관점에서 Deep Learning Application의 성배이지만, 이 이상은 항상 실용적이거나 산업 환경에서 달성할 수 있는 것은 아닙니다. 

SOTA Model을 사용하는데 있어서 시간이 많이 걸리고 오류가 발생이 많다면 SOTA Model에 대한 Access는 무의미합니다. 

따라서 가장 좋은 Model에 Access할 수 있는 Framework를 고려하는 것 외에도 각 Framework에서 종단 간 Deep Learning Process를 고려하는 것이 중요합니다.


TensorFlow는 처음부터 배포 지향 Application을 위한 필수 Framework였으며 그럴만한 이유가 있습니다. 

TensorFlow에는 종단 간 Deep Learning Process를 쉽고 효율적으로 만드는 수많은 관련 도구가 있습니다. 

특히 배포의 경우 TensorFlow Serving 및 TensorFlow Lite를 사용하면 Cloud, Server, Mobile 및 IoT 장치에 손쉽게 배포할 수 있습니다.


PyTorch는 배포 관점에서 극도로 부진했지만 최근 몇 년 동안 이 격차를 좁히기 위해 노력했습니다.

작년에 TorchServe와 불과 몇 주 전에 PyTorch Live가 도입되면서 꼭 필요한 기본 배포 도구가 제공되었지만 PyTorch가 업계 환경에서 사용할 가치가 있을 만큼 배포 격차를 좁혔는지 한 번 살펴보도록 하겠습니다.

<br>

## 3.1. Tensorflow

TensorFlow는 Inference 성능에 최적화된 정적 그래프로 확장 가능한 결과를 제공합니다. 

TensorFlow로 Model을 배포할 때 Application에 따라 TensorFlow Serving 또는 TensorFlow Lite를 사용합니다.

<br>

### 3.1.1. Tensorflow Serving

[TensorFlow Serving](https://www.tensorflow.org/tfx/guide/serving)은 TensorFlow Model을 사내 또는 Cloud에 배포할 때 사용하며 [TensorFlow Extended(TFX)](https://www.tensorflow.org/tfx) 종단 간 Machine Learning Platform 내에서 사용됩니다. 
Serving을 사용하면 Model Tag가 있는 잘 정의된 디렉터리로 Model을 쉽게 직렬화하고 Server Architecture와 API를 정적으로 유지하면서 Inference 요청을 만드는 데 사용할 Model을 선택할 수 있습니다.

Serving을 사용하면 고성능 [RPC](https://grpc.io/)를 위한 Google의 Open Source Framework를 실행하는 특수 gRPC Server에 Model을 쉽게 배포할 수 있습니다. 

gRPC는 다양한 Micro Service Ecosystem를 연결하기 위해 설계되었으므로 이러한 Server는 Model 배포에 적합합니다. 

전체 서비스는 [Vertex AI](https://cloud.google.com/vertex-ai)를 통해 Google Cloud와 긴밀하게 통합되고 [Kubernetes](https://kubernetes.io/) 및 [Docker](https://www.docker.com/)와 통합됩니다.

<br>

### 3.1.2. Tensorflow Lite

[TensorFlow Lite(TFLite)](https://www.tensorflow.org/lite)는 Mobile 또는 IoT/Embedded 장치에 TensorFlow Model을 배포할 때 사용합니다. 

TFLite는 이러한 장치에 대한 Model을 압축 및 최적화하고 On Device AI에 대한 5가지 제약(대기 시간, 연결성, 개인 정보 보호, 크기 및 전력 소비)을 보다 광범위하게 해결합니다. 

동일한 Pipeline을 사용하여 표준 Keras 기반 저장된 Model(Serving과 함께 사용)과 TFLite Model을 동시에 내보내므로 Model 품질을 비교할 수 있습니다.


TFLite는 [Micro Controller(Bazel](https://bazel.build/) 또는 [CMake](https://cmake.org/)가 있는 ARM) 및 Embedded Linux(예: [Coral](https://coral.ai/) 장치)뿐 아니라 Android 및 iOS 모두에 사용할 수 있습니다. 
TensorFlow의 Python, Java, C++, JavaScript 및 Swift용 API는 개발자에게 다양한 언어 옵션을 제공합니다.

<br>

## 3.2. PyTorch

PyTorch는 이전에 이 분야에서 악명 높았던 배포를 보다 쉽게 만드는 데 투자했습니다. 

이전에는 PyTorch 사용자가 Flask 또는 Django를 사용하여 Model 위에 REST API를 Build해야 했지만 이제는 TorchServe 및 PyTorch Live 형태의 기본 배포 옵션이 있습니다.

<br>

### 3.2.1. TorchServe

[TorchServe](https://pytorch.org/serve/)는 AWS와 Facebook(현재 Meta) 간의 협업을 통해 생성된 Open Source 배포 Framework로 2020년에 출시되었습니다. 

Endpoint Specification, Model Archiving 및 메트릭 관찰과 같은 기본 기능이 있습니다. 

그러나 TensorFlow 대안보다 열등합니다. REST 및 gRPC API는 모두 TorchServe에서 지원됩니다.

<br>

### 3.2.2. PyTorch Live

PyTorch는 2019년에 [PyTorch Mobile](https://pytorch.org/mobile/home/)을 처음 출시했습니다. 

PyTorch Mobile은 Android, iOS 및 Linux에 최적화된 Machine Learning Model 배포를 위한 종단 간 Workflow를 생성하도록 설계되었습니다.


[PyTorch Live](https://playtorch.dev/)는 Mobile 기반으로 12월 초에 출시되었습니다. 

JavaScript 및 React Native를 사용하여 관련 UI가 있는 Platform 간 iOS 및 Android AI 기반 앱을 만듭니다. 

기기 내 Inference는 여전히 PyTorch Mobile에서 수행됩니다. 

Live는 Bootstrap할 예제 Project와 함께 제공되며 향후 Audio 및 Video 입력을 지원할 계획이 있습니다.

<br>
<br>

## 3.3. 배포 - 최종 단어

현재 TensorFlow는 여전히 배포 측면에서 우위를 점하고 있습니다. 

Serving 및 TFLite는 PyTorch보다 훨씬 강력하며 Google의 Coral 장치와 함께 Local AI에 TFLite를 사용하는 기능은 많은 산업 분야에서 필수 요소입니다. 

대조적으로 PyTorch Live는 Mobile에만 초점을 맞추고 TorchServe는 아직 초기 단계에 있습니다. 

향후 몇 년 동안 배포 영역이 어떻게 변경되는지 보는 것은 흥미로울 것이지만 현재로서는 PyTorch 대 TensorFlow 논쟁의 2라운드가 TensorFlow가 유리합니다.


**Model Availability 및 배포 문제에 대한 마지막 참고 사항**
- TensorFlow Deployment Infrastructure를 사용하고 싶지만 PyTorch에서만 사용할 수 있는 Model에 Access하려는 경우 [ONNX](https://onnx.ai/)를 사용하여 PyTorch에서 TensorFlow로 Model을 이식하는 것을 고려하십시오.

<br>
<br>
<br>

# 4. PyTorch vs TensorFlow - Ecosystems

<br>

2022년 PyTorch와 TensorFlow를 구분하는 마지막 중요한 고려 사항은 이들이 위치한 Ecosystem입니다. 

PyTorch와 TensorFlow는 모두 Model링 관점에서 볼 때 유능한 Framework이며, 이 시점에서의 기술적 차이점은 쉬운 배포, 관리, 분산 교육 등을 위한 도구를 제공하는 주변 Ecosystem보다 덜 중요합니다. 

각 Framework의 Ecosystem를 살펴보겠습니다.

<br>

## 4.1. PyTorch

<br>

### 4.1.1. Hub

HuggingFace와 같은 Platform 외에도 Pre-Trained Model과 Repository를 공유하기 위한 연구 중심 Platform인 공식 PyTorch Hub도 있습니다. 

Hub에는 Audio, Vision 및 NLP용 Model을 포함하여 다양한 Model이 있습니다. 

또한 유명인 얼굴의 고품질 이미지를 생성하기 위한 GAN을 포함한 생성 Model도 있습니다.

<br>

### 4.1.2. PyTorch-XLA

Google Cloud TPU에서 PyTorch Model을 학습시키려면 [PyTorch-XLA](https://pytorch.org/xla/release/1.9/index.html)가 적합한 도구입니다. 

PyTorch-XLA는 둘을 XLA Deep Learning Compiler와 연결하는 Python Package입니다. 

여기에서 PyTorch-XLA의 [GitHub Repository](https://github.com/pytorch/xla)를 확인할 수 있습니다.

<br>

### 4.1.3. TorchVision

[TorchVision](https://pytorch.org/vision/stable/index.html)은 PyTorch의 공식 Computer Vision Library입니다. 

여기에는 [Model Architecture](https://pytorch.org/vision/stable/models.html) 및 인기 있는 [Data Set](https://pytorch.org/vision/stable/datasets.html)를 포함하여 Computer Vision Project에 필요한 모든 것이 포함되어 있습니다.

더 많은 Vision Model은 [TIMM(pyTorch IMage Models)](https://github.com/rwightman/pytorch-image-models)에서 확인할 수 있습니다. 

여기에서 [TorchVision GitHub Repository](https://github.com/pytorch/vision)를 확인할 수 있습니다.

<br>

### 4.1.4. TorchText

전문 분야가 Computer Vision이 아닌 Natural Language Processing이라면 [TorchText](https://pytorch.org/text/stable/index.html)를 확인하는 것이 좋습니다. 

여기에는 NLP Domain에서 자주 볼 수 있는 DataSet과 이러한 DataSet 및 기타 DataSet에서 작동하는 Data 처리 Utility가 포함됩니다. 

Text에 대한 번역 및 요약과 같은 작업을 수행할 수 있도록 [Fairseq](https://github.com/facebookresearch/fairseq)도 확인하고 싶을 수 있습니다. 

여기에서 [TorchText GitHub Repository](https://github.com/pytorch/text)를 확인할 수 있습니다.

<br>

### 4.1.5. TorchAudio

아마도 Text를 처리하기 전에 [ASR](https://www.assemblyai.com/blog/kaldi-speech-recognition-for-beginners-a-simple-tutorial/)을 사용하여 Audio File에서 Text를 추출해야 합니다. 

이 경우 [TorchAudio](https://pytorch.org/audio/stable/index.html) - PyTorch의 공식 Audio Library를 확인하십시오. 

TorchAudio에는 DeepSpeech 및 Wav2Vec과 같은 인기 있는 [Audio Model](https://pytorch.org/audio/stable/models.html)이 포함되어 있으며 ASR 및 기타 작업을 위한 [연습](https://pytorch.org/audio/stable/tutorials/speech_recognition_pipeline_tutorial.html) 및 [Pipeline](https://pytorch.org/audio/stable/pipelines.html#)을 제공합니다. 

여기에서 [TorchAudio의 GitHub Repository](https://github.com/pytorch/audio)를 확인할 수 있습니다.

<br>

### 4.1.6. SpeechBrain

TorchAudio가 원하는 것이 아니라면 PyTorch용 Open Source 음성 Toolkit인 [SpeechBrain](https://speechbrain.github.io/)을 확인하는 것이 좋습니다. 

[SpeechBrain](https://www.assemblyai.com/blog/introducing-assemblyai-auto-chapters-summarize-audio-and-video-files/)은 ASR, 화자 인식, 확인 및 분할 등을 지원합니다! 

Model을 구축하지 않고 대신 자동 챕터, 감정 분석, 엔티티 감지 등과 같은 기능을 갖춘 Plug & Play 도구를 원하는 경우 AssemblyAI의 자체 [Speech-to-Text API](https://www.assemblyai.com/blog/the-top-free-speech-to-text-apis-and-open-source-engines/)를 확인하십시오.

<br>

### 4.1.7. ESPnet

[ESPnet](https://github.com/espnet/espnet)은 Kaldi의 Data 처리 스타일과 함께 PyTorch를 사용하는 종단 간 음성 처리용 Toolkit입니다. 

ESPnet을 사용하면 종단 간 음성 인식, 번역, 분할 등을 구현할 수 있습니다!

<br>

### 4.1.8. AllenNLP

더 많은 NLP 도구를 찾고 있다면 PyTorch를 기반으로 구축되고 Allen Institute for AI(https://allenai.org/)가 지원하는 Open Source NLP 연구 Library인 [AllenNLP](https://github.com/allenai/allennlp)를 확인하는 것이 좋습니다.

<br>

### 4.1.9. Ecosystem Tools

Computer Vision 또는 Natural Language Processing에 맞게 조정된 Library와 같이 유용할 수 있는 다른 Library에 대해서는 [PyTorch의 도구](https://pytorch.org/ecosystem/) Page를 확인하십시오. 

여기에는 최신 모범 사례를 사용하여 신경망을 생성하기 위한 인기 있는 Library인 [fast.ai](https://docs.fast.ai/)가 포함됩니다.

<br>

### 4.1.10. TorchElastic

[TorchElastic](https://pytorch.org/elastic/latest/)은 2020년에 출시되었으며 AWS와 Facebook의 협업 결과입니다. 

Train에 영향을 미치지 않고 동적으로 변경할 수 있는 Computing Node Cluster에서 Model을 Train할 수 있도록 작업자 Process를 관리하고 재시작 동작을 조정하는 분산 Train용 도구입니다. 

따라서 TorchElastic은 Server 유지 관리 이벤트 또는 Network 문제와 같은 문제로 인한 치명적인 오류를 방지하여 Train 진행 상황을 잃지 않도록 합니다. 

TorchElastic은 Kubernetes와의 통합 기능을 제공하며 PyTorch 1.9+에 통합되었습니다.

<br>

### 4.1.11. TorchX

[TorchX](https://pytorch.org/torchx/latest/)는 Machine Learning Application의 빠른 구축 및 배포를 위한 SDK입니다. 

TorchX에는 지원되는 Scheduler에서 분산 PyTorch Application을 시작하기 위한 Training Session Manager API가 포함되어 있습니다. 

TorchElastic에서 Local로 관리하는 작업을 기본적으로 지원하면서 분산 작업을 시작하는 역할을 합니다.

<br>

### 4.1.12. Lightning

[PyTorch Lightning](https://www.pytorchlightning.ai/)은 PyTorch의 Keras라고도 합니다. 

이 비교가 약간 오해의 소지가 있지만 Lightning은 PyTorch의 Model Engineering 및 교육 Process를 단순화하는 데 유용한 도구이며 2019년 초기 Release 이후 상당히 성숙했습니다. 

Lightning은 객체 지향 방식으로 Model링 Process에 접근하여 재사용 가능하고 Project 전체에서 활용할 수 있는 공유 가능한 구성 요소입니다. 

Lightning에 대한 자세한 내용과 해당 Workflow를 Vanilla PyTorch와 비교하는 방법에 대한 자세한 내용은 이 [자습서](https://www.assemblyai.com/blog/pytorch-lightning-for-dummies/)를 참조하세요.

<br>
<br>

## 4.2. TensorFlow

<br>

### 4.2.1. Hub

[TensorFlow Hub](https://www.tensorflow.org/hub)는 Fine-Tuning할 준비가 된 Train된 Machine Learning Model의 Repository이므로 몇 줄의 Code로 BERT와 같은 Model을 사용할 수 있습니다. 

Hub에는 다양한 사용 사례를 위한 TensorFlow, TensorFlow Lite 및 TensorFlow.js Model이 포함되어 있으며 이미지, Video, Audio 및 Text 문제 Domain에 사용할 수 있는 Model이 있습니다. 

[여기](https://www.tensorflow.org/hub/tutorials)에서 Tutorial을 시작하거나 여기에서 [Model 목록](https://tfhub.dev/s?subtype=module,placeholder)을 확인하세요.

<br>

### 4.2.2. Model Garden

바로 사용할 수 있는 Pre-Trained Model이 Application에서 작동하지 않을 경우 TensorFlow의 [Model Garden](https://github.com/tensorflow/models)은 SOTA Model의 Source Code를 사용할 수 있도록 하는 Repository입니다. 

Model이 작동하는 방식을 이해하기 위해 내부적으로 들어가거나 자신의 필요에 맞게 수정하려는 경우 유용합니다. 

이는 Transfer Learning 및 Fine-Tuning을 넘어 직렬화된 Pre-Trained Model로는 불가능한 일입니다.

Model Garden에는 Google에서 유지 관리하는 공식 Model과 연구원이 유지 관리하는 연구 Model 및 커뮤니티에서 유지 관리하는 선별된 커뮤니티 Model에 대한 디렉토리가 포함되어 있습니다. 

TensorFlow의 장기 목표는 Hub의 Model Garden에서 Pre-Trained Model 버전을 제공하고 Hub의 Pre-Trained Model이 Model Garden에서 사용 가능한 Source Code를 갖도록 하는 것입니다.

<br>

### 4.2.3. Extended (TFX)

[TensorFlow Extended](https://www.tensorflow.org/tfx)는 Model 배포를 위한 TensorFlow의 종단 간 Platform입니다. 

Data를 로드, 검증, 분석 및 변환할 수 있습니다. Model Train 및 Evaluate, Serving 또는 Lite를 사용하여 Model을 배포합니다. 

TFX는 Jupyter 또는 Colab과 함께 사용할 수 있으며 Orchestration을 위해 [Apache Airflow/Beam](https://beam.apache.org/) 또는 Kubernetes를 사용할 수 있습니다. 

TFX는 Google Cloud와 긴밀하게 통합되어 있으며 Vertex AI Pipelines와 함께 사용할 수 있습니다.

<br>

### 4.2.4. Vertex AI

[Vertex AI](https://cloud.google.com/vertex-ai)는 Google Cloud의 통합 Machine Learning Platform입니다. 

올해 출시되었으며 GCP, AI Platform 및 AutoML의 서비스를 하나의 Platform으로 통합하려고 합니다. 

Vertex AI는 Serverless 방식으로 Workflow를 Orchestration하여 Machine Learning System을 자동화, Monitoring 및 관리하는 데 도움이 될 수 있습니다. 

Vertex AI는 또한 Workflow의 아티팩트를 저장할 수 있으므로 종속성과 Model의 Train Data, Hyperparameter 및 Source Code를 추적할 수 있습니다.

<br>

### 4.2.5. MediaPipe

[MediaPipe](https://mediapipe.dev/)는 얼굴 감지, 다중 손 추적, 객체 감지 등에 사용할 수 있는 다중 모드, 교차 Platform 적용 Machine Learning Pipeline을 구축하기 위한 Framework입니다. 

이 Project는 [Open Source](https://github.com/google/mediapipe)이며 Python, C++ 및 JavaScript를 비롯한 여러 언어로 된 Binding이 있습니다. 

MediaPipe 및 즉시 사용 가능한 Solution 시작에 대한 자세한 정보는 [여기](https://google.github.io/mediapipe/)에서 찾을 수 있습니다.

<br>

### 4.2.6. Coral

Cloud 기반 AI에 의존하는 다양한 SaaS 회사가 있지만 많은 산업 분야에서 Local AI에 대한 수요가 증가하고 있습니다. 

[Google Coral](https://coral.ai/)은 이러한 요구 사항을 해결하기 위해 만들어졌으며 Local AI로 제품을 구축하기 위한 완벽한 Toolkit입니다. 

Coral은 2020년에 출시되었으며 개인 정보 보호 및 효율성을 포함하여 배포 Section의 TFLite 부분에 언급된 Onboard AI 구현의 어려움을 해결합니다.


Coral은 프로토타이핑, 생산 및 감지를 위한 일련의 HW 제품을 제공하며, 그 중 일부는 AI Application을 위해 특별히 제작된 본질적으로 더 강력한 Raspberry Pi입니다. 

그들의 제품은 저전력 장치에서 고성능 Inference을 위해 [Edge TPU](https://coral.ai/docs/edgetpu/faq/#what-is-the-edge-tpus-processing-speed)를 활용합니다. 

Coral은 또한 이미지 분할, 포즈 추정, 음성 인식 등을 위해 미리 Compile된 Model을 제공하여 자체 Local AI System을 만들려는 개발자에게 스캐폴딩을 제공합니다. 

Model을 만드는 필수 단계는 아래 순서도에서 볼 수 있습니다.

<br>

<p align="center">
  <img src="/assets/PyTorch_vs_Tensorflow/04.png">
</p>

<br>

### 4.2.7. TensorFlow.js

[TensorFlow.js](https://www.tensorflow.org/js)는 Node.js를 사용하여 Browser와 Server 측 모두에서 Model을 교육하고 배포할 수 있는 Machine Learning용 JavaScript Library입니다.    

그들은 Python Model을 가져오는 방법에 대한 예제와 정보가 포함된 [문서](https://www.tensorflow.org/js/tutorials), 즉시 사용할 수 있는 [Pre-Trained Model](https://www.tensorflow.org/js/models), 관련 Code가 있는 [라이브 데모](https://www.tensorflow.org/js/demos)를 제공합니다.

<br>

### 4.2.8. Cloud

[TensorFlow Cloud](https://www.tensorflow.org/cloud)는 Local 환경을 Google Cloud에 연결할 수 있는 Library입니다. 

제공된 API는 Cloud Console을 사용할 필요 없이 Local Machine의 Model Build 및 Debugging에서 GCP의 분산 학습 및 Hyperparameter 조정에 이르는 격차를 해소하도록 설계되었습니다.

<br>

### 4.2.9. Colab

[Google Colab](https://colab.research.google.com/?utm_source=scs-index)은 Jupyter와 매우 유사한 Cloud 기반 Notebook 환경입니다. 

GPU 또는 TPU Train을 위해 Colab을 Google Cloud에 쉽게 연결할 수 있습니다. 

PyTorch는 Colab에서도 사용할 수 있습니다.

<br>

### 4.2.10. Playground

[Playground](https://playground.tensorflow.org/#activation=tanh&batchSize=10&dataset=circle&regDataset=reg-plane&learningRate=0.03&regularizationRate=0&noise=0&networkShape=4,2&seed=0.79936&showTestData=false&discretize=false&percTrainData=50&x=true&y=true&xTimesY=false&xSquared=false&ySquared=false&cosX=false&sinX=false&cosY=false&sinY=false&collectStats=false&problem=classification&initZero=false&hideText=false)는 신경망의 기초를 이해하기 위한 작지만 세련된 교육 도구입니다. 

깔끔한 UI 내에서 시각화된 단순한 조밀한 Network를 제공합니다. Network의 Layer 수와 크기를 변경하여 기능을 학습하는 방법을 실시간으로 확인할 수 있습니다. 

또한 Learning Rate 및 정규화 강도와 같은 Hyperparameter 변경이 다양한 Data Set의 학습 Process에 어떤 영향을 미치는지 확인할 수 있습니다. 

Playground를 사용하면 학습 과정을 실시간으로 재생하여 교육 과정에서 입력이 어떻게 변환되는지 매우 시각적으로 볼 수 있습니다. 

Playground에는 Network의 기본 사항을 이해할 수 있도록 구축된 작은 [Open Source 신경망 Library](https://github.com/tensorflow/playground/blob/master/src/nn.ts)도 함께 제공됩니다.

<br>

### 4.2.11. Datasets

[Google Research의 Datasets](https://research.google/tools/datasets/)는 Google이 주기적으로 DataSet를 출시하는 DataSet Resource입니다. 

Google은 또한 더 광범위한 DataSet Database에 Access할 수 있는 [DataSet 검색](https://datasetsearch.research.google.com/)을 제공합니다. 

물론 PyTorch 사용자도 이러한 Data Set를 활용할 수 있습니다.

<br>
<br>

## 4.3. Ecosystems - Final Words

이 Round는 세 가지 중 가장 가깝지만 궁극적으로 TensorFlow는 우수한 Ecosystem를 가지고 있습니다. 

Google은 엔드 투 엔드 Deep Learning Workflow의 각 관련 영역에서 사용 가능한 제품이 있는지 확인하는 데 많은 투자를 했지만 이러한 제품이 얼마나 잘 다듬어진지는 이 환경에 따라 다릅니다. 

그럼에도 불구하고 TFX와 함께 Google Cloud와의 긴밀한 통합으로 종단 간 개발 Process가 효율적이고 조직화되었으며 Google Coral 기기로 Model을 쉽게 이식할 수 있으며 일부 산업에서는 TensorFlow가 압도적인 승리를 거두었습니다.

PyTorch 대 TensorFlow 논쟁의 3라운드는 TensorFlow의 승리입니다.

<br>
<br>
<br>


# 5. PyTorch 또는 TensorFlow를 사용해야 하나요?

예상대로 PyTorch 대 TensorFlow 논쟁에는 정답이 없습니다. 

특정 사용 사례와 관련하여 한 Framework가 다른 Framework보다 우수하다고 말하는 것이 합리적입니다. 

어떤 Framework가 가장 적합한지 결정하는 데 도움이 되도록 권장 사항을 아래의 순서도에 정리했으며 각 차트는 서로 다른 관심 영역에 맞게 조정되었습니다.

<br>

## 5.1. What if I’m in Industry?

<br>

<p align="center">
  <img src="/assets/PyTorch_vs_Tensorflow/05.png">
</p>

<br>

산업 환경에서 Deep Learning Engineering을 수행하는 경우 TensorFlow를 사용하고 있을 가능성이 높으며 계속 사용해야 합니다. 

TensorFlow의 강력한 배포 Framework와 종단 간 TensorFlow Extended Platform은 Model을 생산해야 하는 사람들에게 매우 중요합니다. 

Model Monitoring 및 아티팩트 추적과 함께 gRPC Server에 쉽게 배포하는 것은 업계에서 사용하기 위한 중요한 도구입니다. 

TorchServe의 최근 Release에서는 PyTorch에서만 사용할 수 있는 SOTA Model에 Access해야 하는 경우와 같이 합당한 이유가 있는 경우 PyTorch 사용을 고려할 수 있습니다. 

이 경우 ONNX를 사용하여 TensorFlow의 배포 Workflow 내에서 변환된 PyTorch Model을 배포하는 것을 고려하십시오.


Mobile Application을 Build하는 경우 Audio 또는 Video 입력이 필요한 경우가 아니면 TensorFlow를 사용해야 하는 경우를 제외하고 PyTorch Live의 최근 Release를 고려하여 PyTorch 사용을 고려할 수 있습니다. 

AI를 활용하는 Embedded System 또는 IoT 장치를 구축하는 경우 TFLite + Coral Pipeline을 감안할 때 여전히 TensorFlow를 사용해야 합니다.

**결론: 하나의 Framework를 선택해야 한다면 TensorFlow를 선택하십시오.**

<br>

## 5.2. What if I’m a Researcher?

<br>

<p align="center">
  <img src="/assets/PyTorch_vs_Tensorflow/06.png">
</p>

<br>

연구원이라면 거의 확실하게 PyTorch를 사용하고 있으며 현재로서는 계속 사용하고 있을 것입니다. 

PyTorch는 사실상의 연구 Framework이므로 대부분의 SOTA Model은 PyTorch 전용입니다.


이 규칙에는 몇 가지 주목할만한 예외가 있으며, 가장 주목할만한 것은 Reinforcement Learning에 있는 사람들이 TensorFlow 사용을 고려해야 한다는 것입니다. 

TensorFlow에는 Reinforcement Learning을 위한 [기본 에이전트](https://www.tensorflow.org/agents/overview) Library가 있으며 DeepMind의 [Acme](https://github.com/deepmind/acme) Framework는 TensorFlow에서 구현됩니다. 

[OpenAI의 Gym](http://www.gymlibrary.ml/)은 TensorFlow 또는 PyTorch와 함께 사용할 수 있지만 OpenAI의 [Baselines](https://github.com/openai/baselines) Model 저장소는 TensorFlow에서 구현됩니다. 

연구에 TensorFlow를 사용할 계획이라면 DeepMind의 [Sonnet](https://github.com/deepmind/sonnet)에서 더 높은 수준의 추상화도 확인해야 합니다.

TensorFlow를 사용하고 싶지 않다면 TPU 교육을 하고 있다면 Google의 JAX 탐색을 고려해야 합니다. 

신경망 Framework 자체는 아니지만 자동 분화 기능이 있는 G/TPU용 NumPy 구현에 더 가깝습니다. 

"Sonnet for JAX"라고 부르는 DeepMind의 [Haiku](https://github.com/deepmind/dm-haiku)는 JAX를 고려하고 있다면 살펴볼 가치가 있는 JAX를 기반으로 구축된 신경망 Library입니다. 

또는 Google의 [Flax](https://github.com/google/flax)를 확인할 수 있습니다. 

TPU Train을 하지 않는다면 지금은 PyTorch를 사용하는 것이 가장 좋습니다.


어떤 Framework를 선택하든 2022년 JAX를 주시해야 합니다. 

특히 커뮤니티가 성장하고 더 많은 출판물에서 이를 활용하기 시작함에 따라 더욱 그렇습니다.

**결론: 하나의 Framework를 선택해야 하는 경우 PyTorch를 선택하십시오.**

<br>
<br>

## 5.3. What if I’m a Professor?

<br>

<p align="center">
  <img src="/assets/PyTorch_vs_Tensorflow/07.png">
</p>

<br>

교수라면 Deep Learning 과정에 사용할 Framework는 교육 과정의 목표에 따라 달라집니다. 

코스의 초점이 Deep Learning 이론뿐만 아니라 전체 Deep Learning Process에서 역량을 발휘하여 기초를 다질 수 있는 Deep Learning Engineer를 양성하는 것이라면 TensorFlow를 사용해야 합니다. 

이 경우 종단 간 실습 Project와 함께 TensorFlow Ecosystem 및 도구에 대한 노출은 매우 유익하고 가치가 있습니다.


코스의 초점이 Deep Learning 이론과 Deep Learning Model의 기본 이해에 있다면 PyTorch를 사용해야 합니다. 

이는 특히 고급 학부 과정이나 Deep Learning 연구를 수행할 수 있도록 학생들을 준비시키는 초기 대학원 수준 과정을 가르치는 경우에 해당됩니다.


이상적으로는 학생들이 각 Framework에 노출되어야 하며 한 학기라는 시간적 제약에도 불구하고 Framework 간의 차이점을 이해하는 데 시간을 할애하는 것이 가치가 있을 것입니다. 

코스가 서로 다른 주제에 전념하는 많은 클래스가 있는 더 큰 Machine Learning 프로그램의 일부인 경우 두 가지 모두에 노출하려고 하는 것보다 코스 자료에 가장 적합한 Framework를 고수하는 것이 더 나을 수 있습니다.

<br>

## 5.4. What if I’m Looking for a Career Change?

<br>

<p align="center">
  <img src="/assets/PyTorch_vs_Tensorflow/08.png">
</p>

<br>

경력을 바꾸고 싶다면 PyTorch나 TensorFlow가 좋은 선택입니다. 

이 경우에 할 수 있는 가장 중요한 것은 즉시 가치를 가져올 수 있다는 것을 입증하는 것이므로 Project Portfolio를 보유하는 것이 중요합니다. 

창의적인 사용 사례에 Deep Learning을 적용하거나 Project를 끝까지 수행하여 업계 준비가 되어 있음을 보여줌으로써 평범한 것을 넘어 당신을 매우 유리한 위치에 놓을 것입니다.


따라서 보다 쉽게 작업할 수 있는 Framework를 사용하십시오. 보다 직관적인 Framework를 사용하면 Portfolio를 효율적으로 구축할 수 있습니다. 

이는 특정 Framework의 API에 익숙해지는 것보다 훨씬 더 중요합니다. 

즉, 완전히 Framework에 구애받지 않는 경우 선호되는 산업 Framework인 TensorFlow를 사용하십시오. 

아래 그래프의 각 Framework에 대해 다양한 직업 WebSite의 채용 공고 수를 집계했으며 TensorFlow가 PyTorch를 크게 압도했습니다.

<br>

<p align="center">
  <img src="/assets/PyTorch_vs_Tensorflow/09.png">
</p>

<br>

**결론: OpenAI 또는 TensorFlow에서 일하는 목표가 절망적으로 직관적이지 않은 경우와 같이 PyTorch를 사용해야 하는 특별한 이유가 있다면 자유롭게 사용하십시오. 
하지만 TensorFlow에 집중하는 것이 좋습니다.**

<br>

## 5.5. What if I’m a Hobbyist?

<br>

<p align="center">
  <img src="/assets/PyTorch_vs_Tensorflow/10.png">
</p>

<br>

Deep Learning에 관심이 있는 취미 생활자라면 사용하는 Framework는 목표에 따라 다릅니다.

더 큰 Project의 일부로 Deep Learning Model을 구현하는 경우 특히 IoT/Embedded 장치에 배포하는 경우 TensorFlow를 사용하고 싶을 것입니다. 

PyTorch Live가 출시되면 Mobile Application에 PyTorch를 사용할 수 있지만 현재로서는 TensorFlow + TFLite가 여전히 선호되는 방법론입니다.


목표가 Deep Learning 자체를 배우는 것이라면 이 경우에 가장 적합한 Framework는 배경에 따라 다릅니다. 

일반적으로 PyTorch는 아마도 여기서 더 나은 옵션일 것입니다. 

특히 Python 작업에 익숙하다면 더욱 그렇습니다. 

Deep Learning에 대해 이제 막 배우기 시작한 완전 초보자라면 다음 Section을 참조하세요.

<br>

## 5.6. What if I’m a Total Beginner?

<br>

<p align="center">
  <img src="/assets/PyTorch_vs_Tensorflow/11.png">
</p>

<br>

Deep Learning에 관심이 있고 막 시작하려는 완전 초보자라면 Keras를 사용하는 것이 좋습니다. 

높은 수준의 구성 요소를 사용하면 Deep Learning의 기본 사항을 쉽게 이해할 수 있습니다. 

Deep Learning의 기본 사항을 더 철저히 이해할 준비가 되면 다음과 같은 몇 가지 옵션이 있습니다.


새 Framework를 설치하고 싶지 않고 귀하의 역량이 새 API로 얼마나 잘 변환될지 걱정된다면 Keras에서 TensorFlow로 "Dropdown"을 시도할 수 있습니다. 

배경에 따라 TensorFlow가 혼란스러울 수 있습니다. 이 경우 PyTorch로 이동해 보십시오.


기본적으로 Python처럼 느껴지는 Framework를 원한다면 PyTorch로 이동하는 것이 가장 좋은 방법일 수 있습니다. 

이 경우 새 Framework를 설치하고 잠재적으로 사용자 지정 스크립트를 다시 작성해야 합니다. 

또한 PyTorch가 다소 번거롭다면 PyTorch Lightning을 사용하여 Code를 구획화하고 일부 상용구를 제거할 수 있습니다.


완전한 초보자라면 TensorFlow와 PyTorch 모두에서 YouTube 자습서를 보고 어떤 Framework가 더 직관적으로 느껴지는지 결정하는 것이 좋습니다.


**마지막으로**

보시다시피 PyTorch 대 TensorFlow 논쟁은 지형이 끊임없이 변화하는 미묘한 차이가 있으며 오래된 정보로 인해 이러한 상황을 이해하기가 훨씬 더 어렵습니다. 

2022년에는 PyTorch와 TensorFlow 모두 매우 성숙한 Framework이며 핵심 Deep Learning 기능이 크게 겹칩니다. 

오늘날에는 Model Availability, 배포 시간 및 관련 Ecosystem과 같은 각 Framework의 실질적인 고려 사항이 기술적인 차이점을 대체합니다.


두 Framework 모두 좋은 문서, 많은 학습 Resource 및 활성 커뮤니티를 가지고 있기 때문에 두 Framework를 선택하는 데 실수가 없습니다. 

PyTorch는 연구 커뮤니티에서 폭발적으로 채택된 후 사실상의 연구 Framework가 되었고 TensorFlow는 Legacy 산업 Framework로 남아 있지만 두 Domain 모두에서 각각에 대한 사용 사례가 확실히 있습니다.

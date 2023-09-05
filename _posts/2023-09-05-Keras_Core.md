---
title: "Keras Core"
date: 2023-09-04 08:26:28 -0400
categories: Deep Learning
---

# Keras Core

<br>
<br>
<br>

## 0. Keras란 무엇인가?   

<br>
<br>

### 0.0. 정의   

<br>

* Keras는 딥러닝 Model을 구축하고 훈련하기 위한 고수준의 인터페이스를 제공하는 오픈 소스 소프트웨어 라이브러리입니다.  

<br>  

* 2015년에 François Chollet이 빠르게 Model을 돌려볼 수 있도록 하기 위해서 User Friendly한 API를 제공하는 것을 목적으로 최초로 개발되었고,
당시 Theano , Microsoft Cognitive Toolkit (CNTK) 및 TensorFlow Backend를 지원하게 됨으로써 점점 Deep Learning 사용자들 사이에서 인기를 얻기 시작했습니다.  

<br>
  
* 2017년에는 Tensorflow가 공식 High Level API로 채택하였고, 이로 인해 tf.keras로 정식으로 Tensorflow 내부에 포함되었습니다.

<br>

* 하지만, 이 때부터 Keras는 Tensorflow 전용으로 개발하게 됩니다. 이런 결정을 내린 이유는 당시에는 Theano와 CNTK는 개발이 중단되었기 때문에 Tensorflow만이 유일한 Framework이었습니다.

<br>

* 2019년 Tensorflow 2.0이 발표되면서 더욱더 입지를 확고해 했지만, Tensorflow 이외의 Framework를 지원하지 않게 되었습니다.

<br>
<br>
<br>

### 0.1. 장점

<br>

**User-friendly**  
Keras는 기본적으로 High Level API를 지원하기 때문에 개발자의 작업 부담을 줄이고 큰 그림에 집중할 수 있게끔 설계되었습니다. 
사용하기 쉽고 디버깅이 빠르며 코딩이 우아하고 간결하며 유지 보수가 쉽고, 쉽게 배포할 수 있게 설계되었습니다.

<br>

**Adaptable**  
간단하게 시작하여 점차 복잡성을 펼쳐 보여주는 것을 선호합니다. 기본적인 작업을 빠르고 쉽게 만들고 보다 복잡한 작업을 단계별로 수행할 수 있게끔 하는 것이 목표로 만들어 졌기 때문입니다.

<br>

**Powerful**  
전문가 수준의 성능과 확장성을 제공합니다.

<br>

**Modularity**  
Keras는 거의 모든 것이 모듈로 구성되어 있습니다. 이 모듈들은 서로 독립적이며, 개별 설정이 가능하며, 최대한 확장성을 가집니다. 
예를 들어, Dense Layer, Loss Function, Optimizer, 초기화 방식, Activation Function, 정규화 방식 등이 모두 독립적인 모듈로 취급됩니다.

<br>

**Multi Backend**  
기본적으로 TensorFlow를 Backend로 사용하지만, Theano나 Microsoft Cognitive Toolkit (CNTK)와 같은 다른 딥러닝 Framework를 Backend로 설정할 수도 있습니다. (단, 최근에는 TensorFlow에 집중되고 있음)

<br>

**Pre-Trained Models**  
VGG, ResNet, Inception 등과 같은 다양한 사전 훈련된 Model을 제공합니다. 
이 Model들을 사용하여 전이 학습을 쉽게 수행할 수 있습니다.
또한, 새로운 Model이 발표되면 빠르게 추가됩니다.

<br>

**Utility for Train & Evaluation**  
Model Training, Evaluation, Inference을 위한 다양한 도구와 유틸리티를 제공합니다. 
예를 들어, Data Augmentation, Checkpoint Save, Early Stop 등의 기능을 지원합니다.

<br>

**Scalability**  
User Defined Layer, Loss Function, Metric, Optimizer를 쉽게 추가할 수 있도록 설계되어 있습니다. 
이는 고급 사용자들이 자신만의 구성 요소를 쉽게 개발할 수 있게 합니다.   

<br>
<br>
<br>

## 1. Keras Core란 무엇인가?

<br>
<br>

### 1.0. 정의

<br>

* 이와 같이 Tensorflow에 흡수되면서 원래의 개발 의도를 상실해 버린 Keras가 올해 2023년 가을에 Keras Core(Keras 3.0)라는 이름으로 출시된다고 하네요.  

<br>

* 이번 Keras Core는 TensorFlow, JAX, PyTorch Framework를 지원한다고 합니다.  

<br>

* 아래는 Keras Core 발표문의 일부입니다. ( https://keras.io/keras_core/announcement/ )  
 
 **"Keras Core는 multi-backend로의 회귀입니다. 
 얼마 전만 해도, Keras는 Theano, TensorFlow, 그리고 CNTK(심지어 MXNet까지!) 위에서 실행될 수 있었습니다. 
 2018년에, 우리는 TensorFlow 전용으로 Keras 개발에 집중하기로 결정했습니다. 
 당시에는, TensorFlow만이 유용한 옵션이었습니다: Theano와 CNTK는 개발을 중단했습니다. 
 여러 Backend를 지원하기 위한 추가 비용은 더 이상 그만한 가치가 없었습니다.**

 **그러나 2023년에는 이러한 상황이 사라졌습니다. 
 2023 대규모 개발자 설문조사에 따르면, TensorFlow는 55%에서 60%의 시장 점유율을 가지고 있으며, 생산 ML에 있어서 최상의 선택입니다. 
 동시에, PyTorch는 40%에서 45%의 시장 점유율을 가지고 있으며 ML 연구에 있어서 최상의 선택입니다. 
 동시에, JAX는 훨씬 더 작은 시장 점유율을 가지고 있지만, Google DeepMind, Midjourney, Cohere 등 생성적 AI 분야의 주요 플Layer들에게 받아들여졌습니다.**

 **우리는 이러한 각 Framework가 다양하고 중요한 가치를 제공한다고 믿고 있기 때문에 우리가 만든 것은 이 세 가지 모두를 한 번에 활용할 수 있게 할 것입니다."**

<br>
<br>

### 1.1. 장점

<br>

새로운 다중 Backend Keras Core를 사용하는 추가적인 이점은 무엇일까요?   

<br>

* **Always get the best performance for your models**  
  - Keras Core를 사용하면 JAX를 사용하는 경우에 GPU, TPU, 그리고 CPU에서 최상의 Train & Inference 성능을 제공한다고 합니다.
    Model마다 차이가 있기는 하지만, non-XLA TensorFlow가 GPU에서 가끔 더 빠르게 동작하기도 한다고 하네요.

<br>

* **Maximize available ecosystem surface for your models**
  - Keras Core로 만든 모든 Model은 PyTorch Module로 인스턴스화될 수 있고, TensorFlow SavedModel로 저장할 수 있으며, Stateless JAX 함수로 인스턴스화될 수 있다고 합니다.
  - 이 말은 Keras Core로 Model을 만들면 어떤 Framework을 사용하는 환경에서든지 사용 가능한 Model을 생성할 수 있다는 의미입니다.

<br>

* **Maximize distribution for your open-source model releases.**
  - 위와 일맥상통한 내용인 것 같은데, Keras Core로 만든 Pre-Trained Model은 Framework과 관계없이 사용가능하다는 내용입니다.

<br>

* **Use data pipelines from any source**
  - Keras Core의 fit()/evaluate()/predict() Routine에 Input으로 들어갈 Dataset Format으로 각 Framework 고유의 Dataset 객체 뿐만 아니라,
Numpy Array , Pandas Dataframe을 사용할 수 있습니다.
  - 즉, Dataloader로 Pytorch를 사용하고, Train은 Tensorflow Model을 사용하는 것이 가능해 졌다는 의미입니다. 
  - 개인적으로 이 기능은 꽤나 유용할 것 같네요.            

<br>
<br>

### 1.2. 특징

<br>
<br>

#### 1.2.0. The full Keras API, available for TensorFlow, JAX, and PyTorch

<br>

* Keras Core는 이전의 Keras API를 모두 구현하고 있고, 이 API들은 모두 TensorFlow, JAX, PyTorch와 함께 사용할 수 있습니다.  
  
  
* 이 API들은 백여 개의 Layer, 수십 개의 Metric, Loss Function, Optimizers, Callback, Keras Train & Evaluation Loop, 그리고 Keras Save & Serialize 기능 등이 있습니다.  


* 내장 Layer만을 사용하는 모든 Keras Model은 지원되는 모든 Backend에서 즉시 작동합니다. 


* 사실, 내장 Layer만을 사용하는 기존의 tf.keras Model은 keras를 keras_core로 가리키는 import로 변경할 때 JAX와 PyTorch에서 바로 실행될 수 있습니다.      

<br>
<br>
<p align="center">
  <img src="/assets/Keras_Core/pic_00.jpg">
</p>
<br>
<br>   

#### 1.2.1. A cross-framework low-level language for deep learning

<br>

Keras Core를 사용하면 모든 Framework에서 동일하게 작동하는 Component(예: 임의의 사용자 정의 Layer나 Pre-Trained Models)를 생성할 수 있습니다.  

특히, Keras Core는 모든 Backend에서 작동하는 keras_core.ops Namespace에 접근할 수 있는데, 다음과 같은 기능이 들어가 있습니다.   

<br>

* **A near-full implementation of the NumPy API**
  - NumPy API의 완벽하게 구현되어 있습니다. "NumPy와 비슷한" 것이 아니라, 그냥 실제로 NumPy API입니다. 

<br>

* **A set of neural network-specific functions that are absent from NumPy, such as ops.softmax, ops.binary_crossentropy, ops.conv, etc.**  
  - NumPy에서는 없는 신경망 특화 함수(ops.softmax, ops.binary_crossentropy, ops.conv) 집합이 포함되어 있습니다. 
  - keras_core.ops에서 ops만을 사용하면, 사용자 정의 Layer, 사용자 정의 Loss, 사용자 정의 Metric, 사용자 정의 Optimizer는 JAX, PyTorch, TensorFlow에서 같은 Code로 작동합니다. 
  - 즉, 하나의 Component 구현(예: 단일 model.py와 함께하는 단일 Checkpoint 파일)만 유지하면 됩니다. 
  - 그리고 모든 Framework에서 그것을 사용할 수 있으며, 정확히 같은 수치로 작동합니다.      

<br>
<br>
<p align="center">
  <img src="/assets/Keras_Core/pic_01.jpg">
</p>
<br>
<br>      
<br>
   

#### 1.2.2. Seamless integration with native workflows in JAX, PyTorch, and TensorFlow

<br>

* 오래된 Multi Backend Keras 1.0과 달리, Keras Core는 Keras Model, Keras Optimizer, Keras Loss 및 Metric을 정의하고 fit()/evaluate()/predict()를 호출하는 Keras 중심의 Workflow만을 위한 것이 아닙니다.  

<br>

* Keras Core는 JAX Train Loop, TensorFlow Train Loop 또는 PyTorch Train Loop에서 Keras Model(또는 Loss이나 Metric과 같은 다른 Component)을 사용하여 Backend 네이티브 Workflow와 원활하게 작동하도록 설계되었습니다. 

<br>

**Keras Core는 다음과 같은 저수준 구현 유연성을 제공합니다.**

* optax Optimizer, jax.grad, jax.jit, jax.pmap을 사용하여 Keras Model을 훈련하기 위한 저수준 JAX Train Loop를 작성합니다.  

<br>
  
* tf.GradientTape 및 tf.distribute를 사용하여 Keras Model을 훈련시키기 위한 저수준 TensorFlow Train Loop를 작성합니다.  

<br>

* torch.optim Optimizer, torch Loss Function 및 torch.nn.parallel.DistributedDataParallel 래퍼를 사용하여 Keras Model을 훈련시키기 위한 저수준 PyTorch Train Loop를 작성합니다.  

<br>

* Keras Layer나 Model을 torch.nn.Module의 일부로 사용합니다. 이는 PyTorch 사용자가 Keras API를 사용하든 사용하지 않든 Keras Model을 활용할 수 있다는 것을 의미합니다. Keras Model을 다른 PyTorch Module처럼 처리할 수 있습니다.  

<br>
<br>
<p align="center">
  <img src="/assets/Keras_Core/pic_02.jpg">
</p>
<br>
<br>         
<br>
<br>


#### 1.2.3. Support for cross-framework data pipelines with all backends

<br>

* Keras Core Model은 JAX, PyTorch, 또는 TensorFlow Backend를 사용하더라도 다양한 Data pipeline을 사용하여 Train 할 수 있습니다.
  - tf.data.Dataset pipelines: 확장 가능한 생산 ML을 위한 참조.  
  - torch.utils.data.DataLoader 객체.
  - NumPy 배열 및 Pandas dataframes.
  - keras_core.utils.PyDataset 객체.   

<br>
<br>
<br>

#### 1.2.4. Pretrained models

<br>

* Keras Core와 함께 사용할 수 있는 다양한 Pre-Trained Model들이 준비되어 있습니다.  

* 40개의 Keras Applications Model 전체(keras_core.applications Namespace)가 모든 Backend에서 사용 가능합니다.  
  ( PyTorch와의 평균 풀링에서 비대칭 패딩 지원 부족으로 인해 아키텍처적으로 호환되지 않는 Model 1개 제외 )  
  
* KerasCV와 KerasNLP의 다양한 사전 훈련된 Model들(예: BERT, T5, YOLOv8, Whisper 등) 또한 모든 Backend에서 작동합니다.   

<br>
<br>

#### 1.2.5. Progressive disclosure of complexity

<br>

* Progressive disclosure of complexity는 Keras API의 핵심 디자인 원칙입니다. 


* Progressive disclosure of complexity의 의미는 Model 개발 초기에는 Keras가 제공하는 High Level의 API를 이용해서 빠르게 Prototyping을 할 수 있습니다.  


* 그 이후에 세밀하게 Model의 구조를 조정하거나 Customizing이 필요한 경우 또는 유연성을 가미해야 하는 경우에 점진적으로 조금씩 복잡한 설정을 할 수 있도록 구조가 되어 있다는 의미입니다.  


* 개발자들의 요구 사항이 구체적이고 복잡하게 바뀌어도 기존에 만든 Model Code를 거의 그대로 사용하면서 조금씩 복잡한 Code들을 사용하여 Model을 만들 수 있다는 의미입니다.  


* 소개글에서는 '복잡성의 절벽에 갑자기 떨어지지 않도록 한다'라는 표현을 썼더군요. 적절한 비유 같습니다.  


* 이러한 원리가 PyTorch와 TensorFlow에서는 다음과 같이 작동합니다:   

<br>
<br>
<p align="center">
  <img src="/assets/Keras_Core/pic_03.jpg">
</p>
<br>
<br>            
<br>
<br>   

#### 1.2.6. A new stateless API for layers, models, metrics, and optimizers

* 함수형 프로그래밍을 선호하는 개발자들을 위해서 Keras Componets의 stateless API가 추가되었다고 합니다.

<br>
<br>

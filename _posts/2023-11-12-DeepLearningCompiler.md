---
title: "Deep Learning Compiler"
date: 2023-09-06 08:26:28 -0400
categories: Deep Learning
---

# Deep Learning Compiler

<br>
<br>

## 0. Introduction   

<br>
<br>

<img src="https://moonlight314.github.io/assets/DeepLearningCompiler/pic_00.png">

<br>
<br>

우리는 다양한 Deep Learning Framework으로 Model을 만듭니다.



우리가 만든 Model이 실제로 실행되는 환경들은 매우 다양합니다.



PC일수도 있고, Edge Device일수도 있고, Mobile Device, Cloud 환경 등등 매우 다양한 Target Device가 존재합니다.



이런 Target Device의 다양화에 어느 정도 대응할 수 있도록 해주는 것이 ONNX입니다.



ONNX(Open Neural Network Exchange, Link)는 Deep Learning Framework간의 Model Conversion이 주된 목적입니다.



이를 잘 활용하여 우리의 Deep Learning Model을 Target Device에서 작동할 수 있도록 할 수 있습니다.





여기에 추가로 이 Post에서 설명하려고 하는 Deep Learning Compiler는 특정 Deep Learning Framework Model을



Target Device에서 동작하도록 Conversion해 줄 뿐만 아니라, 다양한 최적화 기법을 적용하여 Inference 속도도 향상시켜주는 역할을 합니다.

<br>
<br>
<br>

## 1. Architecture   

<br>
<br>

Deep Learning Compiler는 어떤 방식으로 동작하는지 알아보도록 하겠습니다.



Deep Learning Compiler는 일반적인 Compiler와 어느 정도 비슷하며, 일반적인 Compiler와 마찬가지로



Frontend와 Backend로 구성되어 있습니다.



Frontend는 Software적인 최적화에 중점을 두고 있고, Backend는 Target Device에 맞게 Hardware 최적화에 중점을 두고 있습니다.



Frontend와 Backend 둘 다 IR(Intermediate Representation)이 있습니다.



Frontend 쪽의 IR은 High Level IR 혹은 Graph IR 이라고 말하고, Backend 쪽의 IR은 Low Level IR 혹은 Operator IR 이라고 말합니다.   

<br>
<br>

### 1.0. Frontend

* Deep Learning Compiler의 Frontend는 크게 2가지 동작을 한다고 볼 수 있습니다.   

<br>
<br>

#### 1.0.0. Conversion

* Conversion Part는 우리가 다양한 Deep Learning Framework으로 Train 시킨 Model을 입력으로 받아서 High-Level IR로 변환합니다..   

* High-Level IR에는 Model이 계산을 어떻게 하며 Flow Control을 어떻게 하는가에 대한 정보가 들어가 있습니다.

* High-Level IR은 Model의 구조를 파악하는 단계이기 때문에 아직은 Hardware에 독립적이며 Data와 Operator간의 관계를 확립하는 목표입니다.   

<br>
<br>

#### 1.0.1. Optimization   

* Conversion 단계에서 파악한 Model의 구조를 기반으로 최적화를 수행할 수 있는 여지를 찾아서 적용함으로써 성능 향상을 시킵니다.   

<br>
<br>

### 1.1. Backend   

Backend는 다음과 같은 기능을 합니다.   

<br>

#### 1.1.0. Conversion   

* Frontend에서 생성된 High-Level IR을 입력으로 받아서, Low-Level IR로 변환합니다.  

* Backend의 주된 역할은 Hardware에 독립적이었던 High-Level IR을 Target Device의 Hardware 특성을 반영하여 변환하는 것입니다.      

<br>
<br>

#### 1.1.1. Optimization   

<br>

* 이 단계에서는 Hardware 최적화가 적용됩니다.합니다.   

* 특히 이 단계에서는 다양한 최적화 기법, 예를 들면, intrinsic mapping, memory allocation and fetching, memory latency hiding, parallelization, 
loop oriented optimizations 등과 같은 기법이 적용됩니다.

* 최종적으로 최적화된 Low-Level IR은 Just-In-Time compiler 혹은 Ahead-Of-Time compiler 같은 Compiler를 사용해 Hardware-Specific Code를 생성합니다.         

<br>
<br>

## 2. Optimizations

앞에서 Frontend와 Backend에서 각각 최적화가 이루어진다고 했는데, 구체적으로 어떤 기법들이 사용되는지 알아보도록 하겠습니다.   

<br>
<br>

### 2.0. Frontend   

<br>

Frontend에서 수행되는 최적화는 Software 수준에서 수행되는 computational graph optimization입니다.



NOP 라든지 계산결과가 항상 같은 값이거나, 계산 수행이 다음 계산에 영향을 주지 않는 경우에는 최적화를 할 수 있겠죠.



또한, Block Level에서도 최적화가 이루어집니다.



교환법칙, 결합법칙 및 분배법칙 등의 대수학적인 기법을 이용하여 계산을 단순화하여 최적화를 이루는 방법입니다.



마지막으로, Dataflow Level 최적화를 수행합니다.

 - **Dead Code Elimination** : 계산 결과에 영향을 주지 않는 부분을 제거

 - **Static Memory Planning** : 메모리를 효율적 사용할 수 있게 해줌   

<br>
<br>

### 2.1. Backend

<br>

Backend Optimization은 Hardware에 따라 다르게 적용되며, 다음과 같은 기법들이 있습니다.

 - efficient memory allocation and mapping

 - better data reuse by the method of loop fusion and sliding window.

 - auto-tuning

   * 일련의 매개변수를 최적으로 선택

   * 병렬화를 통해 성능을 가속화

   * 검색 공간에 유전 알고리즘(genetic algorithms)을 적용함으로써 검색 시간을 줄입니다

<br>
<br>
<br>

## 3. Deep Learning Compilers

널리 쓰이고 있는 Deep Learning Compiler 몇 가지를 살펴보도록 하겠습니다.   

<br>
<br>

### 3.1. TVM   

<br>

#### 3.1.0. Overview   

<br>   

<img src="https://tvm.apache.org/assets/images/about-image.svg">

   

* TVM은 CPU , GPU 그 외의 다른 Deep Learning Accelerator들에 맞게 Model을 Compile하는 Framework입니다.
* 다양한 Hardware Backend에 맞게 Model을 최적화하고 효율적으로 계산하도록 만들어 줍니다.   

<br>
<br>

#### 3.1.1. Conversion & Optimization

<br>

* TVM이 Model을 최적화하고 Machine Code를 생성하는 방법을 소개하도록 하겠습니다.   

<img src="https://raw.githubusercontent.com/apache/tvm-site/main/images/tutorial/overview.png">

<br>   

 1. 변환하고자 하는 Tensorflow / PyTorch / Onnx File을 입력 받습니다.


 2. TVM의 High Level Model Language인 'Relay' 형태로 변환합니다. 

 
 3. TE(Tensor Expression) 형태로 Lowering을 합니다.  Lowering이란 High-Level Representation에서 Low-Level Representation으로 바꾸는 것을 말합니다.
 High-Level Optimizations을 수행하고 난 후에, Relay는 FuseOps이라는 것을 수행하여 여러개의 Subgraph로 쪼개고, 여러개의 Subgraph를 TE로 Lowering합니다.
 TE는 Tensor Computations을 나타내는 Domain-Specific Language이며, Tiling, Vectorization, Parallelization, Unrolling, Fusion과 같은
 Low-Level Loop Optimizations도 수행합니다.

 4. AutoTVM 이나 AutoScheduler과 같은 Auto-Tuning Module을 이용하여 최적의 Schedule을 찾습니다.

 5. Compile하기 전에 최적의 설정을 찾습니다. Tuning이 끝나면 Auto-Tuning Module은 개별 Subgraph의 최고의 Schedule을 찾습니다.

 6. 개별 Subgraph는 TVM의 Low-Level Intermediate Representation인 TIR ( Tensor Intermediate Representation )로 Lowering되고 Low-Level Optimization이 수행됩니다. 
 마지막으로, TIR은 각 개별 Hardware에 맞게 Target Compiler로 Lowering됩니다. 
 TVM이 지원하는 Backend 종류에는 다음과 같은 것이 있습니다.
     - **LLVM** :  Standard x86 , ARM processors, AMDGPU , NVPTX로 Code Generation
     - **NVCC** : NVIDIA Compiler
     - **TVM’s Bring Your Own Coegen (BYOC)**

 7. Machine Code로 Compile합니다.   

<br>
<br>

#### 3.1.2. Install

* TVM은 다양한 방법의 Install 방식을 제공합니다. 
* 구체적인 방법은 아래 Link를 참고하시기 바랍니다
  
  https://tvm.apache.org/docs/install/index.html   

<br>
<br>

#### 3.1.3. Tutorial

* 아래의 Link를 참고하면 우리의 Model을 TVM을 이용해서 각 Backend HW에 적합한 Code로 변환하는 일련의 순서를 Python API를 이용해서 보여주고 있습니다.
* 생각보다 간단하네요
  
  https://tvm.apache.org/docs/tutorial/tvmc_python.html#sphx-glr-tutorial-tvmc-python-py   

<br>
<br>

#### 3.1.4. Support Target List

* TVM으로 생성 가능한 Backend Target 관련 정보는 아래의 API를 통해서 알 수 있습니다.
  
  https://tvm.apache.org/docs/reference/api/python/target.html   

<br>
<br>

### 3.2. XLA(Accelerated Linear Algebra)

<br>
<br>

#### 3.2.0. Overview   

* XLA(Accelerated Linear Algebra)는 TensorFlow 모델의 실행 시간을 가속화하기 위한 선형 대수 특화 컴파일러입니다.기만 하면 됩니다.

* TensorFlow 모델을 소스 코드 변경 없이 가속화할 수 있다고는 하지만, 예제를 보면, 가속이 필요한 부분의 Source를 조금 수정을 하는 모습을 볼 수 있었습니다.

* TensorFlow 모델만을 가속할 수 있는 것 처럼 들리지만, 실제로는 JAX 심지어 PyTorch까지도 가속할 수 있습니다.

* XLA는 Tensorflow Package에 포함되어 있고, 가속을 하려는 함수에 tf.function이라는 Wrapping Function을 사용하면 XLA 가속이 가능합니다.

* tf.function는 Tensorflow Function을 포함하는 어떤 Function에서도 사용이 가능하므로, Inference 이외에도 사용이 가능합니다.

* 사용하는 방법은 tf.function을 사용할 때, jit_compile=True 인자를 추가하기만 하면 됩니다.

<br>
<br>

#### 3.2.1. Examples

* 간단하게 XLA를 기존 Code에 적용하는 예제를 살펴보도록 하겠습니다.

* 다음과 같은 Code가 있다고 해봅시다.


```python
import tensorflow as tf

model = tf.keras.Sequential(
    [tf.keras.layers.Dense(10, input_shape=(10,), activation="relu"), tf.keras.layers.Dense(5, activation="softmax")]
)
```

   

위 모델은 차원이 (10, )인 입력을 받아서, 다음과 같이 하면 결과가 나옵니다.

   


```python
# 모델에 대한 임의의 입력을 생성합니다.
batch_size = 16
input_vector_dim = 10
random_inputs = tf.random.normal((batch_size, input_vector_dim))

# 순전파를 실행합니다.
_ = model(random_inputs)
```

<br>
<br>

* 위의 예제를 XLA를 이용하는 Code로 바꾸려면, 아래와 같이 하면 됩니다.   

<br>

```python
xla_fn = tf.function(model, jit_compile=True)
_ = xla_fn(random_inputs)
```

   

* Model의 기본 call()함수를 XLA에 사용하려면 위와 같이 하면되지만, 임의의 함수를 XLA에 적용하려면 아래와 같이 하면 됩니다.   

   


```python
my_xla_fn = tf.function(model.my_xla_fn, jit_compile=True)
```

<br>
<br>

#### 3.2.2. Conversion & Optimization

<br>

* XLA 입력에 사용되는 언어를 "HLO IR" 또는 HLO(High Level Operations)라고 하는데, 간단하게 compiler IR이라고 생각하면 됩니다.

* XLA는 HLO에 정의된 Graph("computations")를 가져와서 다양한 하드웨어 아키텍처에 맞는 machine instructions로 Compile합니다.  

* XLA는 모듈식 구성으로 되어 있어서 다양한 하드웨어 아키텍처에 맞게 바꿀 수 있는 장점이 있습니다.

* x64 및 ARM64용 CPU 백엔드 뿐만 아니라 NVIDIA GPU 백엔드도 TensorFlow 소스 트리에 있습니다.

   

* 다음 다이어그램은 XLA의 컴파일 프로세스를 보여줍니다.

<img src="https://www.tensorflow.org/xla/images/how-does-xla-work.png">   

<br>

* XLA는 먼저 CSE , target-independent operation fusion , buffer analysis for allocating runtime memory for the computation와 같은 target-independent한 optimizations과 analysis를 거칩니다.   

* target-independent 작업을 마친 후에 XLA는 HLO computation을 Backend로 보냅니다.

* backend는 추가적인 HLO Level의 Optimization을 수행하고, 이 때에 Target의 정보가 필요합니다


* 이 단계에서는 backend가 특정 작업 혹은 패턴 조합을 일치시켜서 최적화된 Library 호출을 수행할 수도 있습니다.   

* 다음 단계는 target-specific code generation입니다.

* XLA가 지원하는 Backend는 low-level IR, optimization, code-generation에 LLVM을 사용합니다.

* Backend는 XLA HLO computation을 효율적으로 표현하는데 필요한 LLVM IR을 내보낸 다음에 LLVM을 호출하여서 LLVM IR에서 Native Code를 생성합니다.   

* GPU Backend는 현재 LLVM NVPTX Backend를 통해 NVIDIA GPU를 지원하고 있으며, CPU Backend는 여러 CPU ISA를 지원합니다.   

<br>
<br>

#### 3.2.3. Github

* XLA의 Github 주소는 아래를 참고하시기 바랍니다.
  
  https://github.com/openxla/xla   

<br>
<br>

### 3.3. Glow

<br>

#### 3.3.0. Introduction

* Glow는 Machine / Deeo Learning Compiler이며 학습 Model을 가속하기 위한 Execution Engine입니다.
  
  https://github.com/pytorch/glow


<br>
<br>

#### 3.3.1. Conversion & Optimization

<img src="https://github.com/pytorch/glow/blob/master/docs/3LevelIR.png?raw=true">

<br>

* Glow는 기존 Neural Network Dataflow Graph를 2단계 Strongly-Typed IR(Intermediate Representation)으로 변환합니다.   

* High-Level IR은 Domain-Specific Optimizations수행하고, Lower-Level Instruction-Based Address-only IR은 Instruction Scheduling, Static Memory Allocation , Copy Elimination 등과 같은 메모리 관련 최적화를 수행합니다.   

* Lowest Level에서는 특정 Hardware의 장점을 최대한 활용하기 위한 Machine-Specific Code를 생성합니다.   

<br>
<br>

#### 3.3.2. System Requirements

* MacOS와 Linux만 지원합니다.

* 결과물은 object file / Header File. 즉, C/C++에서만 사용가능합니다.

* 다른 Language에서 사용하려면 다소 귀찮은 작업을 해주어야 합니다.

* 조금 더 자세한 사항은 아래 Link를 참고해 주세요.

  https://github.com/pytorch/glow#getting-started   

<br>
<br>

#### 3.3.3. Example

* 실제 사용법은 아래 Link에서 자세하게 설명되어 있습니다.

  https://github.com/pytorch/glow/blob/master/docs/AOT.md   

* 보시면 아시겠지만, 여타 다른 Deep Learning Compiler와 기본적인 사용법은 크게 다르지 않습니다.   

<br>

---
title: "Learning Rate in Tensorflow"
date: 2023-09-06 08:26:28 -0400
categories: Deep Learning
---

# Learning Rate in Tensorflow

<br>
<br>
<br>

* 이번 Post에서는 Learning Rate에 대해서 알아보고, Tensorflow의 Callbadk중 하나인, Learning Rate Scheduler 에 대해서 알아보도록 하겠습니다.   

<br>
<br>
<br>

## 0. Learning Rate란 ?

<br>

Learning Rate란 Deep Learning Model이 학습할 때 사용하는 Backpropagation 과정에서 오차를 Gradient에 적용하는 비율을 말합니다.

Backpropagation를 수행할 때, Weight는 Loss Function을 오류를 줄이는 방향을 Update됩니다.

이때, 오류를 바로 Weight 변경에 적용하기 않고, Learning Rate를 곱해서 적용하게 됩니다.

예를 들면, Learning Rate가 0.5로 설정된 경우, 0.5 x 오류를 Weight Update에 사용하게 됩니다.   

<br>
<br>
<br>

## 1. Learning Rate 변화와 학습 효과

<br>

Learning Rate는 Optimizer가 Loss Function의 최소값에 도달하는 Step의 크기를 조절하는 역할을 합니다.

개인적으로 Learning Rate는 Model Train의 가장 중요한 Hyperparameter라고 생각합니다.

이 값이 잘못설정되면 학습이 아예 안되기도 하는 매우 중요하고 큰 역할을 하는 값입니다.

아래 그림은 Learning Rate가 Loss를 찾아가는데 어떤 역할을 하는지 잘 보여줍니다.         

<br>

<p align="center">
  <img src="https://substackcdn.com/image/fetch/f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fbucketeer-e05bbc84-baa3-437e-9518-adb32be77984.s3.amazonaws.com%2Fpublic%2Fimages%2F7918ccaf-6e1d-4c98-b410-212f4f6a209f_866x323.png">
</p>

<br>   
<br>

각 그림의 가장 낮은 지점이 Model의 Loss가 Minimum이 되는 위치를 나타내고, 빨간색 선은 Model이 Minimum Loss를 찾아가는 순서를 나타냅니다.   

오른쪽 그림같은 경우에는 너무 큰 Learning Rate를 사용하는 경우를 나타내며, 변화율이 너무 크기 때문에 최소값을 찾지 못하고 크게 요동치거나 심한 경우 Overflow될 수도 있습니다.

반대로 왼쪽 그림처럼 Learning Rate가 너무 작으면 Optimizer가 최소값을 찾는데 너무 오래 걸릴 수가 있습니다.

이런 경우에는 Optimizer가 Minimum을 찾는데 정체가 되거나 Local Minimum값에 갇혀서 Global Minimum을 찾지 못하는 경우가 생깁니다.

가운데 그림이 가장 이상적인 Learning Rate를 찾는 예로서 처음에는 조금 크게 이동하다가 Global Minimum에 가까이 갈수록 점점 작아지는 형태로 변하는 것을 보여줍니다.

<br>
<br>

처음부터 이상적인 Learning Rate를 찾기란 쉽지 않습니다. 그래서 이 문제를 해결하기 위해서 Learning Rate Scheduler가 도입되었습니다.

Learning Rate Scheduler는 처음부터 끝까지 동일한 Learning Rate를 사용하여 Train을 진행하지 않고 상황에 맞게 다양한 Learning Rate를 적용할 수 있도록 하는 것이 목적입니다.

다음에 설명드리는 방법들은 어떻게 Learning Rate를 바꿀 것인가에 중점을 두고 구현된 기능들입니다.

<br>
<br>
<br>

## 2. Learning Rate Scheduling API

<br>
<br>

### 2.0. 고정 학습률

<br>

 * 가장 기본적인 방법은 학습률을 고정된 값으로 설정하는 것입니다.

<br>

```python
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
```

<br>
<br>

### 2.1. Learning Rate Scheduling
 * 학습의 진행에 따라 학습률을 동적으로 변경할 수 있는 방법입니다.

<br>
<br>

#### 2.1.0. Step Decay   
  * 일정 epoch 간격마다 학습률을 감소시킵니다.
  * 설명 : https://www.tensorflow.org/api_docs/python/tf/keras/optimizers/schedules/ExponentialDecay

<br>

```python
tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate, 
    decay_steps, 
    decay_rate, 
    staircase=False, 
    name=None
)
```

<br>

 * **initial_learning_rate** : 최초 Learning Rate 값  

 * **decay_steps** : Learning Rate를 변화시킬 Step 값. 이 값만큰 Step이 될 때마다 Learning Rate가 변경됩니다.

 * **decay_rate** : Learning Rate를 변화시킬 비율. 이 값만큼 곱해져서 Learning Rate가 변화됩니다.

<br>

 * 아래는 실제 사용 예제입니다.   

<br>

```python
lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=0.001,
    decay_steps=10000,
    decay_rate=0.9)

optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
```

<br>
<br>
<br>

#### 2.1.1. Polynomial Decay   
  * 지정된 최대 epoch에 도달할 때까지 Learning Rate을 다항식에 따라 감소시킵니다.
  * 설명 : https://www.tensorflow.org/api_docs/python/tf/keras/optimizers/schedules/PolynomialDecay

<br>

```python
tf.keras.optimizers.schedules.PolynomialDecay(
    initial_learning_rate,
    decay_steps,
    end_learning_rate=0.0001,
    power=1.0,
    cycle=False,
    name=None
)
```

<br>

 * **initial_learning_rate** : 최초 Learning Rate 값니다.

 * **decay_steps** : Learning Rate를 변화시킬 Step 값. 이 값만큰 Step이 될 때마다 Learning Rate가 변경됩니다.

 * **end_learning_rate** : 최종 Learning Rate 값

 * **power** : Learning Rate를 변화시킬 때 지수에 들어갈 값입니다.         

<br>

 * 다음과 같이 사용할 수 있습니다.   

<br>

```python
starter_learning_rate = 0.1
end_learning_rate = 0.01
decay_steps = 10000
learning_rate_fn = tf.keras.optimizers.schedules.PolynomialDecay(
    starter_learning_rate,
    decay_steps,
    end_learning_rate,
    power=0.5)

model.compile(optimizer=tf.keras.optimizers.SGD(
                  learning_rate=learning_rate_fn),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
```

<br>
<br>
<br>

#### 2.1.2. Cosine Decay
  * Learning Rate를 cosine 함수를 사용하여 감소시킵니다.
  * 설명 : https://www.tensorflow.org/api_docs/python/tf/keras/optimizers/schedules/CosineDecay

<br>

```python
tf.keras.optimizers.schedules.CosineDecay(
    initial_learning_rate,
    decay_steps,
    alpha=0.0,
    name=None,
    warmup_target=None,
    warmup_steps=0
)
```

<br>
<br>
<br>

### 2.2. LearningRateScheduler Callback
 * Callback을 사용하여 Train 중에 Learning Rate를 동적으로 조정할 수 있습니다.
 * 설명 : https://www.tensorflow.org/api_docs/python/tf/keras/callbacks/LearningRateScheduler

<br>

```python
tf.keras.callbacks.LearningRateScheduler(
    schedule, verbose=0
)
```

<br>

 * **schedule**
   - Epoch과 현재 Learning Rate를 입력으로 받아서 새로운 Learning Rate를 Return해주는 함수
   - 이 함수를 본인이 원하는 대로 조절하여 Learning Rate를 변경할 수 있습니다.

<br>

 * 아래는 LearningRateScheduler Callback을 사용하는 예제입니다.   

<br>

```python
def scheduler(epoch, lr):
    if epoch < 10:
        return lr
    else:
        return lr * tf.math.exp(-0.1)

callback = tf.keras.callbacks.LearningRateScheduler(scheduler)
```

 * 위의 예제에서는 10 Epoch이하에서는 현재 Learning Rate를 유지하고, 그 이상의 Epoch에서는 Learning Rate를 10%씩 감소하도록 하는 예제입니다.   

<br>
<br>
<br>

### 2.3. ReduceLROnPlateau Callback
 * 이 Callback은 'patience' 동안 Monitoring링 지표(예: val_loss)의 개선이 없을 경우 학습률을 감소시킵니다.
 * 설명 : https://www.tensorflow.org/api_docs/python/tf/keras/callbacks/ReduceLROnPlateau

<br>

```python
tf.keras.callbacks.ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.1,
    patience=10,
    verbose=0,
    mode='auto',
    min_delta=0.0001,
    cooldown=0,
    min_lr=0,
    **kwargs
)
```

<br>

 * **monitor** : Monitoring할 Index니다.   

 * **factor** : Learning Rate를 감소시킬 비율

 * **patience**	: 몇 번의 Epoch 동안 Loss가 줄어들지 않을지 지켜보는 횟수.

 * **mode**	: {'auto', 'min', 'max'} 중에 하나.  

 * **min_delta**	 : 개선된 것으로 간주하기 위한 최소한의 변화량입니다. 즉, 최소한 이 값 정도는 개선되어야 개선으로 인정한다는 의미입니다.

 * **cooldown** : Learning rate가 감소한 후, ReduceLROnPlateau 콜백함수를 다시 실행하기 전에 대기 할 Epoch 수입니다. 

 * **min_lr**	: Learning rate의 하한선을 지정합니다. 현재 Learning Rate에 Factor를 곱한 값이 min_lr보다 작아도 min_lr가 새로운 Learning Rate로 적용됩니다.                 

<br>

 * 아래와 같이 사용할 수 있습니다.

<br>

```python
callback = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', 
factor=0.1, 
patience=10)
```

<br>
<br>

지금까지 다양한 Learning Rate 관련 Tensorflow API를 살펴보았습니다.

필요에 맞게 적절히 사용하셔서 좋은 Model 만드시기를 바라겠습니다.

읽어주셔서 감사합니다.   

<br>
<br>

---
title: "Early Stopping in Tensorflow"
date: 2023-09-06 08:26:28 -0400
categories: Deep Learning
---

# Early Stopping in Tensorflow

<br>
<br>
<br>

* 이번 Post에서는 Tensorflow의 Callbadk중 하나인, EarlyStopping에 대해서 알아보도록 하겠습니다.

<br>
<br>

## 0. Callback   

<br>

Tensorflow에서 Train을 시작하는 함수인 .fit()을 호출하면 Tensorflow는 마치 브레이크 고장한 폭주 기관차와 같은 상태가 됩니다.

지정한 Epoch을 다 끝마칠 때까지 멈출수도 없고, 현재 상태가 어떤지 알수도 없으며

각종 Training관련 지표들(Loss , Accuracy 등등)이 어떻게 바뀌고 있는지 확인할 방법이 없습니다.

그래서 Tensorflow에서는 이렇게 Train이 진행되는 동안 다양한 제어 및 관찰을 할 수 있도록 다양한 Callback 기능을 구현해 두었습니다.

Tensorflow에서 지원하는 다양한 Callback들은 아래 Link에서 확인할 수 있습니다.   

https://www.tensorflow.org/api_docs/python/tf/keras/callbacks

<br>
<br>

## 1. Early Stopping

<br>
<br>

그 중에서 이번 Post에서 알아볼 Callback은 EarlyStopping이라는 Callback입니다.

앞서 말했듯이, Train이 시작되면 Tensorflow는 .fit()에서 지정한 Epoch을 다 진행할 때까지 Train을 멈추지 않습니다.

그래서 어느 정도 Model의 성능이 나오는 시점이 되더라도 Train 중간에 멈출 수가 없습니다.

이 때 필요한 것이 Early Stopping Callback입니다.

EarlyStopping은 우리가 지정해준 특정 지표가 기준치에 도달하면 Train을 멈추도록 하는 역할을 합니다.

아래가 EarlyStopping Class입니다.   

<br>
<br>

```python
tf.keras.callbacks.EarlyStopping(
    monitor='val_loss',
    min_delta=0,
    patience=0,
    verbose=0,
    mode='auto',
    baseline=None,
    restore_best_weights=False,
    start_from_epoch=0
)
```

<br>

* **monitor**
  - Monitoring할 값. 기본값은 'val_loss'  

<br>

* **min_delta**
  - 개선으로 간주될 Monitoring된 수량의 최소 변화량. 즉, min_delta보다 적은 절대적인 변화는 개선으로 간주되지 않습니다.
  - 기본값은 0

<br>

* **patience**
  - 개선이 없는 상태로 지속되는 Epoch 횟수. 이 횟수 이후에 훈련이 중단됩니다.
  - 기본값은 0

<br>

* **verbose**
  - 메시지 출력 모드, 0 또는 1. 0이면 아무런 메시지를 출력하지 않고, 1이면 Callback이 작동할 때 메시지를 출력합니다.
  - 기본값은 0

<br>

* **mode**
  - {"auto", "min", "max"} 중 하나.
  - "min" 모드에서는 Monitoring되는 수량의 감소가 멈출 때 훈련이 중단됩니다.
  - "max" 모드에서는 증가가 멈출 때 중단됩니다.
  - "auto" 모드에서는 Monitoring되는 수량의 이름으로부터 방향이 자동으로 추론됩니다.
  - 기본값은 'auto'

<br>

* **baseline**
  - Monitoring되는 수량에 대한 기준값. 모델이 기준값보다 향상되지 않으면 훈련이 중단됩니다.
  - 기본값은 None

<br>

* **restore_best_weights**
  - Monitoring된 수량의 최적값을 가진 에폭에서 모델의 가중치를 복원할지 여부.
  - False인 경우 훈련의 마지막 단계에서 얻은 모델 가중치가 사용됩니다.
  - 기준에 상대적인 성능과 관계없이 Epoch이 복원됩니다.
  - 어떤 Epoch도 기준을 향상시키지 않으면 훈련은 Patience Epoch동안 실행되고 해당 세트에서 최적의 Epoch의 가중치가 복원됩니다.
  - 기본값은 False

<br>

* **start_from_epoch**
  - 개선을 Monitoring하기 시작하기 전에 대기할 Epoch의 횟수.
  - 이를 통해 초기에는 개선이 기대되지 않는 Warming Up 기간을 설정할 수 있으며, 이 기간 동안 훈련이 중단되지 않습니다.
  - 기본값은 0

<br>
<br>
<br>

## 2. Example

<br>
<br>

MNIST Dataset의 Classification Example로 실제로 어떻게 사용하는지 확인해 보도록 하겠습니다.   

<br>

* 필요한 Package를 Load합니다.   

<br>

```python
import tensorflow as tf
import tensorflow_datasets as tfds
```

<br>
<br>

* Train에 사용할 2개의 Dataset을 MNIST로 만듭니다.   

<br>

```python
(ds_train, ds_test), ds_info = tfds.load(
  'mnist',
  split=['train', 'test'],
  shuffle_files=True,
  as_supervised=True,
  with_info=True,
)

print(type(ds_train))
```

<br>

    [1mDownloading and preparing dataset Unknown size (download: Unknown size, generated: Unknown size, total: Unknown size) to C:\Users\Moon\tensorflow_datasets\mnist\3.0.1...[0m
    Dl Completed...: 0 url [00:00, ? url/s]
    Dl Size...: 0 MiB [00:00, ? MiB/s]
    Extraction completed...: 0 file [00:00, ? file/s]
    Generating splits...:   0%|          | 0/2 [00:00<?, ? splits/s]
    Generating train examples...: 0 examples [00:00, ? examples/s]
    Shuffling C:\Users\Moon\tensorflow_datasets\mnist\3.0.1.incompleteXYPQ2U\mnist-train.tfrecord*...:   0%|      …
    Generating test examples...: 0 examples [00:00, ? examples/s]
    Shuffling C:\Users\Moon\tensorflow_datasets\mnist\3.0.1.incompleteXYPQ2U\mnist-test.tfrecord*...:   0%|       …
    [1mDataset mnist downloaded and prepared to C:\Users\Moon\tensorflow_datasets\mnist\3.0.1. Subsequent calls will reuse this data.[0m
    <class 'tensorflow.python.data.ops.dataset_ops.PrefetchDataset'>

<br>
<br>

* Image를 Preprocessing해서 Label과 같이 Return해주는 Mapping 함수를 만듭니다.   

<br>

```python
def normalize_img(image, label):
    return tf.cast(image, tf.float32) / 255., label
```

<br>
<br>

* Train & Val. Dataset을 만듭니다. 좀 전에 만든 Mapping Function을 적용해 줍니다.   

<br>

```python
dataset_train = ds_train.map(normalize_img)
dataset_train = ds_train.batch(128)

dataset_test = ds_test.map(normalize_img)
dataset_test = ds_test.batch(128)
```

<br>
<br>

* Model 구조는 Simplie하게 만들어 줍시다.   

<br>

```python
model = tf.keras.models.Sequential([
tf.keras.layers.Flatten(input_shape=(28, 28)),
tf.keras.layers.Dense(128, activation='relu'),
tf.keras.layers.Dense(10)
])
```

<br>
<br>

* 작성한 Model을 Optimizer와 Loss Function 등을 정의해서 Compile해 줍니다.   

<br>

```python
model.compile(
  optimizer=tf.keras.optimizers.Adam(0.006),
  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
  metrics=[tf.keras.metrics.SparseCategoricalAccuracy()],
)
```

<br>
<br>

* 자, 이제 드디어 Early Stopping을 정의합니다.
* 아래의 Example에서는 Monitoring할 값을 'loss'를 선택했습니다.
* 이 값은 Train Loss를 말합니다.
* 'patience'값은 2로 설정했으며, 이는 **2 Epoch이 지나도 Train Loss가 나아지지 않는다면 Train을 종료**하라는 의미입니다.

<br>

```python
callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=2)
```

<br>

* 'monitor' 값에 들어갈 수 있는 값들은 다음과 같은 것들이 있습니다.
  - **loss** : Train Dataset의 Loss 값
  - **val_loss** : Validation Dataset의 Loss 값
  - **acc** : Train Dataset의 Accuracy 값
  - **val_acc** : Validation Dataset의 Accuracy 값

<br>
 
* 저는 주로 val_loss을 많이 사용하는데, 상황에 맞게 적절한 값을 사용하시면 됩니다.

<br>
<br>

* 이제 fit()을 호출할 때, **callback=[]** 이라는 Parameter에 사용할 Callback들을 추가해 주면 됩니다.
* 이번 Example에서는 EarlyStopping만 사용했기 때문에 EarlyStopping Callback만 추가된 모습을 볼 수 있습니다.

<br>

```python
history = model.fit(
  dataset_train,
  epochs=100,
  validation_data=dataset_test,
  callbacks=[callback]
)

print(len(history.history['loss']))
```

<br>

    Epoch 1/100
    469/469 [==============================] - 4s 5ms/step - loss: 4.8275 - sparse_categorical_accuracy: 0.7964 - val_loss: 0.5860 - val_sparse_categorical_accuracy: 0.8636
    Epoch 2/100
    469/469 [==============================] - 2s 4ms/step - loss: 0.4836 - sparse_categorical_accuracy: 0.8838 - val_loss: 0.4888 - val_sparse_categorical_accuracy: 0.8925
    Epoch 3/100
    469/469 [==============================] - 2s 4ms/step - loss: 0.3803 - sparse_categorical_accuracy: 0.9041 - val_loss: 0.4292 - val_sparse_categorical_accuracy: 0.9078
    Epoch 4/100
    469/469 [==============================] - 2s 4ms/step - loss: 0.3369 - sparse_categorical_accuracy: 0.9140 - val_loss: 0.3596 - val_sparse_categorical_accuracy: 0.9195
    Epoch 5/100
    469/469 [==============================] - 2s 4ms/step - loss: 0.3433 - sparse_categorical_accuracy: 0.9141 - val_loss: 0.3465 - val_sparse_categorical_accuracy: 0.9153
    Epoch 6/100
    469/469 [==============================] - 2s 4ms/step - loss: 0.3428 - sparse_categorical_accuracy: 0.9160 - val_loss: 0.4269 - val_sparse_categorical_accuracy: 0.8972
    6

<br>
<br>

* 각 Epoch마다 Train Loss를 잘 보시면, 4번째 Epoch에서 Loss가 0.3369를 기록하고 그 다음 2번의 Epoch에서 더 이상 Loss가 줄어들지 않았습니다.
* 이 조건이 Early Stopping 조건을 만족했기 때문에 Train이 자동으로 멈추었습니다.

<br>

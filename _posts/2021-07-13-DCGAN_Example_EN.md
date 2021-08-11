---
title: "DCGAN Example (EN)"
date: 2021-07-13 08:26:28 -0400
categories: Deep Learning
---

### DCGAN Example

<br>
<br>
<br>
<br>
<br>
<br>

# DCGAN Example

<br>

* In this post, I will take an example of DCGAN with code.

<br>

* For more information about GaN & DCGAN, please, refer to the below link, good material.

  [GAN](https://moonlight314.github.io/deep/learning/GAN_Paper_Review_KR/)

  [DCGAN](https://moonlight314.github.io/deep/learning/DCGAN_Paper_Review_KR/)

<br>

* This example has been imported from below Tensorflow Tutorial Site.

   [https://www.tensorflow.org/tutorials/generative/dcgan](https://www.tensorflow.org/tutorials/generative/dcgan)

<br>

* In this example, we will see that the trained model generates data similar to MNIST data after learning the probability distribution of MNIST dataset.

<br>
<br>
<br>
<br>
<br>

## 0. Import Package   

<br>

* Load packages

<br>

* imageio is necessary when making GIF animation.

<br>   
<br>

```python
import glob
import imageio
import matplotlib.pyplot as plt
import numpy as np
import os
import PIL
from tensorflow.keras import layers
import time

from IPython import display
```


```python
import tensorflow as tf
```

<br>
<br>
<br>
<br>
<br>


## 1. Loading Dataset & Preprocessing

<br>   

* As mentioned earlier, Generator and Discriminator will train MNIST dataSet.

<br>

* After training, Generator will generate a letter similar to MNIST handwriting.

<br>

```python
(train_images, train_labels), (_, _) = tf.keras.datasets.mnist.load_data()
```

    Downloading data from https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz
    11493376/11490434 [==============================] - 11s 1us/step
    


```python
train_images = train_images.reshape(train_images.shape[0], 28, 28, 1).astype('float32')
```

<br>   

* Normalizing images to [-1, 1]

<br>

```python
train_images = (train_images - 127.5) / 127.5 
```


```python
BUFFER_SIZE = 60000
BATCH_SIZE = 256
```

<br>

* Making dataset batch and shuffle.

<br>

```python
train_dataset = tf.data.Dataset.from_tensor_slices(train_images).shuffle(BUFFER_SIZE).batch(BATCH_SIZE)
```

<br>
<br>
<br>
<br>
<br>

## 2. Making Model

<br>

* Implementing Generator & Discriminator by the way used in DCGAN paper.

<br>
<br>

### 2.1. Generator

<br>

* The generator accepts the noise(random) value as input to train to generate MNIST data.

<br>

* As you can see, it uses Tensorflow conv2dtranspose[TF.KERAS.LAYERS.CONV2DTRANSPOSE] (https://www.tensorflow.org/api_docs/python/tf/keras/layers/conv2dtranspose) for upsampling.

<br>

* Then, making Generator using Batch Normalization / ReLU / Tanh.

<br>
<br>

```python
def make_generator_model():
    model = tf.keras.Sequential()
    model.add(layers.Dense(7*7*256, use_bias=False, input_shape=(100,)))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Reshape((7, 7, 256)))
    assert model.output_shape == (None, 7, 7, 256) # Notice : Batch size as None

    model.add(layers.Conv2DTranspose(128, (5, 5), strides=(1, 1), padding='same', use_bias=False))
    assert model.output_shape == (None, 7, 7, 128)
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Conv2DTranspose(64, (5, 5), strides=(2, 2), padding='same', use_bias=False))
    assert model.output_shape == (None, 14, 14, 64)
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Conv2DTranspose(1, (5, 5), strides=(2, 2), padding='same', use_bias=False, activation='tanh'))
    assert model.output_shape == (None, 28, 28, 1)

    return model
```


```python
generator = make_generator_model()

noise = tf.random.normal([1, 100])
generated_image = generator(noise, training=False)

plt.imshow(generated_image[0, :, :, 0], cmap='gray')
```

<br>

    <matplotlib.image.AxesImage at 0x26b88aaae80>

<br>


<p align="left">
  <img src="/assets/DCGAN_Example/output_51_1.png">
</p>

<br>

* The above image is that the generator has just tried to print without any train.

<br>

* It is just noise. Generator will gradually generates MNIST-like data while going through train ?

<br>

* If you need more information about Batch Normalization, please, refer to the below article.

  [Batch Normalization](https://moonlight314.github.io/deep/learning/Batch_Normalization_KR/)
  
<br>
<br>
<br>
<br>
<br>

### 2.2. Discriminator

<br>

* Discriminator is an image classifier based on CNN.

<br>
<br>

```python
def make_discriminator_model():
    model = tf.keras.Sequential()
    model.add(layers.Conv2D(64, (5, 5), strides=(2, 2), padding='same',
                                     input_shape=[28, 28, 1]))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))

    model.add(layers.Conv2D(128, (5, 5), strides=(2, 2), padding='same'))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))

    model.add(layers.Flatten())
    model.add(layers.Dense(1))

    return model
```


```python
discriminator = make_discriminator_model()
decision = discriminator(generated_image)
print (decision)
```

    tf.Tensor([[0.00050776]], shape=(1, 1), dtype=float32)
    
<br>
<br>
   

* This is important that this Discriminator is an image classifier, but it would not determine what the image generated by the generator is 0 to 9, but it is to determine whether the image generated by ** generator is fake or not. **

<br>

* So the output is dense(1).

<br>

* If it is judged to be real, it outputs a positive number, in case of fake, it outputs negative.

<br>
<br>
<br>
<br>
<br>

## 3. Loss Function & Optimizer

<br>

* G & D의 Loss Function을 정의하겠습니다.   

<br>
<br>

```python
# 이 메서드는 크로스 엔트로피 손실함수 (cross entropy loss)를 계산하기 위해 헬퍼 (helper) 함수를 반환합니다.
cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)
```

* [BinaryCrossentropy](https://www.tensorflow.org/api_docs/python/tf/keras/losses/BinaryCrossentropy)는 2개의 Parameter를 받으며, 첫번째는 실제 Label, 두번째는 Prediction 값을 받습니다.

<br>   
<br>
<br>

### 3.1. Loss Function of Discriminator

<br>

* 이 Method는 Discriminator가 Fake Image와 Real Image를 얼마나 잘 판별하는지 수치화합니다.

<br>

* Real Image에 대한 Discriminator의 예측과 1로 이루어진 행렬을 비교하고, Fake Image에 대한 Discriminator의 예측과 0으로 이루어진 행렬을 비교합니다.

<br>

```python
def discriminator_loss(real_output, fake_output):
    real_loss = cross_entropy(tf.ones_like(real_output), real_output)
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
    total_loss = real_loss + fake_loss
    return total_loss
```

<br>
<br>
<br>

### 3.2. Loss Function of Generator   

<br>

* Generator의 Loss Function은 Discriminator를 얼마나 잘 속였는지를 수치화를 합니다.

<br>

* 직관적으로 Generator가 잘 Train되고 있다면, Discriminator는 Fake Image를 Real Image(또는 1)로 분류를 할 것입니다. 

<br>

* 여기서 우리는 생성된 이미지에 대한 Discriminator의 결정을 1로 이루어진 행렬과 비교를 할 것입니다.

<br>

```python
def generator_loss(fake_output):
    return cross_entropy(tf.ones_like(fake_output), fake_output)
```

<br>
<br>
<br>

* Generator와 Discriminator는 따로 훈련되기 때문에, Optimizer를 따로 정의합니다.

<br>

```python
generator_optimizer = tf.keras.optimizers.Adam(1e-4)
discriminator_optimizer = tf.keras.optimizers.Adam(1e-4)
```

<br>
<br>
<br>

### 3.3. Check Point


```python
checkpoint_dir = './training_checkpoints'

checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")

checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
                                 discriminator_optimizer=discriminator_optimizer,
                                 generator=generator,
                                 discriminator=discriminator)
```

<br>
<br>
<br>
<br>
<br>

## 4. Define Train Loop

<br>

* Train Loop를 정의하도록 하겠습니다.

<br>

* Train 시에 사용할 몇몇 상수를 정의합니다.

<br>
<br>

```python
EPOCHS = 50
noise_dim = 100
num_examples_to_generate = 16

# 이 시드를 시간이 지나도 재활용하겠습니다. 
# (GIF 애니메이션에서 진전 내용을 시각화하는데 쉽기 때문입니다.) 
seed = tf.random.normal([num_examples_to_generate, noise_dim])
```

<br>
<br>
<br>

### 4.1. @tf.fuctnion Annotation

<br>

* @tf.fuctnion은 해당 Function을 Compile해서 Tensorflow 1.x Style로 사용할 수 있게 해줍니다.

<br>

* Tensorflow 1.x에서는 먼저 Train이나 Inference에 사용할 Network을 먼저 정의를 하였습니다. 그 후에, 이 Network을 Train시킬때는 Session을 열어서 Input Data를 넣어서 Train을 하거나, Inference를 하는 방식을 사용했습니다.

<br>

* 이렇게 하면 어떤 계산을 하려면 항상 Session을 열어야 해서 굉장히 불편했습니다.

<br>

* Tensorflow 2.x로 넘어오면서 가장 큰 변화중의 하나가, 이런 Session 개념없이 즉시 어떤 계산을 수행할 수 있는 Eager Execution 기능이 Default로 적용되었습니다.

<br>

* 이제 대부분 Tensorflow 2.x를 사용하는데, 왜 아직 1.x 때의 방식을 쓸 수 있는 저런 Annotation을 남겨두었을까 하는 의문이 드는데, 그 이유는 속도 때문입니다.

<br>

* Tensorflow 2.x에서도 @tf.fuctnion을 붙이면, 마치 Tensorflow 1.x처럼 Network의 생성과 실행이 분리된 것처럼 Function을 만들어 줍니다.

<br>

* 이렇게 하면 상황에 따라서 약간의 속도의 이점을 얻을 수 있습니다만, Debugging이 어려워질 수 있습니다.

<br>

* 모든 기능이 확실하다고 생각될 때 @tf.fuctnion를 붙이는 것이 좋습니다.

<br>
<br>
<br>

### 4.2. Custom Train Loop

<br>

* 보통 Tensorflow & Keras를 사용하면, 이미 만들어진 다양한 Network과 Function들을 이용해 쉽게 Model을 구현하고 Test해 볼 수 있습니다.

<br>

* Data준비하고 Loss Function 정의, Optimizer 정의 후에, .compile() / .fit()만 호출해 주면 Framework이 알아서 Forward Propagation / Back Propagation에서 편미분해서 Weight Update등을 알아서 다 해줍니다.

<br>

* 하지만 특정 경우에는 Tensorflow / Keras에서 아직 지원하지 않는 Model을 사용하고자 하거나, Train을 좀 더 세밀하게 제어하고 싶을 때가 있습니다. 

<br>

* 이런 경우에 Tensorflow의 Custom Training 기능을 사용할 수 있습니다

<br>

* 이 DCGAN Example에서도 Custom Train Loop을 사용하고 있습니다.

<br>
<br>
<br>

### 4.3. tf.GradientTape()   

<br>

* 우리가 Keras / Tensorflow를 사용해서 만든 Network은 Automatic Differentiation을 자동으로 계산해서 Backprpagation을 해주기 때문에 매우 편리합니다.

<br>

* 하지만, 이번 Example에서 처럼 Custom Train Loop를 사용하는 경우에는 Differentiation을 직접 해 주어야 합니다.

<br>

* 전부 다 손수하는 것은 아니지만, Forward Propagation할 때 Gradient를 저장해 두면, Back Propagation할 때 훨씬 더 빠르게 Differentiation을 구할 수 있습니다.

<br>

* 이와 같이 Forward Propagation할 때 필요한 값들을 저장해 주는 역할을 tf.GradientTape()가 해줍니다.

<br>
<br>
<br>

* 위에서 언급한 Annotation을 염두해 두고, 아래 Train Step Function을 보도록 하겠습니다.   

<br>

```python
@tf.function
def train_step(images):
    noise = tf.random.normal([BATCH_SIZE, noise_dim])

    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        generated_images = generator(noise, training=True)
        
        real_output = discriminator(images, training=True)
        fake_output = discriminator(generated_images, training=True)
        
        gen_loss = generator_loss(fake_output)
        disc_loss = discriminator_loss(real_output, fake_output)

    gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
    gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

    generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))
```

<br>

* Forward Propagation 중에 Tape에 변수를 저장한 후에, 나중에 Backpropagation에 앞서 저장한 변수를 사용하는 것을 볼 수 있습니다.

<br>
<br>
<br>

* 다음이 실제 Train Function입니다.

<br>

* Epoch Control도 Custom Train Loop다 보니, 알아서 해야 합니다.

<br>

```python
def train(dataset, epochs):
    
    for epoch in range(epochs):
        start = time.time()

        for image_batch in dataset:
            train_step(image_batch)

        # GIF를 위한 이미지를 바로 생성합니다.
        display.clear_output(wait=True)
        generate_and_save_images(generator,
                                 epoch + 1,
                                 seed)

        # 15 에포크가 지날 때마다 모델을 저장합니다.
        if (epoch + 1) % 15 == 0:
            checkpoint.save(file_prefix = checkpoint_prefix)

        # print (' 에포크 {} 에서 걸린 시간은 {} 초 입니다'.format(epoch +1, time.time()-start))
        print ('Time for epoch {} is {} sec'.format(epoch + 1, time.time()-start))

    # 마지막 에포크가 끝난 후 생성합니다.
    display.clear_output(wait=True)
    
    generate_and_save_images(generator,
                           epochs,
                           seed)
```


```python
def generate_and_save_images(model, epoch, test_input):
    # `training`이 False로 맞춰진 것을 주목하세요.
    # 이렇게 하면 (배치정규화를 포함하여) 모든 층들이 추론 모드로 실행됩니다. 
    predictions = model(test_input, training=False)

    fig = plt.figure(figsize=(4,4))

    for i in range(predictions.shape[0]):
        plt.subplot(4, 4, i+1)
        plt.imshow(predictions[i, :, :, 0] * 127.5 + 127.5, cmap='gray')
        plt.axis('off')

    plt.savefig('image_at_epoch_{:04d}.png'.format(epoch))
    plt.show()
```

<br>
<br>
<br>

## 5. Training   

<br>

* 위에서 정의한 train() method는 Generator와 Discriminator를 동시에 Train 시킵니다.

<br>

* DCGAN을 포함해, GAN은 Train시키기가 매우 까다로울 수 있습니다. 

<br>

* Generator와 Discriminator가 서로 균형을 맞추며 Train시키기가 어렵기 때문입니다.

<br>

* 초반에는 예상하듯이, 생성된 Image는 도데체 뭔지 도저히 알 수 없지만, Epoch이 진행될수록 점점 숫자처럼 보입니다.

<br>
<br>


```python
%%time
train(train_dataset, EPOCHS)
```

<br>

<p align="left">
  <img src="/assets/DCGAN_Example/output_155_0.png">
</p>

    Wall time: 20min 44s
    
<br>
<br>
<br>
<br>

## 6. Make GIF

<br>

* Train 과정을 좀 더 시각적으로 잘 볼 수 있도록, Animation GIF로 만들어 보겠습니다.

<br>

* 마지막 Checkpoint로 Model을 Load하겠습니다.   

<br>

```python
checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))
```

<br>

    <tensorflow.python.training.tracking.util.CheckpointLoadStatus at 0x26c587ad3a0>

<br>
<br>

```python
# 에포크 숫자를 사용하여 하나의 이미지를 보여줍니다.
def display_image(epoch_no):
    return PIL.Image.open('image_at_epoch_{:04d}.png'.format(epoch_no))
```


```python
display_image(EPOCHS)
```
<br>
<br>


<p align="left">
  <img src="/assets/DCGAN_Example/output_167_0.png">
</p>

<br>
<br>
<br>

```python
anim_file = 'dcgan.gif'

with imageio.get_writer(anim_file, mode='I') as writer:
    filenames = glob.glob('image*.png')
    filenames = sorted(filenames)
    last = -1
    
    for i,filename in enumerate(filenames):
        frame = 2*(i**0.5)
        
        if round(frame) > round(last):
            last = frame
        else:
            continue
            
        image = imageio.imread(filename)
        writer.append_data(image)
        
    image = imageio.imread(filename)
    writer.append_data(image)

import IPython

if IPython.version_info > (6,2,0,''):
    display.Image(filename=anim_file)
```

<br>   

* Folder에 보시면 'dcgan.gif' 파일이 생성되어 있는 것을 확인할 수 있습니다.

<br>

* 이 파일을 보시면, Train 과정이 어떻게 진행되었는지 알 수 있습니다.

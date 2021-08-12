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

* Let's make a loss function for G & D.

<br>
<br>

```python
# This method returns a helper function to calculate Cross Entropy Loss (cross entropy loss).
cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)
```

* BinaryCrossEntry gets two parameters, the first is the actual label, the second is the second is a prediction value.

<br>   
<br>
<br>

### 3.1. Loss Function of Discriminator

<br>

* This method numerates how well the Discriminator determines the fake image and real image.

<br>

* Compare discriminator for real image and compare matrix made up of 1, and compare matrices that are predicted and 0 of discriminator for fake image.

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

* The loss function of generator numerates how well generator deceives discriminator.

<br>

* If the generator is well trained, discriminator will classify fake image as real image (or 1).

<br>

* Here it will compare the decriminator's decision on the generated image and compared with the matrix made up of one.

<br>

```python
def generator_loss(fake_output):
    return cross_entropy(tf.ones_like(fake_output), fake_output)
```

<br>
<br>
<br>

* Because generator and discriminator are trained separately, we define optimizer separately.

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

* Defining train loop.

<br>

* Defineing some contant values.

<br>
<br>

```python
EPOCHS = 50
noise_dim = 100
num_examples_to_generate = 16

seed = tf.random.normal([num_examples_to_generate, noise_dim])
```

<br>
<br>
<br>

### 4.1. @tf.fuctnion Annotation

<br>

* @ tf.fuctNion allows you to compile a function and make it as Tensorflow 1.x Style.

<br>

* In Tensorflow 1.x, you first defined the network to use for train or inference. Thereafter, when using network, we opened a session to receive the input data to train, or use the network to do inference.

<br>

* It means that we always have to open a session to do some operation.

<br>

* One of the biggest changes as moving from Tensorflow 1.x to Tensorflow 2.x, the eager execution feature that can perform any calculations without a session concepts has been applied as default.

<br>

* We may have a question that most of them use Tensorflow 2.x, why it is left the annotation that can be used to write a method of 1.x yet, the reason is because of the speed.

<br>

* On Tensorflow 2.x, if you attach @ tf.fuctnion, you create a function as if you are creating and executing network as if Tensorflow 1.x.

<br>

* This allows you to benefit slightly depending on the situation, but debugging can be difficult.

<br>

* It is recommended that you attach @ tf.fuctnion when you think all the functions are confirmed.

<br>
<br>
<br>

### 4.2. Custom Train Loop

<br>

* Usually Tensorflow & Keras allows you to easily implement a model and test using various networks and functions that are already implemented.

<br>

* If you are prepared for data and definition of the loss function, optimizer definition, call .compile () / .fit (), the framework would do the forward propagation / back propagation, and weigh update.

<br>

* However, in certain cases, you may want to use model that is not yet supported by Tensorflow / Keras, or you want to control the train a little more closely.

<br>

* In this case, you can use the custom training feature in Tensorflow

<br>

* This DCGAN example also uses custom train loop.

<br>
<br>
<br>

### 4.3. tf.GradientTape()   

<br>

* Network made using Keras / Tensorflow is very convenient because automatic differentiation will automatically calculate and do backprpgation.

<br>

* However, if you are using custom train loop as in this example, you must directly differentiation by yourself.

<br>

* Not all of them, but if you store the gradient when forwarding, you can save the differentiation much faster when you save the gradient.

<br>

* TF.GradientTape () is the role that stores the values that are required when forwarding the forward procagation.

<br>
<br>
<br>

* Keep the annotation mentioned above and let's look at the train step function below.

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

* After saving variables in tape during forward propagation, you can later know using variables that you saved prior to backPropagation.

<br>
<br>
<br>

* This is the actual train function.

<br>

* Since it's a custom train loop, epoch control should be done by yourself.

<br>

```python
def train(dataset, epochs):
    
    for epoch in range(epochs):
        start = time.time()

        for image_batch in dataset:
            train_step(image_batch)

        display.clear_output(wait=True)
        generate_and_save_images(generator,
                                 epoch + 1,
                                 seed)

        if (epoch + 1) % 15 == 0:
            checkpoint.save(file_prefix = checkpoint_prefix)

        print ('Time for epoch {} is {} sec'.format(epoch + 1, time.time()-start))

    display.clear_output(wait=True)
    
    generate_and_save_images(generator,
                           epochs,
                           seed)
```


```python
def generate_and_save_images(model, epoch, test_input):
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

* The train() method defined above trains Generator and Discriminator at the same time.

<br>

* GANs, including DCGANs, can be very difficult to train.

<br>

* Because it is difficult to train the generator and discriminator at the same time in a balanced way

<br>

* As expected, the generated image is not quite clear at the beginning, but as the epoch progresses, it looks more and more like a number.

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

* In order to see the train process more visually, let's make it into an animation GIF.

<br>

* Let's load the model from our last checkpoint.

<br>

```python
checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))
```

<br>

    <tensorflow.python.training.tracking.util.CheckpointLoadStatus at 0x26c587ad3a0>

<br>
<br>

```python
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

* As you look in the folder, you can see that the 'dcgan.gif' file is created.

<br>

* Looking at this file, you can see how the train process went.

import keras
import matplotlib.pyplot as plt

from keras.layers import Input
from keras.models import Model, Sequential
from keras.layers.core import Dense, Dropout
from keras.layers.advanced_activations import LeakyReLU
from keras.datasets import mnist
from keras.optimizers import Adam
from keras import initializers

from tqdm import tqdm
import numpy as np
import os


def load_mnist():
    (xtrain, ytrain), (xtest, ytest) = mnist.load_data()
    xtrain = (xtrain.astype(np.float32) - 127.5) / 127.5
    xtrain = xtrain.reshape(60000, 784)
    return xtrain, ytrain, xtest, ytest


def make_generator(input_dim, optimizer):
    g = Sequential()

    g.add(Dense(256 ,input_dim=random_dim, kernel_initializer=initializers.RandomNormal(stddev=0.02)))
    g.add(LeakyReLU(0.2))
    g.add(Dense(512))
    g.add(LeakyReLU(0.2))
    g.add(Dense(1024))
    g.add(LeakyReLU(0.2))
    g.add(Dense(784, activation="tanh"))

    g.compile(loss="binary_crossentropy", optimizer=optimizer)

    return g

def make_discriminator(input_dim, optimizer):
    d = Sequential()

    d.add(Dense(1024, input_dim=784, kernel_initializer=initializers.RandomNormal(stddev=0.02)))
    d.add(LeakyReLU(0.2))
    d.add(Dropout(0.3))
    d.add(Dense(512))
    d.add(LeakyReLU(0.2))
    d.add(Dropout(0.3))
    d.add(Dense(256))
    d.add(LeakyReLU(0.2))
    d.add(Dropout(0.3))
    d.add(Dense(1, activation="sigmoid"))

    d.compile(loss="binary_crossentropy", optimizer=optimizer)

    return d

def plot_images(epoch, generator, examples=100, dim=(10,10), figsize=(10,10)):
    noise = np.random.normal(0,1,size=[examples, random_dim])
    generated_images = generator.predict(noise)
    generated_images = generated_images.reshape(examples, 28, 28)
    plt.figure(figsize=figsize)

    for i in range(generated_images.shape[0]):
        plt.subplot(dim[0], dim[1], i+1)
        plt.imshow(generated_images[i], interpolation="nearest", cmap="gray_r")
        plt.axis("off")
    plt.tight_layout()

    if not os.path.isdir("out"):
        os.mkdir("out")

    plt.savefig(f"out/epoch{epoch}.png")

def train(epochs=1, batch_size=128):
    xtrain, ytrain, xtest, ytest = load_mnist()
    batch_count = xtrain.shape[0] / batch_size

    random_dim = 100
    batch_count = xtrain.shape[0] // batch_size
    for e in range(1, epochs +1):
        print("-"*10, f"Epoch {e}", "-"*10)

        for _ in tqdm(range(batch_count)):
            noise = np.random.normal(0, 1, size=[batch_size, random_dim])
            image_batch = xtrain[np.random.randint(0, xtrain.shape[0], size=batch_size)]
            generated_images = generator.predict(noise)

            X = np.concatenate([image_batch, generated_images])
            y_dis = np.zeros(2 * batch_size)
            y_dis[:batch_size] = 0.9

            discriminator_trainable = True
            discriminator.train_on_batch(X, y_dis)
            noise = np.random.normal(0, 1, size=[batch_size, random_dim])
            y_gen = np.ones(batch_size)

            discriminator.trainable = False
            gan.train_on_batch(noise, y_gen)

        plot_images(e, generator)

if __name__ == "__main__":
    np.random.seed(1000)
    random_dim = 100

    optimizer = Adam(lr=0.0002, beta_1=0.5)

    generator = make_generator(random_dim, optimizer)
    discriminator = make_discriminator(784, optimizer)

    # train the generator first
    discriminator.trainable = False
    gan_input = Input(shape=(random_dim,))
    x = generator(gan_input)
    gan_output = discriminator(x)
    gan = Model(inputs=gan_input, outputs=gan_output)
    gan.compile(loss="binary_crossentropy", optimizer=optimizer)

    train(25, 128)

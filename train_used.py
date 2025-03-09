import numpy as np
import tensorflow as tf
from losses import *
from neural_network import *
from optimizers import *
from utils import *

if __name__ == '__main__':
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    x_train = (x_train/255.0).reshape(-1, 784)
    y_train = one_hot(y_train, num_classes=10)
    x_test = (x_test/255.0).reshape(-1, 784)
    y_test = one_hot(y_test, num_classes=10)

    train_size = int(0.9 * len(x_train))
    indices = np.arange(len(x_train))
    np.random.shuffle(indices)
    x_train = x_train[indices]
    y_train = y_train[indices]

    x_train_split = x_train[:train_size]
    y_train_split = y_train[:train_size]

    x_val_split = x_train[train_size:]
    y_val_split = y_train[train_size:]

    nn = NeuralNetwork(
        layer_info=[784, 10],
        num_hidden_layers=3,
        num_node_per_hidden_layer=256,
        weight_decay=0.01,
        learning_rate=0.001,
        optimizer='adam',
        batch_size=32,
        loss='cross_entropy',
        weights_init='xavier',
        activation='relu'
    )
    print(x_train_split.shape, y_train_split.shape)
    print(x_val_split.shape, y_val_split.shape)
    print(x_test.shape, y_test.shape)
    nn.train((x_train_split, y_train_split), (x_val_split,
             y_val_split), (x_test, y_test), epochs=10)

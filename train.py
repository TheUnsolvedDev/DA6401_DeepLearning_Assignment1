import numpy as np
import wandb
import argparse

from activation import *
from optimizers import *
from neural_network import *
from losses import *


def get_name(params):
    """
    Generate a string representation of the hyperparameters used for training.

    Parameters:
    params (dict): A dictionary containing the hyperparameters. The keys are the hyperparameter names,
                   and the values are the corresponding hyperparameter values.

    Returns:
    str: A string representation of the hyperparameters. Each hyperparameter is represented as a
         shortened key-value pair separated by a colon. The pairs are joined by underscores.
    """
    keys = [key for key in params.keys()]
    values = [params[key] for key in keys]

    name = ''
    for key, val in zip(keys, values):
        name += ''.join([i[0] for i in key.split('_')])+':'+str(val)+'_'
    return name


def train(args):
    """
    This function trains a neural network using TensorFlow and Weights & Biases (W&B) for tracking experiments.
    It loads the specified dataset, preprocesses the data, splits it into training, validation, and testing sets,
    configures the hyperparameters, initializes the neural network, and trains it.

    Parameters:
    args (argparse.Namespace): An object containing the command-line arguments. It should have the following attributes:
        - wandb_project (str): The name of the project used to track experiments in W&B dashboard.
        - dataset (str): The dataset to use for training. It should be either 'mnist' or 'fashion_mnist'.
        - epochs (int): The number of epochs to train the neural network.
        - batch_size (int): The batch size used to train the neural network.
        - optimizer (str): The optimizer to use for training. It should be one of 'sgd', 'momentum', 'nag', 'rmsprop', 'adam', or 'nadam'.
        - learning_rate (float): The learning rate for the optimizer.
        - beta (float): The beta value used by the rmsprop optimizer.
        - beta1 (float): The beta1 value used by the adam and nadam optimizers.
        - beta2 (float): The beta2 value used by the adam and nadam optimizers.
        - epsilon (float): The epsilon value used by the optimizers.
        - weight_decay (float): The weight decay value used by the optimizers.
        - weight_init (str): The weight initialization method. It should be one of 'random', 'xavier_normal', or 'xavier_uniform'.
        - num_layers (int): The number of hidden layers in the neural network.
        - hidden_size (int): The number of hidden neurons in each feedforward layer.
        - activation (str): The activation function to use. It should be one of 'sigmoid', 'tanh', or 'relu'.
        - loss_fn (str): The choice of loss function. It should be either 'cross_entropy' or 'squared_error'.

    Returns:
    None
    """
    import tensorflow as tf

    with wandb.init(project=args.wandb_project) as run:
        config = wandb.config
        config.dataset = args.dataset

        if config.dataset == 'mnist':
            (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
        elif config.dataset == 'fashion_mnist':
            (x_train, y_train), (x_test,
                                 y_test) = tf.keras.datasets.fashion_mnist.load_data()

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

        config.epochs = args.epochs
        config.batch_size = args.batch_size
        config.loss = args.loss_fn
        config.optimizer = args.optimizer
        config.learning_rate = args.learning_rate
        config.beta = args.beta
        config.beta1 = args.beta1
        config.beta2 = args.beta2
        config.epsilon = args.epsilon
        config.weight_decay = args.weight_decay
        config.weight_init = args.weight_init
        config.num_layers = args.num_layers
        config.hidden_size = args.hidden_size
        config.activation = args.activation

        run.name = get_name(wandb.config)
        print(get_name(wandb.config))

        nn = NeuralNetwork(num_hidden_layers=config.num_layers, num_node_per_hidden_layer=config.hidden_size, weight_decay=config.weight_decay,
                           learning_rate=config.learning_rate, optimizer=config.optimizer, batch_size=config.batch_size, weights_init=config.weight_init, activation=config.activation, loss=config.loss)
        nn.train((x_train_split, y_train_split), (x_val_split, y_val_split), (x_test, y_test),
                 epochs=config.epochs,log=True)


def train_sweep(config=None):
    """
    This function performs hyperparameter sweep using Weights & Biases (W&B) for tracking experiments.
    It loads the Fashion-MNIST dataset, preprocesses the data, splits it into training, validation, and testing sets,
    configures the hyperparameters, initializes the neural network, and trains it.

    Parameters:
    config (dict, optional): A dictionary containing the hyperparameters for the sweep.
                             If None, the function will use default values.
                             The dictionary should have the following keys:
                             - number_of_epochs: The number of epochs to train the neural network.
                             - number_of_hidden_layers: The number of hidden layers in the neural network.
                             - size_of_every_hidden_layer: The number of hidden neurons in each feedforward layer.
                             - weight_decay: The weight decay value used by the optimizers.
                             - learning_rate: The learning rate for the optimizer.
                             - optimizer: The optimizer to use for training.
                             - batch_size: The batch size used to train the neural network.
                             - weight_initialisation: The weight initialization method.
                             - activation_functions: The activation function to use.

    Returns:
    None
    """
    import tensorflow as tf
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()
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

    with wandb.init(config=config) as run:
        config = wandb.config
        run.name = get_name(config)

        nn = NeuralNetwork(
            num_hidden_layers=config.number_of_hidden_layers,
            num_node_per_hidden_layer=config.size_of_every_hidden_layer,
            weight_decay=config.weight_decay,
            learning_rate=config.learning_rate,
            optimizer=config.optimizer,
            batch_size=config.batch_size,
            weights_init=config.weight_initialisation,
            activation=config.activation_functions,
            beta=0.9,
            beta1=0.9,
            beta2=0.999,
            epsilon=1e-8,
            loss='cross_entropy',
        )
        nn.train((x_train_split, y_train_split), (x_val_split, y_val_split), (x_test, y_test),
                 epochs=config.number_of_epochs)


def main():
    """
    The main function is responsible for parsing command-line arguments, setting up the neural network training process,
    and performing hyperparameter sweep using Weights & Biases (W&B) if specified.

    Parameters:
    None

    Returns:
    None
    """
    parser = argparse.ArgumentParser(
        description="Train neural network with specified hyperparameters")
    parser.add_argument("-wp", "--wandb_project", default="DA24D402_DL_1",
                        help="Project name used to track experiments in Weights & Biases dashboard")
    parser.add_argument("-we", "--wandb_entity", default="Shuvrajeet",
                        help="defines the enitity of the project")
    parser.add_argument("--sweep", action="store_true",
                        help="Perform hyperparameter sweep using wandb")
    parser.add_argument("-d", "--dataset", default="fashion_mnist",
                        choices=["mnist", "fashion_mnist"], help="Dataset to use for training")
    parser.add_argument("-e", "--epochs", type=int, default=10,
                        help="Number of epochs to train neural network.")
    parser.add_argument("-b", "--batch_size", type=int, default=64,
                        help="Batch size used to train neural network.")
    parser.add_argument("-o", "--optimizer", default="sgd", choices=[
                        "sgd", "momentum", "nag", "rmsprop", "adam", "nadam"], help="Optimizer")
    parser.add_argument("-lr", "--learning_rate", type=float,
                        default=0.1, help="Learning rate")
    parser.add_argument("-beta", "--beta", type=float,
                        default=0.9, help="Beta used by rmsprop optimizer")
    parser.add_argument("-beta1", "--beta1", type=float, default=0.99,
                        help="Beta1 used by adam and nadam optimizers")
    parser.add_argument("-beta2", "--beta2", type=float, default=0.995,
                        help="Beta2 used by adam and nadam optimizers")
    parser.add_argument("-eps", "--epsilon", type=float,
                        default=1e-6, help="Epsilon used by optimizers")
    parser.add_argument("-w_d", "--weight_decay", type=float,
                        default=0.0, help="Weight decay used by optimizers")
    parser.add_argument("-w_i", "--weight_init", default="random",
                        choices=["random", "xavier_normal", 'xavier_uniform'], help="Weight initialization")
    parser.add_argument("-nhl", "--num_layers", type=int,
                        default=1, help="Number of hidden layers")
    parser.add_argument("-sz", "--hidden_size", type=int, default=4,
                        help="Number of hidden neurons in a feedforward layer")
    parser.add_argument("-a", "--activation", default="sigmoid",
                        choices=["sigmoid", "tanh", "relu"], help="Activation function")
    parser.add_argument("-loss", "--loss_fn", type=str, default="cross_entropy",
                        choices=['cross_entropy', 'squared_error'], help="choice of loss function")
    args = parser.parse_args()

    if args.sweep:
        sweep_config = {
            'method': 'bayes',
        }

        metric = {
            'name': 'validation_accuracy',
            'goal': 'maximize'
        }
        sweep_config['metric'] = metric
        parameters_dict = {
            'number_of_epochs': {
                'values': [5, 10, 25, 50]
            },
            'number_of_hidden_layers': {
                'values': [3, 4, 5]
            },
            'size_of_every_hidden_layer': {
                'values': [32, 64, 128]
            },
            'weight_decay': {
                'values': [0, 0.0005, 0.5]
            },
            'learning_rate': {
                'values': [1e-3, 1e-4]
            },
            'optimizer': {
                'values': ['sgd', 'momentum', 'nesterov', 'rmsprop', 'adam', 'nadam']
            },
            'batch_size': {
                'values': [16, 32, 64]
            },
            'weight_initialisation': {
                'values': ['random', 'xavier_normal', 'xavier_uniform']
            },
            'activation_functions': {
                'values': ['sigmoid', 'tanh', 'relu']
            },
        }
        sweep_config['parameters'] = parameters_dict

        sweep_id = wandb.sweep(
            sweep_config, project=args.wandb_project)
        wandb.agent(sweep_id, function=train_sweep, count=100)
    else:
        train(args)


if __name__ == "__main__":
    main()

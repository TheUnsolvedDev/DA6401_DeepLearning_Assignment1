import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import wandb


class Solutions:
    def __init__(self):
        print('Solutions to Assignment1!')
        self.see_all()

    def exercise1(self):
        """
        This function loads the Fashion MNIST dataset, selects one image from each class,
        and displays them along with their corresponding labels. It also logs the images
        to a Weights & Biases project for visualization.

        Parameters:
        None

        Returns:
        None
        """
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()
        indices = [np.argwhere(y_train == i)[0][0] for i in range(10)]
        item = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress',
                'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
        images = x_train[indices]

        plt.figure(figsize=(16, 2))
        for i in range(10):
            plt.subplot(1, 10, i+1)
            plt.xticks([])
            plt.yticks([])
            plt.imshow(images[i])
            plt.xlabel(item[i])
        plt.show()

        # wandb.login()
        wandb.init(project="DA24D402_DL_1", id="Question-1")
        wandb.log({"plot": [wandb.Image(img, caption=caption)
                            for img, caption in zip(images, item)]})
        
    def exercise2(self):
        raise NotImplementedError

    def see_all(self):
        self.exercise1()


if __name__ == '__main__':
    sol = Solutions()
    sol.see_all()

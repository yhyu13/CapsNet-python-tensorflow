from tensorflow.examples.tutorials.mnist import input_data
from config import FLAGS, HPS

mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
import numpy as np
from numpy.random import RandomState

prng = RandomState(1234567890)


def get_batch(train=True, batch_size=64):
    """Generate training image and lables."""
    if train:
        xs, ys = mnist.train.next_batch(batch_size)
    else:
        # known MNIST dataset test data has 1000 sample
        # draw batch_size from them with predefined random state
        # in order to make test set selection repeatable
        # But it is recommended to run full evaluation if
        # a computer is resourceful enough
        rows = prng.randint(1000, size=batch_size)
        xs, ys = mnist.test.images[rows, :], mnist.test.labels[rows, :]
        #xs, ys = mnist.test.images[:1000, :], mnist.test.labels[:1000, :]
    xs = np.reshape(xs, (batch_size, 28, 28))
    xs = np.expand_dims(xs, 3)
    return {'x': xs, 'y': ys}

from matplotlib import pyplot as plt
import numpy as np


def plot_imgs(inputs):
    """Plot smallNORB images helper"""
    fig = plt.figure()
    plt.title('Show images')
    r = np.floor(np.sqrt(len(inputs))).astype(int)
    for i in range(r**2):
        sample = inputs[i]
        # print(np.asarray(sample).shape)
        a = fig.add_subplot(r, r, i + 1)
        a.imshow(sample, cmap='gray')
    plt.show()

# Dynamic Routing Between Capsules

This is python TensorFlow implementation of [Dynamic Routing Between Capsules](https://arxiv.org/pdf/1710.09829.pdf)

[![License](https://img.shields.io/github/license/mashape/apistatus.svg?maxAge=2592000)](https://github.com/yhyu13/CapsNet-python-tensorflow/blob/master/LICENSE)

## Requirment:


```python3.6```
```pip install -r requirment.txt```


## Other Implementations

- Keras:
  - [XifengGuo/CapsNet-Keras](https://github.com/XifengGuo/CapsNet-Keras)
  I referred to some functions in this repository.

- TensorFlow:
  - [naturomics/CapsNet-Tensorflow](https://github.com/naturomics/CapsNet-Tensorflow.git)   
  XifengGuo referred to some functions in this repository.
  - [InnerPeace-Wu/CapsNet-tensorflow](https://github.com/InnerPeace-Wu/CapsNet-tensorflow)   
  - [chrislybaer/capsules-tensorflow](https://github.com/chrislybaer/capsules-tensorflow)

- PyTorch:
  - [timomernick/pytorch-capsule](https://github.com/timomernick/pytorch-capsule)
  - [gram-ai/capsule-networks](https://github.com/gram-ai/capsule-networks)
  - [nishnik/CapsNet-PyTorch](https://github.com/nishnik/CapsNet-PyTorch.git)
  - [leftthomas/CapsNet](https://github.com/leftthomas/CapsNet)

- MXNet:
  - [AaronLeong/CapsNet_Mxnet](https://github.com/AaronLeong/CapsNet_Mxnet)

- Chainer:
  - [soskek/dynamic_routing_between_capsules](https://github.com/soskek/dynamic_routing_between_capsules)

- Matlab:
  - [yechengxi/LightCapsNet](https://github.com/yechengxi/LightCapsNet)

---

## Experiment

### MNIST

See training result:

```tensorboard --logdir=train_log/ --host=0.0.0.0 --port=8080```

```tensorboard --logdir=test_log/ --host=0.0.0.0 --port=6060```

Training result show between CapsNet (paper version) and CNN baseline (paper version). The cost of CapsNet is marginal loss plus l2 regularization. The cost of CNN baseline is the sum of cross entropy and l2 loss. Notice the CE loss is more sensitive than the marginal loss. Tensor neuron (aka. Capsule)'s loss function is more stable (?), it also support existence of multiple classes (<-one of the purpose of this paper). The CapsNet trains 3 times faster than the CNN baseline, partially due to a simpler implementation that takes advantage of TensorFlow reshape mechanism.

![](/figure/Nov24train.png)

![](/figure/Nov24test.png)

### CIFAR10

TBD

## Architecture

***Reconstruction layer will come soon***


Credit: XifengGuo

### Capsules


According to Paper:

> One very special property is the existence of the instantiated entity in the image. An obvious way to represent existence is by using a separate logistic unit whose output is the probability that the entity exists. ***In this paper we explore an interesting alternative which is to use the overall length of the vector of instantiation parameters to represent the existence of the entity and to force the orientation
of the vector to represent the properties of the entity1. We ensure that the length of the vector output of a capsule cannot exceed 1 by applying a non-linearity that leaves the orientation of the vector unchanged but scales down its magnitude.***

As the follow up paper--[MATRIX CAPSULES WITH EM ROUTING](https://openreview.net/pdf?id=HJWLfGWRb)--states, the CapsNet has the following defects:

>1. It uses the length of the pose vector to represent the probability that the entity represented by a capsule is present. To keep the length less than 1 requires an unprincipled non-linearity that prevents there from being any sensible objective function that is minimized by the iterative routing procedure.
2. It uses the cosine of the angle between two pose vectors to measure their agreement. Unlike the log variance of a Gaussian cluster, the cosine is not good at distinguishing between quite good agreement and very good agreement.
3. It uses a vector of length n rather than a matrix with n elements to represent a pose, so its transformation matrices have n2 parameters rather than just n.

## Conclusion

TBD

## Concept Explanation

[(Chinese) 如何看待Hinton的论文《Dynamic Routing Between Capsules》？ - SIY.Z的回答 - 知乎](https://www.zhihu.com/question/67287444/answer/251241736)

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

![](/figure/Nov23_1.png)

![](/figure/Nov23_2.png)

### CIFAR10

TBD

## Architecture

***Reconstruction layer will come soon***

![](https://raw.githubusercontent.com/XifengGuo/CapsNet-Keras/master/result/model.png)

Credit: XifengGuo

### Capsules


According to Paper:

> One very special property is the existence of the instantiated entity in the image. An obvious way to represent existence is by using a separate logistic unit whose output is the probability that the entity exists. ***In this paper we explore an interesting alternative which is to use the overall length of the vector of instantiation parameters to represent the existence of the entity and to force the orientation
of the vector to represent the properties of the entity1. We ensure that the length of the vector output of a capsule cannot exceed 1 by applying a non-linearity that leaves the orientation of the vector unchanged but scales down its magnitude.***

TDB


## Conclusion

TBD

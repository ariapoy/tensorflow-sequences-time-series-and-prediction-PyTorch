'''
# Week 2 Lesson 1
In the screencast for this lesson I go through steps for slicing the windows of time series.
At the end, you can generate features and labels of dataset.

'''

# # Setup
import numpy as np
import torch as t
from torch.utils.data import DataLoader
import pdb

np.random.seed(0)
t.manual_seed(0)

dataset = t.randn(19)


# ref: https://gist.github.com/teoliphant/96eb779a16bd038e374f2703da62f06d
def window(x, size, shift=None, stride=1):
    try:
        nd = len(size)
    except TypeError:
        size = tuple(size for i in x.shape)
        nd = len(size)
    if nd != x.ndimension():
        raise ValueError("size has length {0} instead of "
                         "x.ndim which is {1}".format(len(size), x.ndimension())) 
    out_shape = tuple(xi-wi+1 for xi, wi in zip(x.shape, size)) + size
    if not all(i>0 for i in out_shape):
        raise ValueError("size is bigger than input array along at "
                         "least one dimension")
    out_strides = x.stride() * 2
    return t.as_strided(x, out_shape, out_strides)
# t.as_strided(dataset, (6,5), (1,1), 0)
# window(dataset, out_shape, out_strides, 0)

dataset = window(dataset, 5)


# Ref: https://discuss.pytorch.org/t/a-fast-way-to-apply-a-function-across-an-axis/8378
def apply(func, M):
    res = [func(m) for m in t.unbind(M, dim=0) ]
    return res 

dataset = apply(lambda window: (window[:-1], window[-1:]), dataset)

dataset = DataLoader(dataset, shuffle=True, batch_size=2)
for x, y in dataset:
    print("x = ", x.numpy())
    print("y = ", y.numpy())

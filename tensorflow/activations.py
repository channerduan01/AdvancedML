#!/usr/bin/env python
# encoding: utf-8
"""
activations.py

Created by Darcy on 24/04/2016.
Copyright (c) 2016 Darcy. All rights reserved.
"""

import sys
import os
import time
import math
import numpy as np
from matplotlib import pyplot as plt


def relu(x):
    return max(x, 0)


def relu6(x):
    return min(max(x, 0), 6)


def sigmod(x):
    return 1/(1 + math.exp(-x))


def tanh(x):
    # https://en.wikipedia.org/wiki/Hyperbolic_function
    return 2 * sigmod(2*x) - 1


def demo(fn=None, name=''):
    assert fn is not None
    num = 500
    x = np.linspace(-4*np.pi, 4*np.pi, num)
    y = np.zeros(num)

    for i in range(0, num):
        y[i] = fn(x[i])
    m = np.max(y) + 1
    n = np.min(y) - 1

    plt.plot(x, y)
    plt.ylim(n, m)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.axhline(0, color='black')
    plt.axvline(0, color='black')
    plt.grid()
    plt.title(name)
    plt.show()


def main():
    start = time.time()
    demo(sigmod, 'sigmod')
    demo(relu, 'relu')
    demo(relu6, 'relu6')
    demo(tanh, 'tanh')
    end = time.time()
    print 'CPU time: %f seconds.' % (end - start,)


if '__main__' == __name__:
    main()

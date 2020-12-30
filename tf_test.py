import os
import re

import tensorflow as tf
import numpy as np
from tensorflow.python.layers.core import flatten

tf.executing_eagerly()

conv1_weights_tf = np.load('params/conv1_Wfuck.npy')
conv1_biases = np.load('params/conv1_b.npy')

conv2_weights_tf = np.load('params/conv2_W.npy')
conv2_biases = np.load('params/conv2_b.npy')

fc1_biases = np.load('params/fc1_b.npy')
fc1_weights = np.load('params/fc1_W.npy')

fc2_biases = np.load('params/fc2_b.npy')
fc2_weights = np.load('params/fc2_W.npy')

fc3_biases = np.load('params/fc3_b.npy')
fc3_weights = np.load('params/fc3_W.npy')

def test_file(name):
    infile = np.loadtxt(name)
    indata = infile[np.newaxis,:,:,np.newaxis]
    conv1 = tf.nn.conv2d(indata, conv1_weights_tf, strides=[1, 1, 1, 1], padding='SAME') + conv1_biases
    fuck = conv1.numpy()
    fuck2 = np.einsum('bijc->bcij', fuck)
    fuck3 = fuck2[0][0]
    fuck3 = fuck2[0][1]
    fuck3 = fuck2[0][2]
    conv1 = tf.nn.relu(conv1)
    conv1 = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')

    conv2 = tf.nn.conv2d(conv1, conv2_weights_tf, strides=[1, 1, 1, 1], padding='VALID') + conv2_biases
    conv2 = tf.nn.relu(conv2)
    conv2 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')

    fc0 = flatten(conv2)
    fc1 = tf.matmul(fc0, fc1_weights) + fc1_biases
    fc1 = tf.nn.relu(fc1)

    fc2 = tf.matmul(fc1, fc2_weights) + fc2_biases
    fc2 = tf.nn.relu(fc2)

    logits = tf.matmul(fc2,fc3_weights) + fc3_biases

    return np.argmax(logits[0])

test_file('test_img/0_7.txt')
"""
accu = 0
all = 0
for f in os.listdir('test_img2'):
    fname = 'test_img2/' + f
    r = test_file(fname)
    ans = int(re.search(r'_(.*)\.', f).group(1))
    print("{} => {}".format(f, r))
    all += 1
    if r == ans:
        accu += 1

print("Accu = {}, all = {}".format(accu, all))
"""

import numpy as np
import scipy.signal
import re
import io
import os


conv1_weights_tf = np.load('params/conv1_W.npy')
conv1_weights = np.einsum('ijqk->qkij', conv1_weights_tf)
conv1_biases = np.load('params/conv1_b.npy')

conv2_weights_tf = np.load('params/conv2_W.npy')
conv2_weights = np.einsum('ijqk->qkij', conv2_weights_tf)
conv2_biases = np.load('params/conv2_b.npy')

fc1_biases = np.load('params/fc1_b.npy')
fc1_weights = np.load('params/fc1_W.npy')

fc2_biases = np.load('params/fc2_b.npy')
fc2_weights = np.load('params/fc2_W.npy')

fc3_biases = np.load('params/fc3_b.npy')
fc3_weights = np.load('params/fc3_W.npy')


def convolve_same(dat, w):
    def get_dat(x, y):
        if 0 <= x < dat.shape[0] and 0 <= y < dat.shape[1]:
            return dat[x][y]
        return 0
    result = np.zeros(dat.shape)
    for x in range(dat.shape[0]):
        for y in range(dat.shape[1]):
            ans = 0
            offset_i = w.shape[0] // 2
            offset_j = w.shape[1] // 2
            for i in range(w.shape[0]):
                for j in range(w.shape[1]):
                    n1 = get_dat(x + i - offset_i, y + j - offset_j)
                    n2 = w[i][j]
                    ans += n1 * n2
            result[x][y] = ans
    return result


def convolve_valid(dat, w):



def test_file(name):
    input = np.loadtxt(name)

    conv1_ch = 6
    conv2_ch = 16
    conv2_results = np.zeros((conv2_ch, 10, 10))

    def max_pool(img, factor: int):
        """ Perform max pooling with a (factor x factor) kernel"""
        ds_img = np.full((img.shape[0] // factor, img.shape[1] // factor), -float('inf'), dtype=img.dtype)
        np.maximum.at(ds_img, (np.arange(img.shape[0])[:, None] // factor, np.arange(img.shape[1]) // factor), img)
        return ds_img

    def relu(X):
        return np.maximum(0, X)

    relu_mat = np.vectorize(relu)

    for i in range(0, conv1_ch):
        # result = scipy.signal.convolve2d(input, conv1_weights[0][i], mode='same')
        result = convolve_same(input, conv1_weights[0][i])
        result += conv1_biases[i]
        actved = relu_mat(result)
        ans = max_pool(actved, 2)
        for j in range(0, conv2_ch):
            c2r = scipy.signal.convolve2d(ans, conv2_weights[i][j], mode='valid')
            conv2_results[j] += c2r

    fc_inputs = np.zeros((conv2_ch, 5, 5))
    for j in range(0, conv2_ch):
        result = conv2_results[j] + conv2_biases[j]
        actved = relu_mat(result)
        ans = max_pool(actved, 2)
        fc_inputs[j] = ans

    dat = fc_inputs.flatten()
    fc1 = dat @ fc1_weights
    fc1 += fc1_biases
    fc1 = relu_mat(fc1)

    fc2 = fc1 @ fc2_weights
    fc2 += fc2_biases
    fc2 = relu_mat(fc2)

    fc3 = fc2 @ fc3_weights
    fc3 += fc3_biases

    res = np.argmax(fc3)
    return res


test_file('test_img/0_7.txt')
"""
for f in os.listdir('test_img'):
    fname = 'test_img/' + f
    r = test_file(fname)
    print("{} => {}".format(f, r))
    """

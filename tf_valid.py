import tensorflow as tf
import numpy as np


tf.executing_eagerly()

mat = np.array([[0,1,2,3,4],[5,6,7,8,9],[10,11,12,13,14],[15,16,17,18,19],[20,21,22,23,24]],dtype=np.int32)
mat_tf = mat[np.newaxis,:,:,np.newaxis]
kernel = np.array([[0,1,2],[3,4,5],[6,7,8]],dtype=np.int32)
kernel_tf = np.einsum('qkij->ijqk', kernel[np.newaxis,np.newaxis,:,:])

conv2 = tf.nn.conv2d(mat_tf, kernel_tf, strides=[1, 1, 1, 1], padding='SAME') 

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


print(conv2)
print(convolve_same(mat, kernel))

import numpy as np

param = np.zeros((3, 3, 1, 6))

param[0][0][0][0] = 1
param[1][0][0][1] = 1
param[0][1][0][2] = 1

np.save('params/conv1_Wfuck.npy', param)

from PCA import PCA
from SNR import SNR
from scipy.io import loadmat
from matplotlib import pyplot as plt
import numpy as np

mat = loadmat('Z:\\traces\\pca_dom_101_1101.mat')
traces = mat['pca']
Y = mat['Y']

snr_t = np.abs(SNR.SNR(traces, Y, 256, 20, np.float32))

fig, ax = plt.subplots()
ax.plot(range(0, 20), snr_t)
ax.set_title("SNR of PCA in time")
ax.set_xlabel("dimension-time")
plt.show()

mat = loadmat('Z:\\traces\\pca_freq_101_1101.mat')
traces = mat['pca']
Y = mat['Y']

snr_t = np.abs(SNR.SNR(traces, Y, 256, 20, np.complex128))

fig, ax = plt.subplots()
ax.plot(range(0, 20), snr_t)
ax.set_title("SNR of PCA in frequency")
ax.set_xlabel("dimension-frequency")
plt.show()

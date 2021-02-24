# from utils import traces, Y, SAMPLES
from scipy.io import loadmat
from SNR import SNR
import numpy as np
import matplotlib.pyplot as plt
import hdf5storage

path = 'Z:\\traces\\'
mat = hdf5storage.loadmat(path+'pca_dom_101_1001.mat')
traces = mat['pca']
Y = mat['Y']
SAMPLES = np.shape(traces)[1]
Queries = np.shape(traces)[0]
# snr of PCA in time domain
snr_t = SNR.SNR(traces[:, :], Y[:], 256, SAMPLES, np.float64)
plt.plot(range(0, SAMPLES), np.abs(snr_t))
plt.xlabel("t")
plt.title("SNR of PCA T.D.")
plt.show()


mat = hdf5storage.loadmat(path+'pca_freq_101_1001.mat')
traces = mat['pca']
Y = mat['Y']
# compute ifft then SNR
SAMPLES = np.shape(traces)[1]
Queries = np.shape(traces)[0]
# snr of PCA in freq domain
snr_t = SNR.SNR(traces[:, :], Y[:], 256, SAMPLES, np.complex128)
plt.plot(range(0, SAMPLES), np.abs(snr_t))
plt.xlabel("t")
plt.title("SNR of PCA FFT")
plt.show()




import hdf5storage

from PCA.PCA import PCA
from SNR import SNR
from scipy.io import loadmat, savemat
from matplotlib import pyplot as plt
import numpy as np

crop = range(300, 1300)

mat = loadmat('Z:\\traces\\traces_101_601_full.mat')
traces = mat['traces'][:, crop]
SAMPLES = np.shape(traces)[1]
Y = mat['Y']
pca = PCA(traces).fit(1000)[1]
snr_t = np.abs(SNR.SNR(pca, Y, 256, SAMPLES, np.float32))

fig, ax = plt.subplots()
ax.plot(range(0, SAMPLES), snr_t)
ax.set_title("SNR of PCA in time")
ax.set_xlabel("dimension-time")
plt.show()

mat = hdf5storage.loadmat('Z:\\traces\\fft_301_401.mat')
traces = mat['fft']
Y = mat['Y']
SAMPLES = np.shape(traces)[1]
pca = PCA(traces)

snr_t = np.abs(SNR.SNR(pca.fit(1400)[1], Y, 256, SAMPLES, np.complex128))

fig, ax = plt.subplots()
ax.plot(range(0, SAMPLES), snr_t)
ax.set_title("SNR of PCA in frequency")
ax.set_xlabel("dimension-frequency")
plt.show()
indexes = pca.feature_select(70)
print(indexes)
filter = np.zeros((1, 1400))
filter[0,indexes] = 1
savemat("filter_PCA_FFT.mat", {"filter": filter})
ifft = np.zeros(np.shape(traces), np.complex128)
ifft[:, indexes] = traces[:, indexes]
ifft = np.fft.ifft(ifft, n=1400, axis=1)
plt.plot(ifft[1, :])
plt.show()
snr_t = np.abs(SNR.SNR(ifft, Y, 256, SAMPLES, np.complex128))

fig, ax = plt.subplots()
ax.plot(range(300, 1300), snr_t[300:1300])
ax.set_title("SNR of PCA in frequency with ifft")
ax.set_xlabel("time")
plt.show()

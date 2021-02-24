from SNR import SNR
import matplotlib.pyplot as plt
import hdf5storage
import numpy as np

path = 'Z:\\traces\\'
mat = hdf5storage.loadmat(path+'fft_501_600.mat')
traces = mat['ffts']
Y = mat['Y']
#mat = hdf5storage.loadmat(path+'fft_601_700.mat')
#traces = np.append(traces, mat['ffts'], axis=0)
#Y = np.append(Y, mat['Y'])

SAMPLES = np.shape(traces)[1]
Queries = np.shape(traces)[0]
crop = range(0, int(Queries/8))

snr_t = SNR.SNR_wrapper(traces[crop, :], Y[crop], 256, SAMPLES, np.complex128, frames=28)
plt.plot(range(0, SAMPLES), np.abs(snr_t))
plt.show()

from SNR import SNR
from scipy.io import loadmat, savemat
from scipy.signal import spectrogram
import numpy as np
from matplotlib import pyplot as plt
from PCA.PCA import PCA
from SSA.SSA import OVSSA, ssa_wrapper
import hdf5storage
from time import time


mat = hdf5storage.loadmat('Z:\\traces\\CMOS\\ovssa.mat')
traces = (mat['traces'])
Y = mat['Y']
snr_t = SNR.SNR_wrapper(traces, Y, 256, 1000, frames=10)

plt.plot(snr_t)
plt.show()

mat = hdf5storage.loadmat('Z:\\traces\\unshuffled\\traces_crop.mat')
traces = (mat['traces'])
Y = mat['Y']
snr_t = SNR.SNR_wrapper(traces, Y, 256, 1600, frames=1)
plt.plot(snr_t-0.005)

mat = hdf5storage.loadmat('Z:\\traces\\shuffled_rp_2\\traces_crop.mat')
traces = (mat['traces'])
Y = mat['Y']
snr_t = SNR.SNR_wrapper(traces, Y, 256, 1600, frames=1)
plt.plot(snr_t)
mat = hdf5storage.loadmat('Z:\\traces\\shuffled_rp_4\\traces_crop.mat')
traces = (mat['traces'])
Y = mat['Y']
snr_t = SNR.SNR_wrapper(traces, Y, 256, 1600, frames=1)
plt.plot(snr_t)
mat = hdf5storage.loadmat('Z:\\traces\\shuffled_rp_8\\traces_crop.mat')
traces = (mat['traces'])
Y = mat['Y']
snr_t = SNR.SNR_wrapper(traces, Y, 256, 1600, frames=1)
plt.plot(snr_t)

plt.title("SNR of shuffled designs")
plt.xlabel("t[us]")
plt.ylabel("SNR")
plt.legend(["unshuffled", "shuffled rp_2", "shuffled rp_4", "shuffled rp_8"])
plt.yscale('log')
plt.savefig('shuffling.pdf')
plt.show()
exit(0)
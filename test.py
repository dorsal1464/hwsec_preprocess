from SNR import SNR
from scipy.io import loadmat, savemat
from scipy.signal import spectrogram
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.colors import Normalize
from PCA.PCA import PCA
from SSA.SSA import OVSSA, ssa_wrapper
import hdf5storage
from time import time

# -------------------ACTION-------------------------
path = 'Z:\\traces\\CMOS'
mat = loadmat(path+'\\traces_101_601_full.mat')
traces = mat['traces']

fig, axes = plt.subplots(1, 3)

f, t, Sxx = spectrogram(np.mean(traces, axis=0), 1, nperseg=200, scaling='spectrum')
f = 1400*f
cf = axes[0].pcolormesh(t, f[np.where(f<500)], Sxx[np.where(f < 500)[0], :], shading='gouraud', cmap='Greys', norm=Normalize())
# fig.colorbar(cf, ax=axes[0])
axes[0].set_ylabel('Frequency [MHz]')
axes[0].set_xlabel('Time samples')
axes[0].set_title('CMOS spectogram')

path = 'Z:\\traces\\DUALRAIL'
mat = loadmat(path+'\\traces_101_601_full.mat')
traces = mat['traces']

f, t, Sxx = spectrogram(np.mean(traces, axis=0), 1, nperseg=200, scaling='spectrum')
f = 1400*f
cf = axes[1].pcolormesh(t, f[np.where(f<500)], Sxx[np.where(f < 500)[0], :], shading='gouraud', cmap='Greys', norm=Normalize())
# fig.colorbar(cf, ax=axes[1])
axes[1].set_ylabel('Frequency [MHz]')
axes[1].set_xlabel('Time samples')
axes[1].set_title("DUALRAIL spectogram")

path = 'Z:\\traces\\shuffled_rp_2'
mat = loadmat(path+'\\traces_crop.mat')
traces = mat['traces']

f, t, Sxx = spectrogram(np.mean(traces, axis=0), 1, nperseg=200, scaling='spectrum')
f = 1400*f
cf = axes[2].pcolormesh(t, f[np.where(f<500)], Sxx[np.where(f < 500)[0], :], shading='gouraud', cmap='Greys', norm=Normalize())
fig.colorbar(cf, ax=axes[2])
axes[2].set_ylabel('Frequency [MHz]')
axes[2].set_xlabel('Time samples')
axes[2].set_title("rp_2 spectogram")
plt.tight_layout(pad=1.2)
plt.savefig("spectogram.pdf")
plt.show()

path = 'Z:\\traces\\CMOS\\'

mat = hdf5storage.loadmat(path+'fullssa.mat')
# np.complex64
traces = (mat['traces'])
Y = mat['Y']

Y = np.reshape(Y, (np.size(Y),))

SAMPLES = np.shape(traces)[1]
Queries = np.shape(traces)[0]
from main_fft import plot_roc
plot_roc(traces, Y, None, 40, 'fullssa_base', [0, 1000])
#plot_roc(traces, Y, "Z:\\filters\\shuffled_rp_8\\filter_BPF.mat", 40, 'rp8_bpf', [480, 800])
#plot_roc(traces, Y, "Z:\\filters\\shuffled_rp_8\\filter_SNR_FFT.mat", 40, 'rp8_snr', [480, 800])
exit(0)

path = 'Z:\\traces\\shuffled_rp_4\\'

mat = hdf5storage.loadmat(path+'fft_crop.mat')
traces = np.complex64(mat['fft'])
Y = mat['Y']
Y = np.reshape(Y, (np.size(Y),))

SAMPLES = np.shape(traces)[1]
Queries = np.shape(traces)[0]
from main_fft import plot_roc
#plot_roc(traces, Y, None, 40, 'rp8_base')
plot_roc(traces, Y, "Z:\\filters\\shuffled_rp_4\\filter_BPF.mat", 40, 'rp4_bpf', [480, 730])
plot_roc(traces, Y, "Z:\\filters\\shuffled_rp_4\\filter_SNR_FFT.mat", 40, 'rp4_snr', [480, 730])


path = 'Z:\\traces\\shuffled_rp_2\\'
mat = hdf5storage.loadmat(path+'fft_crop.mat')
traces = np.complex64(mat['fft'])
Y = mat['Y']

Y = np.reshape(Y, (np.size(Y),))

SAMPLES = np.shape(traces)[1]
Queries = np.shape(traces)[0]
from main_fft import plot_roc
#plot_roc(traces, Y, None, 40, 'rp8_base')
plot_roc(traces, Y, "Z:\\filters\\shuffled_rp_2\\filter_BPF.mat", 40, 'rp2_bpf', [480, 600])
plot_roc(traces, Y, "Z:\\filters\\shuffled_rp_2\\filter_SNR_FFT.mat", 40, 'rp2_snr', [480, 600])


exit(0)

i = time()
L = None
ssa = OVSSA(traces[1, 300:1300], L, 201, 100)
print("dur: ", time()-i)
i = time()
s, t, n = ssa_wrapper(traces[1, 300:1300], L)
print(time()-i)
fig, ax = plt.subplots()
ax.plot(range(0, 1000), ssa)
ax.plot(s+t)
ax.set_title("SNR of OVASSA in time")
ax.set_xlabel("dimension-time")
plt.show()

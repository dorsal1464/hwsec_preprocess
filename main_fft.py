from SNR import SNR
import matplotlib.pyplot as plt
import hdf5storage
import numpy as np

# PART 1 - load ffts

path = 'Z:\\traces\\'
FILTER_FACTOR = 0.6
mat = hdf5storage.loadmat(path+'fft_201_301.mat')
traces = mat['fft']
Y = mat['Y']
#mat = hdf5storage.loadmat(path+'fft_301_401.mat')
#traces = np.append(traces, mat['ffts'], axis=0)
#Y = np.append(Y, mat['Y'])

SAMPLES = np.shape(traces)[1]
Queries = np.shape(traces)[0]
crop = range(0, int(Queries))

fig, ax = plt.subplots()
# PART 2 - plot SNR of fft
snr_t = SNR.SNR_wrapper(traces[crop, :], Y[0, crop], 256, SAMPLES, np.complex128, frames=14)
snr_t = np.abs(snr_t)
ax.plot(range(0, SAMPLES), snr_t)
ax.set_title("SNR in frequency domain")
ax.set_xlabel("f[Hz]")
plt.show()

# PART 3 - plot max SNR vs filter factor


def filter_freq_signal(snr_t, traces, filter_factor=FILTER_FACTOR):
    max_snr = np.max(snr_t)
    indexes = (snr_t >= filter_factor * max_snr)

    filter_traces = np.zeros(np.shape(traces), dtype=np.complex128)
    filter_traces[:, indexes] = traces[:, indexes]

    return np.fft.ifft(filter_traces, axis=1, n=SAMPLES)


max_snr_f = np.zeros(11, dtype=np.float32)

for i in range(1, 11):
    filter_traces = filter_freq_signal(snr_t, traces, float(i) / 10)
    snr = np.abs(SNR.SNR_wrapper(filter_traces, Y, 256, SAMPLES, np.complex128, frames=14)[300:1300])
    max_snr_f[i] = np.max(snr)
    print(np.max(snr))

fig, ax = plt.subplots()
ax.plot(range(0, 11), max_snr_f)
ax.set_title("max SNR vs filter threshold")
ax.set_xlabel("filter x10")
plt.show()

# PART 4 - plot SNR with MAXIMAL FACTOR

filter_traces = filter_freq_signal(snr_t, traces, 0.6)
snr = np.abs(SNR.SNR_wrapper(filter_traces, Y, 256, SAMPLES, np.complex128, frames=14))
fig, ax = plt.subplots()
ax.plot(range(0, SAMPLES), snr)
ax.set_title("SNR of max filter threshold")
ax.set_xlabel("t[sec]")
plt.show()

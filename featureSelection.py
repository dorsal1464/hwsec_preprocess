import numpy as np
import hdf5storage
from SNR import SNR
from matplotlib import pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import SelectKBest, SelectPercentile
from sklearn.feature_selection import chi2, mutual_info_classif

path = 'Z:\\traces'

mat = hdf5storage.loadmat(path+'\\fft_201_301.mat')
traces = mat['fft']
Y = mat['Y']
SAMPLES = np.shape(traces)[1]
Queries = np.shape(traces)[0]
scaler = MinMaxScaler()
# scaler.fit(traces)
# tr_scaled = scaler.transform(traces)
# chi2 ???

def determine_percentile():
    max_snr = list()
    for i in np.arange(10, 20):
        select = SelectPercentile(chi2, percentile=i)
        select.fit(np.abs(traces), np.reshape(Y, (Queries,)))
        indexes = select.get_support(True)
        print(indexes)
        filter_traces = np.zeros(np.shape(traces), np.complex128)
        filter_traces[:, indexes] = traces[:, indexes]
        filter_traces = np.fft.ifft(filter_traces, n=SAMPLES, axis=1)
        snr_t = np.abs(SNR.SNR(filter_traces, Y, 256, SAMPLES, np.complex128))
        max_snr.append(np.max(snr_t[300:1300]))
    fig, ax = plt.subplots()
    ax.plot(np.arange(10, 20), max_snr)
    ax.set_title("max SNR vs feature selection FFT percentile")
    ax.set_xlabel("percentile")
    plt.show()

# max is at 15 percentile...


select = SelectPercentile(chi2, percentile=15)
select.fit(np.abs(traces), np.reshape(Y, (Queries,)))
indexes = select.get_support(True)
print(indexes)
filter_traces = np.zeros(np.shape(traces), np.complex128)
filter_traces[:, indexes] = traces[:, indexes]
filter_traces = np.fft.ifft(filter_traces, n=SAMPLES, axis=1)
snr_t = np.abs(SNR.SNR(filter_traces, Y, 256, SAMPLES, np.complex128))

fig, ax = plt.subplots()
ax.plot(range(0, SAMPLES), snr_t)
ax.set_title("SNR of feature selection FFT chi2")
ax.set_xlabel("time")
plt.show()
# test with mutual info instead --> reduce samples
mark = 10000
select = SelectPercentile(mutual_info_classif, percentile=15)
select.fit(np.abs(traces[:mark, :]), np.reshape(Y, (Queries,)))
indexes = select.get_support(True)
print(indexes)
filter_traces = np.zeros(np.shape(traces), np.complex128)
filter_traces[:, indexes] = traces[:, indexes]
filter_traces = np.fft.ifft(filter_traces, n=SAMPLES, axis=1)
snr_t = np.abs(SNR.SNR(filter_traces, Y, 256, SAMPLES, np.complex128))

fig, ax = plt.subplots()
ax.plot(range(0, SAMPLES), snr_t)
ax.set_title("SNR of feature selection FFT MI")
ax.set_xlabel("time")
plt.show()
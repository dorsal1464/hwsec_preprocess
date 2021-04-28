import numpy as np
import hdf5storage
from SNR import SNR
from matplotlib import pyplot as plt
from sklearn.preprocessing import MinMaxScaler, PolynomialFeatures
from sklearn.feature_selection import SelectKBest, SelectPercentile
from sklearn.feature_selection import chi2, mutual_info_classif

path = 'Z:\\traces'
savepath = 'Z:\\figures'

mat = hdf5storage.loadmat(path+'\\fft_201_301.mat')
traces = mat['fft']
Y = mat['Y']
SAMPLES = np.shape(traces)[1]
Queries = np.shape(traces)[0]
scaler = MinMaxScaler()
# scaler.fit(traces)
# tr_scaled = scaler.transform(traces)
# chi2 ???


def multivariate_fs(traces, Y, degree):
    _len = np.shape(traces)[1]
    index_dict = dict()
    # prepare reverse index table...
    for i in range(_len):
        for j in range(i+1, _len):
            calc_index = _len-1 + (_len - 1 - i)*i + j - i
            index_dict[calc_index] = (i, j)

    poly = PolynomialFeatures(degree, include_bias=False, interaction_only=True)
    poly_traces = poly.fit_transform(traces)
    print(np.shape(poly_traces))
    select = SelectPercentile(chi2, percentile=15)
    select.fit(np.abs(poly_traces), np.reshape(Y, (Queries,)))
    indexes = select.get_support(True)
    # get real indexes...
    real_index = list()
    for num in indexes:
        if num < _len:
            real_index.append(num)
        # else, find in reverse hash table
        else:
            (i, j) = index_dict[num]
            real_index.append(i)
            real_index.append(j)
    return real_index


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

#indexes = multivariate_fs(np.abs(traces), Y, 2)
#print(indexes)


# test with mutual info instead --> reduce samples
mark = 100000
select = SelectPercentile(mutual_info_classif, percentile=5)
select.fit(np.abs(traces[:mark, :]), np.reshape(Y, (Queries,))[:mark])
indexes = select.get_support(True)
print(indexes)
t = np.zeros(1400)
t[indexes] = 1
fig, ax = plt.subplots()
ax.plot(t)
ax.set_title("SNR of feature selection FFT mi - filter shape")
ax.set_xlabel("freq")
plt.show()
filter_traces = np.zeros(np.shape(traces), np.complex128)
filter_traces[:, indexes] = traces[:, indexes]
filter_traces = np.fft.ifft(filter_traces, n=SAMPLES, axis=1)
snr_t = np.abs(SNR.SNR(filter_traces, Y, 256, SAMPLES, np.complex128))

fig, ax = plt.subplots()
ax.plot(range(300, 1300), snr_t[300:1300])
ax.set_title("SNR of feature selection FFT MI")
ax.set_xlabel("time")
plt.show()

select = SelectPercentile(chi2, percentile=15)
select.fit(np.abs(traces), np.reshape(Y, (Queries,)))
indexes = select.get_support(True)
print(indexes)
filter_traces = np.zeros(np.shape(traces), np.complex128)
filter_traces[:, indexes] = traces[:, indexes]
filter_traces = np.fft.ifft(filter_traces, n=SAMPLES, axis=1)
snr_t = np.abs(SNR.SNR(filter_traces, Y, 256, SAMPLES, np.complex128))

fig, ax = plt.subplots()
ax.plot(range(300, 1300), snr_t[300:1300])
ax.set_title("SNR of feature selection FFT chi2")
ax.set_xlabel("time")
plt.show()
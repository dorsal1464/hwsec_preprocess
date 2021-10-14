import numpy as np
import hdf5storage
from scipy.io import savemat
from SNR import SNR
from matplotlib import pyplot as plt
from sklearn.preprocessing import MinMaxScaler, PolynomialFeatures
from sklearn.feature_selection import SelectKBest, SelectPercentile
from sklearn.feature_selection import chi2, mutual_info_classif

path = 'Z:\\traces\\shuffled_rp_8\\'
savepath = 'Z:\\figures\\shuffled_rp_2\\'

mat = hdf5storage.loadmat(path+'fft_crop.mat')
traces = mat['fft']
Y = mat['Y']
SAMPLES = np.shape(traces)[1]
Queries = np.shape(traces)[0]
scaler = MinMaxScaler()
# scaler.fit(traces)
# tr_scaled = scaler.transform(traces)
# chi2 ???


def load_filtered_traces(traces, Y, filterpath, savepath):
    _mat = hdf5storage.loadmat(filterpath)
    _filter = _mat['filter']
    _ifft = np.fft.ifft(_filter * traces, 1400, axis=1)
    snr_t = np.abs(SNR.SNR_wrapper(_ifft, Y, 256, np.shape(traces)[0], type=np.complex128))
    savemat(savepath, {'x': np.arange(300, 1300), 'y': snr_t[300:1300]})
    return snr_t


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
    select = SelectPercentile(chi2, percentile=5)
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
    for i in np.arange(5, 20):
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

rng = np.append(np.arange(950, 1050), np.arange(450, 550))
#indexes = multivariate_fs(np.abs(traces[:500000, rng]), Y, 2)
#print(indexes)
#copy with offset - 700

# test with chi2...
select = SelectPercentile(chi2, percentile=5)
select.fit(np.abs(traces), np.reshape(Y, (Queries,)))
indexes = select.get_support(True)
z = np.zeros(np.shape(traces)[1])
print(indexes)
z[indexes] = 1
savemat("filter_FS_CHI2.mat", {'filter': z})
filter_traces = np.zeros(np.shape(traces), np.complex128)
filter_traces[:, indexes] = traces[:, indexes]
filter_traces = np.fft.ifft(filter_traces, n=SAMPLES, axis=1)
snr_t = np.abs(SNR.SNR(filter_traces, Y, 256, SAMPLES, np.complex128))

fig, ax = plt.subplots()
ax.plot(range(300, 1300), snr_t[300:1300])
ax.set_title("SNR of feature selection FFT chi2")
ax.set_xlabel("time")
plt.show()
savemat("snr_FS_CHI2.mat", {'x': np.arange(0, np.size(snr_t)), 'y': snr_t})
# test with mutual info instead --> reduce samples?
mark = 2000000
select = SelectPercentile(mutual_info_classif, percentile=5)
select.fit(np.abs(traces[:mark, :]), np.reshape(Y, (Queries,))[:mark])
indexes = select.get_support(True)
z = np.zeros(np.shape(traces)[1])
print(indexes)
z[indexes] = 1
savemat("filter_FS_MI.mat", {'filter': z})
filter_traces = np.zeros(np.shape(traces), np.complex128)
filter_traces[:, indexes] = traces[:, indexes]
filter_traces = np.fft.ifft(filter_traces, n=SAMPLES, axis=1)
snr_t = np.abs(SNR.SNR(filter_traces, Y, 256, SAMPLES, np.complex128))

fig, ax = plt.subplots()
ax.plot(range(300, 1300), snr_t[300:1300])
ax.set_title("SNR of feature selection FFT MI")
ax.set_xlabel("time")
plt.show()
savemat("snr_FS_MI.mat", {'x': np.arange(0, np.size(snr_t)), 'y': snr_t})
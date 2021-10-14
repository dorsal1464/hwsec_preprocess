from SNR import SNR
import matplotlib.pyplot as plt
import hdf5storage
from scipy.io import savemat, loadmat
import numpy as np


def plot_roc(traces, Y, filter_path, graph_sz, prefix, rng):
    crp = np.arange(rng[0], rng[1])
    SAMPLES = np.shape(traces)[1]
    Queries = np.shape(traces)[0]
    # apply filter to measurements...
    if filter_path is not None:
        filter = loadmat(filter_path)['filter']
        traces = traces * filter
    x = np.zeros(graph_sz, np.float32)
    y = np.zeros(graph_sz, np.float32)
    for i in range(0, graph_sz):
        crop = range(0, int(Queries * (i + 1) / graph_sz))
        x[i] = max(crop)
        if filter_path is not None:
            snr_t = SNR.SNR_wrapper(np.fft.ifft(traces[crop, :], SAMPLES, 1), Y[crop], 256, SAMPLES, np.complex64, frames=8)[crp]
        else:
            snr_t = SNR.SNR_wrapper(traces[crop, :], Y[crop], 256, SAMPLES, np.float32, frames=8)[crp]
        print(np.argmax(snr_t), snr_t[np.argmax(snr_t)])
        # use integral instead of maxSNR
        # y[i] = np.sum(snr_t) / np.size(snr_t)
        # use maxSNR
        y[i] = np.max(snr_t)
    savemat(prefix+"_roc.mat", {'x': x, 'y': y})
    plt.plot(x, y)
    plt.show()


def filter_freq_signal(snr_t, traces, filter_factor=0.7, save=False):
    max_snr = np.max(snr_t)
    indexes = (snr_t >= filter_factor * max_snr)

    filter_traces = np.zeros(np.shape(traces), dtype=np.complex128)
    filter_traces[:, indexes] = traces[:, indexes]
    print(np.where(snr_t >= filter_factor * max_snr))
    if save:
        filter = np.zeros(np.shape(snr_t))
        filter[indexes] = 1
        savemat("filter_SNR_FFT.mat", {'filter': filter})
    return np.fft.ifft(filter_traces, axis=1, n=SAMPLES)


if __name__ == "__main__":
    # PART 1 - load ffts
    # CHANGE ME
    path = 'Z:\\traces\\CMOS\\'
    exp = False
    FILTER_FACTOR = 0.7
    mat = hdf5storage.loadmat(path+'fft_201_301.mat')
    traces = np.complex64(mat['fft'])
    Y = mat['Y']
    if exp:
        mat = hdf5storage.loadmat(path+'fft_301_401.mat')
        traces = np.append(traces, np.complex64(mat['fft']), axis=0)
        Y = np.append(Y, mat['Y'])
        mat = hdf5storage.loadmat(path+'fft_401_501.mat')
        traces = np.append(traces, np.complex64(mat['fft']), axis=0)
        Y = np.append(Y, mat['Y'])

    Y = np.reshape(Y, (1, np.size(Y)))
    SAMPLES = np.shape(traces)[1]
    Queries = np.shape(traces)[0]
    crop = range(0, int(Queries))
    """
    filter = loadmat("Z:\\filters\\DUALRAIL\\filter_BPF.mat")['filter']
    traces = traces * filter
    snr_t = SNR.SNR_wrapper(np.fft.ifft(traces, SAMPLES, 1), Y, 256, SAMPLES, np.complex64, frames=7)
    plt.plot(snr_t)
    plt.show()
    savemat("snr_BPF.mat", {'x': range(0, SAMPLES), 'y': snr_t})
    """
    fig, ax = plt.subplots()
    # PART 2 - plot SNR of fft
    snr_t = SNR.SNR_wrapper(traces[crop, :], Y[0, crop], 256, SAMPLES, np.complex64, frames=1)
    snr_t = np.abs(snr_t)
    ax.plot(range(0, SAMPLES), snr_t)
    ax.set_title("SNR in frequency domain")
    ax.set_xlabel("f[Hz]")
    plt.show()
    savemat("SNR_FFT.mat", {'x': np.arange(0, SAMPLES), 'y': snr_t})

    # PART 3 - plot max SNR vs filter factor

    max_snr_f = np.zeros(11, dtype=np.float32)
    for i in range(1, 11):
        filter_traces = filter_freq_signal(snr_t, traces, float(i) / 10)
        snr = np.abs(SNR.SNR_wrapper(filter_traces, Y, 256, SAMPLES, np.complex128, frames=1)[300:700])
        max_snr_f[i] = np.max(snr)
        print(np.max(snr))

    fig, ax = plt.subplots()
    ax.plot(range(0, 11), max_snr_f)
    ax.set_title("max SNR vs filter threshold")
    ax.set_xlabel("filter x10")
    plt.show()

    best_threshold = np.argmax(max_snr_f)
    # PART 4 - plot SNR with MAXIMAL FACTOR

    filter_traces = filter_freq_signal(snr_t, traces, float(best_threshold / 10), save=True)
    snr = np.abs(SNR.SNR_wrapper(filter_traces, Y, 256, SAMPLES, np.complex128, frames=1))
    fig, ax = plt.subplots()
    ax.plot(range(0, SAMPLES), snr)
    ax.set_title("SNR of max filter threshold="+str(best_threshold / 10))
    ax.set_xlabel("t[sec]")
    plt.show()

    savemat("snr_SNR_FFT.mat", {'x': np.arange(0, SAMPLES), 'y': snr})

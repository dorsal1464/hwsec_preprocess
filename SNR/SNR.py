import numpy as np


# Imat = traces matrix
# Y = class of each query, here it's the sbox result
# N_classes = length of Y
# N_samples = num. of samples
# type = data type
def SNR(Imat, Y, numClassP, numSamples, type=np.float32):
    # reshape and retype Y
    Y = np.reshape(np.uint8(Y), (1, Y.size))
    _means = np.zeros((numClassP, numSamples), dtype=type)
    _vars = np.zeros((numClassP, numSamples), dtype=type)
    for y in range(0, numClassP):
        indexes = np.where(Y == y)[1]
        _means[y, :] = np.mean(Imat[indexes, :], axis=0)
        _vars[y, :] = np.var(Imat[indexes, :], axis=0)
    SNR = np.nanvar(_means, axis=0) / np.nanmean(_vars, axis=0)
    return SNR


# wrapper function: split traces by time (parameter to divide)
def SNR_wrapper(X, Y, N_classes, N_samples, type=np.float32, frames=100):
    frame_len = int(np.shape(X)[1] / frames)
    snr_t = SNR(X[:, 0:frame_len], Y, N_classes, frame_len, type)
    for frame in range(1, frames):
        snr_t = np.append(snr_t, SNR(X[:, frame_len * frame:frame_len * (frame + 1)], Y, N_classes, frame_len, type))
    return snr_t


# FFT wrapper for SNR... (for general case)
def FFT_SNR(X, Y, N_classes, N_samples, type=np.complex128):
    X_fft = np.fft.fft(X, axis=1)
    print(np.shape(X))
    return SNR_wrapper(X, Y, N_classes, N_samples, type, 10)


def Xfft_SNR(X_fft, Y, crop, SAMPLES):
    snr_t = SNR_wrapper(X_fft, Y[:crop], 256, SAMPLES, np.complex128, 10)
    # filtering if abs value is bigger than ...
    filter = np.abs(snr_t) < 3
    X_fft_filtered = X_fft
    X_fft_filtered[:, filter] = np.complex128(0)


def freq_filter(X_fft ,lower, upper):
    ans = np.zeros(np.shape(X_fft),dtype=np.complex128)
    ans[:,lower:upper] = X_fft[:,lower:upper]
    ans = np.fft.ifft(ans, axis=1)
    print('1')
    return ans


def by_freq_SNR(X_fft, Y, parts, crop, SAMPLES):
    delta = int(SAMPLES / parts)
    snr_freq = [0] * parts
    snr_freq[0] = SNR_wrapper(freq_filter(X_fft, 0, delta), Y[:crop], 256, SAMPLES, np.complex128, 10)
    for i in range(1, parts):
        tmp = SNR_wrapper(freq_filter(X_fft, i * delta, (i + 1) * delta), Y[:crop], 256, SAMPLES, np.complex128, 10)
        snr_freq[i] = tmp
        # print(i*delta, i*delta+delta)

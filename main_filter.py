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

# PART 2 - split fft by frame slices and calc max SNR


def split_by_bandwidth(traces, Y, samples, frame_num=14, show=False):
    bandwidth = int(samples / frame_num)
    frame = np.zeros(np.shape(traces), dtype=np.complex128)
    snr_l = np.zeros(frame_num, dtype=np.float32)
    for i in range(0, frame_num):
        print(str(i / frame_num * 100)+"%")
        frame = np.fft.ifft(traces[:, i*bandwidth:(i+1)*bandwidth], axis=1, n=samples)
        snr_l[i] = np.max(np.abs(SNR.SNR_wrapper(frame, Y, 256, samples, np.complex128, frames=14))[300:1300])

    if show:
        fig, ax = plt.subplots()
        ax.plot(range(0, frame_num), snr_l)
        ax.set_title("max SNR vs freq slice num, frames=" + str(frame_num))
        ax.set_xlabel("freq slice num")
        plt.show()
    return snr_l


# show maximal freq slice
split_by_bandwidth(traces, Y, SAMPLES, show=True)

# PART 3 - find optimal frame width


def plot_by_frame_num(traces, Y, samples, frames_gap, max_frames):
    frames = 0
    max_snr_by_frames = list()
    while frames <= max_frames:
        if frames > 0:
            max_snr_by_frames.append(np.max(split_by_bandwidth(traces, Y, samples, frames)))
        frames += frames_gap

    fig, ax = plt.subplots()
    ax.plot(range(frames_gap, max_frames, frames_gap), max_snr_by_frames)
    ax.set_title("max SNR vs num of freq slices")
    ax.set_xlabel("num of frames")
    plt.show()
    return (np.argmax(max_snr_by_frames) + 1) * frames_gap


# plot results
max_frame_num = plot_by_frame_num(traces, Y, SAMPLES, 10, 140)

print(max_frame_num, SAMPLES / max_frame_num)
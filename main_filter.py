from SNR import SNR
import matplotlib.pyplot as plt
import hdf5storage
from scipy.io import savemat
import numpy as np
from concurrent.futures import ThreadPoolExecutor
import pickle
import config

# PART 1 - load ffts
path = 'Z:\\traces\\shuffled_rp_8\\'
exp = False

mat = hdf5storage.loadmat(path+'fft_201_301.mat')
traces = np.complex64(mat['fft'])
Y = mat['Y']
if exp:
    mat = hdf5storage.loadmat(path+'fft_301_401.mat')
    traces = np.append(traces, np.complex64(mat['ffts']), axis=0)
    Y = np.append(Y, mat['Y'])

SAMPLES = np.shape(traces)[1]
Queries = np.shape(traces)[0]
crop = range(0, int(Queries))

# PART 2 - split fft by frame slices and calc max SNR


def split_by_bandwidth(traces, Y, samples, frame_num=14, show=False):
    bandwidth2 = int(samples / frame_num / 2)
    s2 = int(samples / 2)
    frame = np.zeros(np.shape(traces), dtype=np.complex128)
    snr_l = np.zeros(frame_num, dtype=np.float32)
    offset = np.zeros(frame_num, dtype=np.float32)
    for i in range(0, frame_num):
        print(str(i / frame_num * 100)+"% frame "+str(frame_num))
        offset[i] = bandwidth2*i
        slice = np.append(np.arange(s2-bandwidth2*(i+1) ,s2-bandwidth2*i), np.arange(s2+bandwidth2*i, s2+bandwidth2*(i+1)))
        frame = np.fft.ifft(traces[:, slice], axis=1, n=samples)
        snr_l[i] = np.max(np.abs(SNR.SNR_wrapper(frame, Y, 256, samples, np.complex128, frames=1))[300:1300])

    if show:
        fig, ax = plt.subplots()
        ax.plot(range(0, frame_num), snr_l)
        ax.set_title("max SNR vs freq slice num")
        ax.legend("frames="+str(frame_num))
        ax.set_xlabel("freq slice num")
    return offset, snr_l


def filter_by_bandwidth(traces, Y, samples, frames=54, show=False):
    bandwidth2 = int(samples / frames / 2)
    s2 = int(samples / 2)
    offset ,snr_l = split_by_bandwidth(traces, Y, samples, frames)
    i = np.argmax(snr_l)
    slice = np.append(np.arange(s2-bandwidth2*(i+1) ,s2-bandwidth2*i), np.arange(s2+bandwidth2*i, s2+bandwidth2*(i+1)))
    z = np.zeros(samples)
    z[slice] = 1
    savemat("filter_BANDWIDTH.mat", {'filter': z})
    frame = np.fft.ifft(traces[:, slice], axis=1, n=samples)
    snr_t = np.abs(SNR.SNR_wrapper(frame, Y, 256, samples, np.complex128, frames=1))
    
    if show:
        fig, ax = plt.subplots()
        ax.plot(range(0, samples), snr_t)
        ax.set_title("SNR of frame num "+str(i)+", frames=" + str(frames))
        ax.set_xlabel("freq slice num")
        plt.show()
        savemat("snr_BANDWIDTH.mat", {'x': np.arange(0, samples), 'y': snr_t})


#filter_by_bandwidth(traces, Y, SAMPLES, 50, True)
# show maximal freq slice
print(split_by_bandwidth(traces, Y, SAMPLES, 14, show=True))
print(split_by_bandwidth(traces, Y, SAMPLES, frame_num=20, show=True))
print(split_by_bandwidth(traces, Y, SAMPLES, frame_num=54, show=True))
plt.show()
# PART 3 - find optimal frame width


def plot_by_frame_num(traces, Y, samples, frames_gap, max_frames, strt=0):
    fig, ax = plt.subplots()
    frames = strt
    max_snr_by_frames = list()
    x = list()
    threads = list()
    executor = ThreadPoolExecutor(config.max_workers)
    while frames <= max_frames:
        # print("frame: " + str(frames))
        if frames > 0:
            x.append(samples / frames)
            threads.append(executor.submit(
                split_by_bandwidth, traces, Y, samples, frames))
        frames += frames_gap

    i = 0
    imax = 0
    mx = np.NINF
    for xi in x:
        y, z = threads[i].result()
        if np.max(z) > mx:
            mx = np.max(z)
            imax = i
        print({'x': x, 'y': y, 'z': z})
        max_snr_by_frames.append(z)
        ax.scatter([xi]*len(y), y, c=z, cmap=config.colormap)
        i += 1
    
    ax.set_title("max SNR vs bandwidth, offset")
    ax.set_xlabel("bandwidth")
    ax.set_ylabel("offset")
    plt.show()
    return x[imax]


# plot results
max_frame_num = plot_by_frame_num(traces, Y, SAMPLES, 1, 80, strt=4)

print(max_frame_num)

exit(0)
# PART 4 - 20 to 30

max_frame_num = plot_by_frame_num(traces, Y, SAMPLES, 2, 30, strt=20)

print(max_frame_num)

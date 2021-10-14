# from utils import traces, Y, SAMPLES
from SNR import SNR
import numpy as np
import hdf5storage
import matplotlib.pyplot as plt

exp = False
path = 'Z:\\traces\\CMOS\\'
graph_sz = 20

mat = hdf5storage.loadmat(path+'ovssa.mat')
traces = mat['traces']
Y = mat['Y']
if exp:
    mat = hdf5storage.loadmat(path+'traces_601_1101_full.mat')
    traces = np.append(traces, mat['traces'], axis=0)
    Y = np.append(Y, mat['Y'])
    mat = hdf5storage.loadmat(path + 'traces_1101_1601_full.mat')
    traces = np.append(traces, mat['traces'], axis=0)
    Y = np.append(Y, mat['Y'])
    mat = hdf5storage.loadmat(path + 'traces_1601_2101_full.mat')
    traces = np.append(traces, mat['traces'], axis=0)
    Y = np.append(Y, mat['Y'])

Y = np.reshape(Y, Y.size)
SAMPLES = np.shape(traces)[1]
Queries = np.shape(traces)[0]

# new_t, new_y = SNR.avg_traces(traces, Y, count=10)

#fig, (ax0, ax1) = plt.subplots(2)
#snr_t = np.abs(SNR.SNR_wrapper(new_t, new_y, 256, SAMPLES, frames=7))
#ax0.plot(range(300, 1300), snr_t[300:1300])
#ax0.set_title("SNR with averaging")
#ax0.set_xlabel("time[us]")
snr_t = np.abs(SNR.SNR_wrapper(traces, Y, 256, SAMPLES, frames=7))
plt.plot(range(300+80, 1294+80), snr_t)
plt.title('ovssa of 500K samples')
plt.xlabel('time')
plt.ylabel('Amp.')
#ax1.set_title("SNR")
#ax1.set_xlabel("time[us]")
plt.show()
exit(0)
ans = np.zeros(graph_sz, np.float32)
for i in range(0, graph_sz):
    crop = range(0, int(Queries*(i+1)/graph_sz))
    snr_t = SNR.SNR_wrapper(traces[crop, :], Y[crop], 256, SAMPLES, frames=14)
    # plt.plot(range(0, SAMPLES), np.abs(snr_t))
    print(np.argmax(snr_t[:1300]), snr_t[np.argmax(snr_t[:1300])])
    ans[i] = snr_t[np.argmax(snr_t[:1300])]
    # POI = 381
plt.plot(range(0, graph_sz), ans)
plt.show()

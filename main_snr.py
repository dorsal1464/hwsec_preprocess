# from utils import traces, Y, SAMPLES
from SNR import SNR
import numpy as np
import hdf5storage
import matplotlib.pyplot as plt

exp = True
path = 'Z:\\traces\\'
graph_sz = 50

mat = hdf5storage.loadmat(path+'traces_101_601_full.mat')
traces = mat['traces']
Y = mat['Y']
if exp:
    mat = hdf5storage.loadmat(path+'traces_601_1101_full.mat')
    traces = np.append(traces, mat['traces'], axis=0)
    Y = np.append(Y, mat['Y'])
    #mat = hdf5storage.loadmat(path + 'traces_1101_1601_full.mat')
    #traces = np.append(traces, mat['traces'], axis=0)
    #Y = np.append(Y, mat['Y'])

Y = np.reshape(Y, Y.size)
SAMPLES = np.shape(traces)[1]
Queries = np.shape(traces)[0]

fig, ax = plt.subplots()
snr_t = np.abs(SNR.SNR_wrapper(traces, Y, 256, SAMPLES, frames=14))
plt.plot(range(300, 1300), snr_t[300:1300])
ax.set_title("SNR")
ax.set_xlabel("time[us]")
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

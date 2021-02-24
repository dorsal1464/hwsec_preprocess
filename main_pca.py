from PCA import PCA
from SNR import SNR
from scipy.io import loadmat
from matplotlib.pyplot import *

mat = loadmat('Z:\\traces\\traces_13.mat')
traces = mat['traces']
Y = np.bitwise_xor(mat['plaintext'][:, 0], mat['key'][:, 0])
Y = np.transpose(Y)
s, out = PCA.PCA(traces, 20)

plot(out[0])
show()

snr_t = np.abs(SNR.SNR(out, Y, 256, 20, np.uint16, 0))

plot(snr_t)
show()

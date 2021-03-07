from scipy.io import loadmat, savemat
import numpy as np
from tool_box_games.Utils.sboxes import sbox_aes as sbox
from os import system
from PCA.PCA import PCA


path = 'Z:\\Onion\\OpCode_8ExpMode_1ClkDiv_3_V2'
crop = range(0, 1400)
isFFT = False
index = [101, 121]


# -------------------ACTION-------------------------
mat = loadmat(path+'\\traces_'+str(index[0])+'.mat')
if isFFT:
    traces = np.fft.fft(mat['traces'][:, crop], axis=1)
else:
    traces = mat['traces'][:, crop]
traces = PCA(traces, 20)[1]
plaintexts = mat['plaintext']
keys = mat['key']
for i in range(index[0]+1, index[1]):
    mat = loadmat(path+'\\traces_'+str(i)+'.mat')
    if isFFT:
        temp = np.fft.fft(mat['traces'][:, crop], axis=1)
    else:
        temp = mat['traces'][:, crop]
    temp = PCA(temp, 20)[1]
    traces = np.append(traces, temp, axis=0)
    plaintexts = np.append(plaintexts, mat['plaintext'], axis=0)
    keys = np.append(keys, mat['key'], axis=0)
    system("cls")
    print((i-index[0]+1) / (index[1]-index[0]) * 100, "%")

SAMPLES = np.shape(traces)[1]
Queries = np.shape(traces)[0]

Y = np.zeros((1, Queries), dtype=np.uint8)
for i in range(0, Queries):
    Y[0, i] = sbox[np.bitwise_xor(plaintexts[i][0], keys[i][0])]

print(np.shape(traces))

if isFFT:
    savemat("Z:\\traces\\pca_freq_"+str(index[0])+"_"+str(index[1])+".mat",
            {"pca": traces, "Y": np.transpose(Y), "samples": SAMPLES})
else:
    savemat("Z:\\traces\\pca_dom_" + str(index[0]) + "_" + str(index[1]) + ".mat",
            {"pca": traces, "Y": np.transpose(Y), "samples": SAMPLES})


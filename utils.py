import hdf5storage
from scipy.io import savemat, loadmat
import numpy as np
from tool_box_games.Utils.sboxes import sbox_aes as sbox
from os import system
from SNR import SNR
from matplotlib import pyplot as plt

# opcode 8 - no protection
# opcode 18, 19 - random / dual rail
# opcode 0,1 - minimal protection
# add more samples...
# change start index to 200
path = 'Z:\\Onion\\OpCode_8ExpMode_1ClkDiv_3_V2'
crop = range(0, 1400)
index = [101, 2101]


def load_traces(path, index, crop, destpath):
    print(index)
    mat = loadmat(path+'\\traces_'+str(index[0])+'.mat')
    traces = mat['traces'][:, crop]
    plaintexts = mat['plaintext']
    keys = mat['key']

    for i in range(index[0]+1, index[1]):
        mat = loadmat(path+'\\traces_'+str(i)+'.mat')
        traces = np.append(traces, mat['traces'][:, crop], axis=0)
        plaintexts = np.append(plaintexts, mat['plaintext'], axis=0)
        keys = np.append(keys, mat['key'], axis=0)
        system("cls")
        print((i-index[0]) / (index[1]-index[0]-1) * 100, "%")

    SAMPLES = np.shape(traces)[1]
    Queries = np.shape(traces)[0]

    Y = np.zeros((1, Queries), dtype=np.uint8)
    for i in range(0, Queries):
        Y[0, i] = sbox[np.bitwise_xor(plaintexts[i][0], keys[i][0])]

    if crop == range(0, 1400):
        savemat(destpath + '\\traces_' + str(index[0]) + '_' + str(index[1]) + '_full.mat',
                {'traces': traces, 'Y': Y, "samples": SAMPLES})
    else:
        savemat(destpath + '\\traces_' + str(index[0]) + '_' + str(index[1]) + '_crop.mat',
                {'traces': traces, 'Y': Y, "samples": SAMPLES})


# load_traces(path, index, crop, 'Z:\\traces\\CMOS')

load_traces('Z:\\Onion\\OpCode_0ExpMode_1ClkDiv_3_V2', [5101, 5601], crop, 'Z:\\traces\\RAND')
load_traces('Z:\\Onion\\OpCode_0ExpMode_1ClkDiv_3_V2', [5601, 6101], crop, 'Z:\\traces\\RAND')
load_traces('Z:\\Onion\\OpCode_0ExpMode_1ClkDiv_3_V2', [6101, 6601], crop, 'Z:\\traces\\RAND')
load_traces('Z:\\Onion\\OpCode_0ExpMode_1ClkDiv_3_V2', [6601, 7101], crop, 'Z:\\traces\\RAND')
load_traces('Z:\\Onion\\OpCode_0ExpMode_1ClkDiv_3_V2', [7101, 7601], crop, 'Z:\\traces\\RAND')
load_traces('Z:\\Onion\\OpCode_0ExpMode_1ClkDiv_3_V2', [7601, 8101], crop, 'Z:\\traces\\RAND')
load_traces('Z:\\Onion\\OpCode_0ExpMode_1ClkDiv_3_V2', [8101, 8601], crop, 'Z:\\traces\\RAND')
load_traces('Z:\\Onion\\OpCode_0ExpMode_1ClkDiv_3_V2', [8601, 9101], crop, 'Z:\\traces\\RAND')
load_traces('Z:\\Onion\\OpCode_0ExpMode_1ClkDiv_3_V2', [9101, 9601], crop, 'Z:\\traces\\RAND')










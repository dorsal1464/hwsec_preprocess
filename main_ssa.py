
from SSA import SSA
from scipy.io import loadmat
from matplotlib.pyplot import *
from SNR import SNR
from PCA import PCA
from concurrent.futures import ThreadPoolExecutor
import config

executor = ThreadPoolExecutor(config.max_workers)

mat = loadmat('/home/salomod/Desktop/PROJECT/traces/traces_101_601_full.mat')
traces = mat['traces']
Y = mat['Y']
Y = np.reshape(Y, (1, np.size(Y)))
queries = np.shape(traces)[0]
samples = np.shape(traces)[1]


def ssa_wrapper(trace, length, a, b):
    ss, types, ssa = SSA.SSA(trace, length, a, b)
    n = np.zeros(np.shape(ssa[0]))
    t = np.zeros(np.shape(ssa[0]))
    s = np.zeros(np.shape(ssa[0]))
    for i in range(0,len(types)):
        if types[i] == 0:
            t += ssa[i]
        elif types[i] == 1:
            s += ssa[i]
        else:
            n += ssa[i]
    return s, t, n


def parse_ssa(traces, Y, crop, a, b):
    samples = np.shape(traces)[1]
    print(len(crop))
    ssa_mat = np.zeros((len(crop), samples-1), np.float32)
    futures = list()
    for j in crop:
        futures.append(executor.submit(ssa_wrapper, traces[j, :], 20, a, b))

    j = 0
    for f in futures:
        s, t, n = f.result()
        ssa_mat[j, :] = s + t
        print(j / len(crop) * 100, '%')
        j = j + 1
    
    #ssa_mat = PCA.multi_PCA(ssa_mat, 20)
    #samples = 20
    snr = SNR.SNR_wrapper(ssa_mat, Y[0, crop], 256, samples-1, frames=14)
    plot(range(0, len(snr)), snr)
    show()
    return np.max(snr)

b = 0.6
snr_l = list()
rng = np.arange(0.2, b, 1)
# 0.0040237885 -- 100k
for i in rng:
    crop = range(0, 100000)
    snr_l.append(parse_ssa(traces[:, 300:1300], Y, crop, i, b))

print({'a': rng, 'b': b, 'snr': snr_l})

close()
plot(rng, snr_l, 'o-')
xlabel('fraction a')
title('maxSNR of SSA vs a, b = '+str())
show()

exit(0)

ss, types, ssa = SSA.SSA(traces[1], 20)
s, t, n = ssa_wrapper(traces[1], 20)
subplot(311)
plot(range(0,20),ss, 'o-')
title("eigenvalues")
xlabel("t")
# let's see the results...
subplot(312)
for i in range(0, 20):
    # print(ssa[i])
    plot(ssa[i], '.-')
title("components")
xlabel("t")
subplot(313)
plot(traces[1])
title("original trace")
xlabel("t")
show()

# show s, t, n together
subplot(311)
plot(s)
title("ocsillations")
xlabel("t")
subplot(312)
plot(t)
title("trends")
xlabel("t")
subplot(313)
plot(n)
title("noise")
xlabel("t")
show()
# remove noise
subplot(211)
title("removed noise")
plot(s+t)
subplot(212)
plot(traces[1])
title("original trace")
show()

# TODO: check various modes / trends oscilations without noise
#       change threshholds, use at least 200 files to process
#       modes include n, s, t, s+t, 1/2s+t, 1/2t+s, try different groupings
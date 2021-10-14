
from SSA import SSA
from scipy.io import loadmat, savemat
from matplotlib.pyplot import *
from SNR import SNR
from PCA import PCA
from concurrent.futures import ThreadPoolExecutor
import config
from SSA.SSA import ssa_wrapper
import time

executor = ThreadPoolExecutor(30)

mat = loadmat('Z:\\traces\\CMOS\\traces_101_601_full.mat')
traces = mat['traces']
Y = mat['Y']
Y = np.reshape(Y, (1, np.size(Y)))
queries = np.shape(traces)[0]
samples = np.shape(traces)[1]


def parse_ssa(traces, Y, crop, a, b):
    samples = np.shape(traces)[1]
    print(len(crop))
    ssa_mat = np.zeros((len(crop), samples), np.float32)
    futures = list()
    for j in crop:
        futures.append(executor.submit(SSA.OVSSA, traces[j, :], None, 201, 100))
    print(".")
    j = 0
    for f in futures:
        s = f.result()
        ssa_mat[j, :] = s
        print(j / len(crop) * 100, '%')
        j = j + 1
    # done from here
    savemat("rp8_ssa.mat", {'traces': ssa_mat, 'Y': Y, 'samples': np.shape(ssa_mat)[1]})
    #ssa_mat = PCA.multi_PCA(ssa_mat, 20)
    #samples = 20
    snr = SNR.SNR_wrapper(ssa_mat, Y[0, crop], 256, samples-1, frames=14)
    plot(range(0, len(snr)), snr)
    show()
    return np.max(snr)


# parse_ssa(traces, Y, range(0, queries), 0.2, 0.6)
# 0.0040237885 -- 100k
#rc('text', usetex=True)
ss, types, ssa = SSA.SSA(traces[1], None)
ti = time.time()
s, t, n = ssa_wrapper(traces[1], None)
print(time.time()-ti)
subplot(311)
plot(range(0, 20), ss[:20], 'o-')
title(r'Eigenvalues of $S$')
xlabel(r'$i$')
ylabel(r'$\lambda_i$')
# let's see the results...
subplot(312)
lgnd = []
for i in range(0, len(types)):
    # print(ssa[i])
    plot(ssa[i], '-')
    if types[i] == 0:
        lgnd.append('trend')
    elif types[i] == 1:
        lgnd.append('oscillation')
    else:
        lgnd.append('noise')
legend(
    lgnd[:6],
    loc='right',
    bbox_to_anchor=(1.2, 0.5),
    fontsize='x-small'
    )
title("SSA components")
xlabel("Time [us]")
ylabel("Quantized current")
subplot(313)
plot(traces[1])
title("Original trace")
xlabel("Time [us]")
ylabel("Quantized current")
tight_layout(pad=1.1)
savefig('ssa1.pdf')
show()

# show s, t, n together
subplot(311)
plot(s)
title("ocsillations")
xlabel("time")
ylabel("Quantized current")
subplot(312)
plot(t)
title("trends")
xlabel("time")
ylabel("Quantized current")
subplot(313)
plot(n)
title("noise")
xlabel("time")
ylabel("Quantized current")
tight_layout(pad=1.0)
savefig('ssa2.pdf')
show()
# remove noise
tm = time.time()
ovssa = SSA.OVSSA(traces[1], None, 201, 100)
print(time.time()-tm)
title("SSA vs OVSSA trace")
xlabel('time')
ylabel("Quantized current")
plot((s+t)[50:], alpha=0.7)
plot(ovssa, alpha=0.5)
legend(['ssa', 'ovssa'])
tight_layout(pad=1.1)
savefig('ssavsovssa.pdf')
show()


# TODO: check various modes / trends oscilations without noise
#       change threshholds, use at least 200 files to process
#       modes include n, s, t, s+t, 1/2s+t, 1/2t+s, try different groupings
from SSA import SSA
from scipy.io import loadmat
from matplotlib.pyplot import *

mat = loadmat('Z:\\traces\\traces_13.mat')
traces = mat['traces']


ss, types, ssa = SSA.SSA(traces[1], 20, 4)
print(types)

n = np.zeros(np.shape(ssa[0]))
t = np.zeros(np.shape(ssa[0]))
s = np.zeros(np.shape(ssa[0]))
for i in range(0,15):
    if types[i] == 0:
        t += ssa[i]
    elif types[i] == 1:
        s += ssa[i]
    else:
        n += ssa[i]

subplot(311)
plot(range(0,20),ss, 'o-')
title("eigenvalues")
xlabel("t")
# let's see the results...
subplot(312)
for i in range(0,20):
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

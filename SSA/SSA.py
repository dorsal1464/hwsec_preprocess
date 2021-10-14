import numpy as np


def diag_avg(mat):
    ans = np.zeros(np.sum(np.shape(mat))-1)
    for val in range(0, np.size(ans)):
        count = 0
        s = 0
        for i in range(0, np.array([val+1, np.shape(mat)[0]]).min()):
            j = val-i
            if j < np.shape(mat)[1] and i < np.shape(mat)[0]:
                s += mat[i, j]
                count += 1
        ans[val] = s / count
    return ans


def SSA(trace, L=None, a=0.2, b=0.6):
    # create henkel matrix of the signal
    c = 2
    n = np.size(trace)
    if L is not None:
        W = L
    else:
        W = int(np.log(n) ** c)
    # print(W)
    D = n - W + 1
    X = np.zeros((D, W))
    for i in range(0, W):
        indexes = np.arange(i, i+D) % n
        X[:, i] = trace[indexes]
    # the shorter way is to SVD(X)
    u, s, vh = np.linalg.svd(X)
    # s are the singular values in decending order
    # the columns of u are the eigenvectors of XX^t
    # the rows of vh are the eigenvectors of X^t X
    # calculate vectors of principal components
    V = list()
    for i in range(0, W):
        uu = np.reshape(u[:, i], (np.size(u[:, i]), 1))
        Y = np.matmul(uu, np.transpose(uu))
        V.append(np.matmul(Y, X))
    #  eigentriple grouping. (?)
    # diag. avging
    elements = list()
    for i in range(0, np.size(s)):
        elements.append(diag_avg(V[i]))
    ll = np.size(elements[0])
    # attempting manual grouping...
    return group(s, elements, ll, a, b)


def ssa_wrapper(trace, length, a=0.2, b=0.6):
    ss, types, ssa = SSA(trace, length, a, b)
    n = np.zeros(np.shape(ssa[0]))
    t = np.zeros(np.shape(ssa[0]))
    s = np.zeros(np.shape(ssa[0]))
    for i in range(0, len(types)):
        if types[i] == 0:
            t += ssa[i]
        elif types[i] == 1:
            s += ssa[i]
        else:
            n += ssa[i]
    return s, t, n


def OVSSA(trace, L, Z, q):
    ssa = np.zeros(np.shape(trace))
    n = np.size(trace)
    for i in range(0, n, q):
        #print("i - ", i)
        #print(min(i+Z, n))
        # print(min(i, n-Z), ":", min(i+Z, n))
        s, t, noise = ssa_wrapper(trace[min(i, n-Z) : min(i+Z, n)], L)
        tmp = s + t
        # print("offset - ", int(Z/2-q/2))
        # print(i, ":", i+q)
        ssa[i : i+q] = tmp[int(Z/2-q/2):int(Z/2-q/2)+q]
    return ssa


def group(s, elements, n, a, b):
    dt = 1
    derivate = np.abs(np.diff(s) / dt)
    max_der = derivate.max()
    ans = list()
    types = list()
    for i in range(0, np.size(s)):
        ans.append(np.zeros(n))
    index = -1
    typ = 0         # (0, 1, 2) - trend, osci, noise
    ans[0] += elements[0]
    for i in range(0, np.size(s)-1):
        if derivate[i] > b*max_der:
            if typ != 0:
                index += 1
                types.append(typ)
                typ = 0
        elif derivate[i] < a*max_der:
            if typ != 1:
                index += 1
                types.append(typ)
                typ = 1
        else:
            if typ != 2:
                index += 1
                types.append(typ)
                typ = 2
        ans[index] += elements[i+1]
    return s, types, ans


if __name__ == '__main__':
    x = np.arange(0, 500)
    y = 4*np.sin(4*x) + 2*np.cos(0.5*x) + np.random.randn(500)*0.2
    ss, typ, tr = SSA(y, None, 0.2, 0.66)
    # s,t,n = ssa_wrapper(y, None)
    from matplotlib import pyplot as plt
    fig, axes = plt.subplots(2, 1)
    axes[0].plot(ss)
    plt.title('SSA Decomposition')
    i = 0
    lgnd = []
    print(typ)
    for item in tr[:len(typ)]:
        axes[1].plot(item)
    axes[1].legend(typ)
    # axes[1].legend(['trend', 'periodic', 'noise'])
    plt.show()
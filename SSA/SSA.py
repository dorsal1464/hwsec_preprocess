import numpy as np


def diag_avg(mat, n):
    ans = np.zeros(n-1)
    for val in range(0, n-1):
        count = 0
        s = 0
        for i in range(0, np.array([val+1, np.shape(mat)[0]]).min()):
            j = val-i
            if j < np.shape(mat)[1] and i < np.shape(mat)[0]:
                s += mat[i, j]
                count += 1
        ans[val] = s / count
    return ans


def SSA(trace, L, a=0.1, b=0.5):
    # create henkel matrix of the signal
    n = np.size(trace)
    X = np.zeros((L,n-L))
    X_t = np.transpose(X)
    for i in range(0, n-L):
        X[:,i] = trace[i:(i+L)]
    # the shorter way is to SVD(X)
    u, s, vh = np.linalg.svd(X)
    # s are the singular values in decending order
    # the columns of u are the eigenvectors of XX^t
    # the rows of vh are the eigenvectors of X^t X
    # calculate vectors of principal components
    V = list()
    for i in range(0, L):
        uu = np.reshape(u[:, i], (np.size(u[:, i]), 1))
        Y = np.matmul(uu, np.transpose(uu))
        V.append(np.matmul(Y, X))
    #  eigentriple grouping. (?)
    # diag. avging
    elements = list()
    for i in range(0, L):
        elements.append(diag_avg(V[i], n))
    # attempting manual grouping...
    return group(s, elements, n, L, a, b)


def group(s, elements, n, L, a, b):
    dt = 1
    derivate = np.diff(s) / dt
    max_der = derivate.min()
    ans = list()
    types = list()
    for i in range(0, L):
        ans.append(np.zeros(n-1))
    index = -1
    typ = 0         # (0, 1, 2) - trend, osci, noise
    ans[0] += elements[0]
    for i in range(0, L-1):
        if derivate[i] < b*max_der:
            if typ != 0:
                index += 1
                types.append(typ)
                typ = 0
        elif derivate[i] < a*max_der:
            if typ != 2:
                index += 1
                types.append(typ)
                typ = 2
        else:
            if typ != 1:
                index += 1
                types.append(typ)
                typ = 1
        ans[index] += elements[i+1]
    return s, types, ans

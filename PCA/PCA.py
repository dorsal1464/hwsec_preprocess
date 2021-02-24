import numpy as np


def PCA(X, dim):
    # standardize the data
    # X = (X - np.mean(X)) / np.std(X)
    # calc cov matrix
    # cov = np.cov(X)
    # np.linalg.eigenvec(X) np.linalg.eigenval(X)
    # the shorter way is to SVD(X)
    u, s, vh = np.linalg.svd(X)
    # s are the singular values in decending order
    # the columns of u are the eigenvectors of XX^t
    # the rows of vh are the eigenvectors of X^t X
    vh = np.transpose(vh)
    # now calc vh^t so the columns are the eigenvectors of X^t X
    return s[:dim], np.matmul(X, vh[:, :dim])
# return (s, pca_mat)

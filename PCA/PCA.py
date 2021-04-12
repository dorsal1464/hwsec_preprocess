import numpy as np


class PCA:
    def __init__(self, X):
        self.mat = X
        self._mean = np.mean(X)
        self._std = np.std(X)
        self.vh = None

    def fit(self, dim):
        # standardize the data
        mat = (self.mat - self._mean) / self._std
        # calc cov matrix
        cov = np.matmul(np.transpose(mat), mat)
        s, vh = np.linalg.eig(cov)

        # the shorter way is to SVD(X), but too compute. expensive
        # u, s, vh = np.linalg.svd(X)
        # s are the singular values in decending order
        # the columns of u are the eigenvectors of XX^t
        # the rows of vh are the eigenvectors of X^t X
        # vh = np.transpose(vh)
        # now calc vh^t so the columns are the eigenvectors of X^t X
        return s[:dim], np.matmul(mat, vh[:, :dim])

    def transform(self):
        # standardize the data
        mat = (self.mat - self._mean) / self._std
        # calc cov matrix
        cov = np.matmul(np.transpose(mat), mat)
        s, vh = np.linalg.eig(cov)
        return np.matmul(mat, vh)

    def itransform(self, mat):
        return np.matmul(mat, np.transpose(self.vh)) * self._std + self._mean

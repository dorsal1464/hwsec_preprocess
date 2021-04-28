import numpy as np


class PCA:
    def __init__(self, X):
        self.mat = X
        self._mean = np.mean(X)
        self._std = np.std(X)
        self.vh = None
        self.s = None

    def fit(self, dim):
        # standardize the data
        mat = (self.mat - self._mean) / self._std
        # calc cov matrix
        cov = np.matmul(np.transpose(mat), mat)
        s, vh = np.linalg.eig(cov)
        self.vh = vh
        self.s = s
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
        self.s, self.vh = np.linalg.eig(cov)
        return np.matmul(mat, self.vh)

    def itransform(self, mat):
        return np.matmul(mat, np.transpose(self.vh)) * self._std + self._mean

    def feature_select(self, dim):
        self.transform()
        ranking = np.sum(self.vh, axis=1)
        print(ranking)
        # assuming that the important features will appear in more eigen vectors...
        best_indexes = np.argsort(ranking)[::-1]
        print(best_indexes)
        return best_indexes[:dim]



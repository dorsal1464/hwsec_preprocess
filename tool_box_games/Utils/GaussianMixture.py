import numpy as np
import scipy.stats


class GaussianMixture:

    def __init__(self,alpha,means,variances):
        """
            Gaussian mixture of dimenssios Nd

            alpha: the probability of each of the modes (Nm)
            means: Nm*Nd array with each row represention the mean of a mode
            variances: Nm*Nd*Nd, each sub Nd*Nd arrays are the varaiances of a mode
        """

        self._alpha = np.array(alpha)
        self._means = np.array(means)
        self._variances = np.array(variances)

    def sample(self,N):
        """
            Generates N samples from the Gaussian Mixture
        """
        def myfunc(i):
            modes = np.random.choice(np.arange(0, len(self._alpha),dtype=int),1, p=self._alpha)[0].astype(np.int)
            samples = np.random.multivariate_normal(self._means[modes],self._variances[modes])
            return samples

        vfunc = np.vectorize(myfunc,signature="()->(k)")

        return vfunc(np.arange(N))

    def pdf(self,x):
        """
            Returns the pdf of Gaussian Mixture
        """
        pdf = 0
        for i in range(len(self._alpha)):
            pdf += self._alpha[i] * scipy.stats.multivariate_normal.pdf(x,self._means[i],self._variances[i])
        return pdf

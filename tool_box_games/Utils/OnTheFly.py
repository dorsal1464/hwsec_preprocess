import numpy as np
from sklearn import linear_model
from tool_box_games.Utils import GaussianMixture
from tool_box_games.Utils.GaussianMixture import GaussianMixture

class Corr:

	def __init__(self):
		self._sXiYi = 0
		self._sYiYi = 0
		self._sYi = 0
		self._sXiXi = 0
		self._sXi = 0

		self._N = 0
		self._corr = 0

	def __del__(self):
		del self._sXiYi
		del self._sYiYi
		del self._sYi
		del self._sXiXi
		del self._sXi

		del self._N
		del self._corr

	def fit(self,X,Y):
		if X.shape == Y.shape:
			print("same shape")
			self._sXiYi += np.sum(X*Y,axis=0)
		else:
			self._sXiYi += np.sum((X.T*Y).T,axis=0)
		self._sYiYi += np.sum(Y*Y,axis=0)
		self._sYi += np.sum(Y,axis=0)
		self._sXiXi += np.sum(X*X,axis=0)
		self._sXi += np.sum(X,axis=0)

		self._N += len(Y)

		mX = self._sXi/(self._N)
		mY = self._sYi/(self._N)

		num = self._sXiYi - mY*self._sXi - mX*self._sYi + self._N*mY*mX
		den = np.sqrt( (self._sXiXi + self._N*mX*mX - 2*mX*self._sXi) * (self._sYiYi + self._N*mY*mY - 2*mY*self._sYi))
		self._corr = num/den

		return self._corr

class Cov:
	def __init__(self):
		self._N = 0
		self._u = 0
		self._Ex2 = 0
		self._S = 0
		# self._sum = 0
		# self._mul = 0

	def fit(self,X):
		N = len(X[:,0])
		self._N += N

		C = np.cov(X.T)
		if np.linalg.det(C)<0:
			print("det negative")
		u = np.mean(X,axis=0)
		u = np.expand_dims(u,1)
		Ex2 = C + np.dot(u,u.T)

		self._Ex2 += (Ex2*N)
		self._u += (u*N)

		# return Ex2 -  np.dot(u,u.T)
		self._S = (self._Ex2/self._N - np.dot(self._u/self._N,self._u.T/self._N))

		return self._S


		# self._sum += np.sum(X,axis=0)
		# self._mul += np.dot(X.T,X)
		#
		# p = len(self._mul)
		# u = self._sum/self._N
		# u = np.expand_dims(u,1)
		# S = (self._mul)/self._N - np.dot(u,u.T) #(np.tile(u,(p,1)).T*u).T
		# self._S = S
		# self._u = u

		return S

class SNR:
	def __init__(self,N_classes,N_samples,type=np.float32):
		self._N_classes = N_classes
		self._N_samples = N_samples
		self._ns = np.zeros((N_classes),dtype=type)
		self._sum = np.zeros((N_classes,N_samples),dtype=type)
		self._sum2 = np.zeros((N_classes,N_samples),dtype=type)
		self._means = np.zeros((N_classes,N_samples),dtype=type)
		self._vars= np.zeros((N_classes,N_samples),dtype=type)
		self._means[:,:] = np.nan
		self._SNR = np.zeros(N_samples,dtype=type)

		self._i = 0
	def __del__(self):
		del self._ns
		del self._sum
		del self._sum2

	def fit(self,X,Y):
		for c in np.unique(Y%self._N_classes):
			indexes = np.where((Y%self._N_classes)==c)[0]
			self._ns[c] += len(indexes)
			self._sum[c,:] += np.sum(X[indexes,:],axis=0)
			self._sum2[c,:] += np.sum(X[indexes,:]**2,axis=0)

			self._means[c,:] = self._sum[c,:] / self._ns[c]
			self._vars[c,:] = (self._sum2[c,:]/self._ns[c]) - (self._means[c,:]**2)

		self._SNR[:] = np.var(self._means,axis=0)/np.mean(self._vars,axis=0)

		return self._SNR

class LR:

	def __init__(self,N_bits,N_samples,Nd=1):
		self._M = np.zeros((N_samples,N_bits),dtype=np.uint8)
		self._X = np.zeros((N_samples,Nd))
		self._i = 0
		self._N_bits = N_bits

	def load(self,M,X):
		indexes = np.arange(len(M[:,0]))+self._i
		self._M[indexes,:] = M
		self._X[indexes] = X
		self._i += len(M[:,0])

	def fit(self,N_bits=None):
		if N_bits is None:
			N_bits = self._N_bits
		self._regr = linear_model.LinearRegression()
		self._regr.fit(self._M[:self._i,-N_bits:],self._X[:self._i])

	def predict(self,M,N_bits=None):
		if N_bits is None:
			N_bits = self._N_bits
		#return np.sum(M[:,-N_bits:],axis=1)
		return self._regr.predict(M[:,-N_bits:])

	def get_coef(self):
		return self._regr.coef_

class GT:
	""" Used to compute Gaussian Templates based on
		Observations"""

	def __init__(self,Nk=256,Nd=1,bin_length=256,cov=True):
		bins = (range(Nk+1),)
		for i in range(Nd):
			bins = bins+(range(bin_length+1),)

		self._bins = bins
		self._Nks = np.zeros(Nk,dtype=np.float64)
		self._sums = np.zeros((Nk,Nd),dtype=np.float64)
		if cov:
			self._muls = np.zeros((Nk,Nd,Nd),dtype=np.float64)
		self._Nk = Nk
		self._Nd = Nd
		self._cov = cov

	def fit(self,traces,keys):
		traces = traces[:,:self._Nd]
		N = np.zeros(self._Nk)
		sums = np.zeros((self._Nk,self._Nd))
		if self._cov:
			mults = np.zeros((self._Nk,self._Nd,self._Nd))

		for k in range(self._Nk):
			indexes = np.where(keys==k)[0]
			self._Nks[k] += len(indexes)
			self._sums[k,:] += np.sum(traces[indexes,:],axis=0)
			if self._cov:
				self._muls[k,:] += (np.dot(traces[indexes,:].T,traces[indexes,:]))

	def get_template(self):
		means = np.zeros((self._Nk,self._Nd))
		vars = np.zeros((self._Nk,self._Nd,self._Nd))

		for k in range(self._Nk):
			u = self._sums[k]/self._Nks[k]

			if self._cov:
				N = self._Nd
				var = (self._muls[k,:]/self._Nks[k]) - (np.tile(u,(N,1)).T*u).T
			else:
				var = 0

			means[k,:] = u
			vars[k,:,:] = var #np.maximum(var,1E-100)

		return means,vars

	def to_GM(self):
		means,vars = self.get_template()
		alphas = np.array([1])
		GM = np.array([GaussianMixture(alphas,np.array([means[k,:]]),np.array([vars[k,:]])) for k in range(self._Nk)])
		return GM

	def input_dist(self):
		return self._Nks/np.sum(self._Nks)

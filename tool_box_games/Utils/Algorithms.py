import sys
import scipy
import numpy as np

def EM(samples,N_modes,N_it=100):

	Ndim = len(samples[0,:])
	N_samples = len(samples)
	rho = np.ones(N_modes)/N_modes

	us = np.random.uniform(low=np.min(samples),high=np.max(samples),size=(N_modes,Ndim))
	vs = np.identity(N_modes) * np.var(samples)

	cov = np.cov(samples.T)
	if Ndim == 1:
		cov = np.array([[np.cov(samples.T)]])

	vs = np.tile(cov,(N_modes,1,1))

	for it in range(N_it+1):
		# the probability of each samples in each mode
		fs = np.zeros((N_modes,N_samples))
		remove = ()
		for j in range(N_modes):
			try:
				fs[j,:] = scipy.stats.multivariate_normal.pdf(samples,us[j],vs[j])
			except:
				# print("Singular Cov Matrix in EM -> Removing one mode %d"%(N_modes),file=sys.stderr)
				remove += (j,)

		j = remove
		N_modes -= len(remove)
		rho = np.delete(rho,j)
		rho = rho/np.sum(rho)
		us = np.delete(us,j,0)
		vs = np.delete(vs,j,0)
		fs = np.delete(fs,j,0)
		if it == N_it:
			break

		# computing the T
		T = (fs.T * rho).T
		T = T/np.sum(T,axis=0)
		if np.isnan(T).any():
			break

		#updating the rho
		rho = np.sum(T,axis=1)/N_samples

		for j in range(N_modes):

			# Compute the updated mean
			num = (samples.T*T[j]).T
			us[j] = np.sum(num,axis=0)/np.sum(T[j,:])

			# Compute the update Covariance
			V = (samples-us[j])
			num = np.dot((V.T*T[j,:]),V)
			vs[j] = num/np.sum(T[j])


	return rho,us,vs

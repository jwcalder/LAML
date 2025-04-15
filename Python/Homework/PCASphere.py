# %%
import numpy as np

#Draw uniformly sampled points on unit sphere
n = 50000
Y = 2*np.random.rand(n,3)-1
norms = np.linalg.norm(Y,axis=1)
inds = norms <= 1
X = Y[inds,:]
X = X/norms[inds,None]

mu = 0.95 #Fraction of variance to capture

#Find points close to north pole
#Need r <= 1 to see the 2-dimensional nature of the sphere
for r in np.arange(2,0.1,-0.1):
    Y = X - [0,0,1]
    norms = np.linalg.norm(Y,axis=1)
    Z = X[norms <= r,:]

    #PCA
    Zc = Z - np.mean(Z,axis=0)
    vals,vecs = np.linalg.eigh(Zc.T@Zc)
    trace = np.sum(vals)
    var_capt = np.array([vals[2]/trace, (vals[1] + vals[2])/trace])
    dim = 3
    if var_capt[1] > 0.95:
        dim = 2
    if var_capt[0] > 0.95:
        dim = 1
    print('\nRadius = ',r)
    print('Variance Captured = ', var_capt)
    print('PCA Dimension = ', dim)

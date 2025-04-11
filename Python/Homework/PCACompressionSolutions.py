# %%
pip install graphlearning

# %%
from scipy import sparse
import numpy as np
from scipy import linalg
import matplotlib.pyplot as plt
import graphlearning as gl

img = gl.datasets.load_image('chairtoy')
X = np.hstack((img[:,:,0],img[:,:,1],img[:,:,2]))

#How many singular vectors to compute
num_eig = 100

#SVD (order of singular values not guaranteed so we have to sort)
P,S,QT = sparse.linalg.svds(X,k=num_eig)
ind = np.argsort(-S)
Q = QT[ind,:].T #Scipy returns the SVD transposed
S = S[ind]

#SVD block-wise
m = 8
X = gl.utils.image_to_patches(img,patch_size=(m,m))
Vals, Q = np.linalg.eigh(X.T@X)
Vals = Vals[::-1]

plt.plot(S, label='rowwise')
plt.plot(np.sqrt(Vals)[:100], label='blockwise')
plt.yscale('log')
plt.legend()
plt.title('Singular Values')
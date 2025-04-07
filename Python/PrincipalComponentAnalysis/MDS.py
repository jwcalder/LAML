# %%
"""
# Multidimensional Scaling (MDS) 

This notebook gives a brief introduction to Multidimensional Scaling (MDS). Let us first define some helper functions that will compute MDS for us and install graphlearning.
"""

# %%
#pip install -q graphlearning

# %%
import numpy as np
from scipy import sparse

#Classical multidimensional scaling 
def mds(H,k=2,center=False):

    #Only center for distance matrices
    if center:
        n = H.shape[0]
        J = np.eye(n) - (1/n)*np.ones((n,n))
        H = -0.5*J@H@J

    #Need to sort eigenvalues, since H may not be positive semidef
    vals,V = sparse.linalg.eigsh(H,k=10*k,which='LM')
    ind = np.argsort(-vals)
    V = V[:,ind]
    vals = vals[ind]

    #Get top eigenvectors and square roots of positive parts of eigenvalues
    P = V[:,:k]
    S = np.maximum(vals[:k],0)**(1/2)

    #Return MDS embedding
    return P@np.diag(S)

# %%
"""
We first consider a couple of toy problems. Play around with the dimension parameters or come up with examples yourself.
"""

# %%
import matplotlib.pyplot as plt
import graphlearning as gl
from sklearn.metrics import pairwise
import numpy as np
plt.ion()

#Toy data on the sphere in d dimensions
n = 1000
d = 3
X = gl.utils.rand_ball(n,d)
X = X/np.linalg.norm(X,axis=1)[:,None]

#MDS using pairwise distances
D = pairwise.euclidean_distances(X,squared=True)
P = mds(D,k=2,center=True)
plt.figure()
plt.title('High dimensional sphere')
plt.scatter(P[:,0],P[:,1])

#Parabola in high dimensions
n = 1000
d = 10
X = np.zeros((n,d))
X[:,0] = np.linspace(-1,1,n)
X[:,-1] = X[:,0]**2

#MDS using pairwise distances
D = pairwise.euclidean_distances(X,squared=True)
P = mds(D,k=2,center=True)
plt.figure()
plt.title('Parabola')
plt.scatter(P[:,0],P[:,1])


# %%
"""
Let's now run this on MNIST and compare to PCA.
"""

# %%
import graphlearning as gl
import numpy as np
import matplotlib.pyplot as plt

#Load MNIST data and subset to a random selection of 5000 images
data, labels = gl.datasets.load('mnist')
ind = np.random.choice(data.shape[0],size=5000)
data = data[ind,:]
labels = labels[ind]

#Subset to a smaller number of digits
num = 3   #Number of digits to use
X = data[labels < num] #subset to 0s and 1s
L = labels[labels < num] #corresponding labels

#MDS
S = pairwise.cosine_similarity(X)
P = mds(S,k=2,center=False)
plt.figure()
plt.title('Cosine Similarity')
plt.scatter(P[:,0],P[:,1],c=L,s=10)

n = X.shape[0]
E = pairwise.euclidean_distances(X,squared=True)/n
S = np.exp(-E)
P = mds(S,k=2,center=False)
plt.figure()
plt.title('Gaussian Similarity')
plt.scatter(P[:,0],P[:,1],c=L,s=10)


# %%
"""
## Exercises
1. Apply MDS to another data set in graphlearning, like 'fashionmnist'.
2. Apply MDS to an sklearn dataset.
2. Compare against PCA and LDA from previous notebooks.
"""




# %%
"""
#t-SNE embedding

This notebook explores the t-SNE embedding method for visualizing high dimensional data.
"""

# %%
#pip install graphlearning annoy

# %%
"""
Below is code for implementing t-SNE from scratch. This only works on small data sets, but is useful for understanding how the algorithm works and playing around with the code.
"""

# %%
def perp(p):
    "Perplexity"

    p = p + 1e-10
    return 2**(-np.sum(p*np.log2(p),axis=1))

def pmatrix(X,sigma):
    "P matrix in t-SNE"

    n = len(sigma)
    I = np.zeros((n,n), dtype=int)+np.arange(n, dtype=int)
    dist = np.sum((X[I,:] - X[I.T,:])**2,axis=2)
    W = np.exp(-dist/(2*sigma[:,np.newaxis]**2))
    W[range(n),range(n)]=0
    deg = W@np.ones(n)
    return np.diag(1/deg)@W   #P matrix for t-SNE

def bisect(X,perplexity):
    "Bisection search to find sigma for a given perplexity"

    m = X.shape[0]
    sigma = np.ones(m)
    P = pmatrix(X,sigma)
    while np.min(perp(P)) < perplexity:
        sigma *= 2
        P = pmatrix(X,sigma)

    #bisection search
    sigma1 = np.zeros_like(sigma)
    sigma2 = sigma.copy()
    for i in range(20):
        sigma = (sigma1+sigma2)/2
        P = pmatrix(X,sigma)
        K = perp(P) > perplexity
        sigma2 = sigma*K + sigma2*(1-K)
        sigma1 = sigma1*K + sigma*(1-K)

    return sigma

def GL(W):
    "Returns Graph Laplacian for weight matrix W"
    deg = W@np.ones(W.shape[0])
    return np.diag(deg) - W

def tsne(X,perplexity=50,h=1,alpha=50,num_early=100,num_iter=1000):
    """t-SNE embedding

    Args:
        X: Data cloud
        perplexity: Perplexity (roughly how many neighbors to use)
        h: Time step
        alpha: Early exaggeration factor
        num_early: Number of early exaggeration steps
        num_iter: Total number of iterations

    Returns:
        Y: Embedded points
    """

    #Build graph using perplexity
    m = X.shape[0]
    sigma = bisect(X,perplexity)
    P = pmatrix(X,sigma)
    P = (P.T + P)/(2*m)

    #For indexing
    I = np.zeros((m,m), dtype=int)+np.arange(m, dtype=int)

    #Initialization
    Y = np.random.rand(X.shape[0],2)

    #Main gradient descent loop
    for i in range(num_iter):

        #Compute embedded matrix Q
        q = 1/(1+np.sum((Y[I,:] - Y[I.T,:])**2,axis=2))
        q[range(m),range(m)]=0
        Z = np.sum(q)
        Q = q/Z

        #Compute gradient
        if i < num_early: #Early exaggeration
            grad = 4*Z*(alpha*GL(P*Q) - GL(Q**2))@Y
        else:
            grad = 4*Z*GL((P-Q)*Q)@Y

        #Gradient descent
        Y -= h*grad

        #Percent complete
        if i % int(num_iter/10) == 0:
            print('%d%%'%(int(100*i/num_iter)))

    return Y,P

# %%
"""
Let's try the t-SNE algorithm on a subset of the MNIST digits.
"""

# %%
import graphlearning as gl
import numpy as np

#Load MNIST labels and results of k-nearest neighbor search
data, labels = gl.datasets.load('MNIST')

print(data.shape)

#Display some random MNIST images
gl.utils.image_grid(data[np.random.permutation(data.shape[0])],n_rows=20,n_cols=20)

# %%
"""
This implementation is for illustration and is in particular not sparse. So we can only run this on relatively small datasets. We run it on 300 images from MNIST below.
"""

# %%
import matplotlib.pyplot as plt
import graphlearning as gl
import numpy as np
from sklearn.decomposition import PCA
plt.ion()

#Load MNIST data and labels
data, labels = gl.datasets.load('mnist')

#Subsample MNIST
sz = 1000
X = data[labels <= 4]
T = labels[labels <= 4]
sub = np.random.choice(len(T),size=sz)
X = X[sub,:]
T = T[sub]

#Run PCA first
pca = PCA(n_components=50)
X = pca.fit_transform(X)

#Run t-SNE
Y,P = tsne(X,perplexity=30,h=sz,alpha=10,num_early=100,num_iter=500)

#Create scatterplot of embedding
plt.figure()
plt.scatter(Y[:,0],Y[:,1],c=T)

# %%
"""
The sklearn implementation of t-SNE uses a faster implementation that can handle larger data sets.
"""

# %%
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import graphlearning as gl
import numpy as np
from sklearn.decomposition import PCA

#Load MNIST data and labels
data, labels = gl.datasets.load('mnist')

#Subsample MNIST
sub = np.random.choice(len(labels),size=5000)
X = data[sub,:]
T = labels[sub]

#Run PCA first
pca = PCA(n_components=50)
X = pca.fit_transform(X)

#Run t-SNE
Y = TSNE(n_components=2, perplexity=30).fit_transform(X)

#Create scatterplot of embedding
plt.figure()
plt.scatter(Y[:,0],Y[:,1],c=T,s=0.5)

# %%
"""
Below we show the t-SNE embedding of a parabola in 10 dimensional space.
"""

# %%
n = 1000
X = np.zeros((n,10))
X[:,0] = np.linspace(-1,1,n)
X[:,1] = X[:,0]**2
X_tsne = TSNE(n_components=2, perplexity=20).fit_transform(X)
plt.figure()
plt.scatter(X_tsne[:,0],X_tsne[:,1],s=2,c=X[:,0])

# %%
"""
## Exercise

1. Run t-SNE on the two moons data set or the circles data set for different values of perplexity, to reproduce the results from the textbook.
2. Try the t-SNE algorithm on a k-nearest neighbor graph, instead of the perplexity graph construction. You will have to modify the provided t-SNE code to do this. Can you get similar results?

"""

# %%
"""
#SNE embedding

This notebook explores the t-SNE embedding method for visualizing high dimensional data.
"""

# %%
pip install graphlearning annoy

# %%
"""
Below is code for implementing t-SNE from scratch. This only works on small data sets, but is useful for understanding how the algorithm works and playing around with the code. We modified the code for `tsne` to allow the matrix $P$ to be specified by the user.
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

def tsne(X,perplexity=50,h=1,alpha=50,num_early=100,num_iter=1000,P=None):
    """t-SNE embedding

    Args:
        X: Data cloud
        perplexity: Perplexity (roughly how many neighbors to use)
        h: Time step
        alpha: Early exaggeration factor
        num_early: Number of early exaggeration steps
        num_iter: Total number of iterations
        P : Weight matrix to use in place of perplexity P

    Returns:
        Y: Embedded points
    """

    #Build graph using perplexity
    m = X.shape[0]
    if P is None:
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

def sne(X,perplexity=50,h=1,alpha=50,num_early=100,num_iter=1000):
    """SNE embedding

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
        q = np.exp(-np.sum((Y[I,:] - Y[I.T,:])**2,axis=2))
        q[range(m),range(m)]=0
        Z = np.sum(q)
        Q = q/Z

        #Compute gradient
        if i < num_early: #Early exaggeration
            grad = 4*(alpha*GL(P) - GL(Q))@Y
        else:
            grad = 4*GL(P-Q)@Y

        #Gradient descent
        Y -= h*grad

        #Percent complete
        if i % int(num_iter/10) == 0:
            print('%d%%'%(int(100*i/num_iter)))

    return Y,P

# %%
"""
This implementation is for illustration and is in particular not sparse. So we can only run this on relatively small datasets. We run it on 300 images from MNIST below. We run t-SNE using the perplexity graph construction as well as a k-nearest neighbor graph.
"""

# %%
import matplotlib.pyplot as plt
import graphlearning as gl
import numpy as np
from sklearn.decomposition import PCA

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
Y,_ = tsne(X,perplexity=30,h=sz,alpha=10,num_early=100,num_iter=500)

#Create scatterplot of embedding
plt.figure()
plt.scatter(Y[:,0],Y[:,1],c=T)
plt.title('t-SNE')

#Run SNE
Y,_ = sne(X,perplexity=30,h=20,alpha=10,num_early=100,num_iter=500)

#Create scatterplot of embedding
plt.figure()
plt.scatter(Y[:,0],Y[:,1],c=T)
plt.title('SNE')
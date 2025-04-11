# %%
pip install -q graphlearning

# %%
import numpy as np
from scipy import sparse

#Computes components (discriminating directions) for LDA
def lda_comp_shrinkage(X,labels,k=2,lam=1e-10):

    within_class,between_class = lda_cov(X,labels)
    within_class += lam*np.eye(X.shape[1])
    vals,V = sparse.linalg.eigsh(between_class,M=within_class,k=k,which='LM')
    V = V[:,::-1]
    vals = vals[::-1]
    return V

#LDA projection
def lda_shrinkage(X,labels,k=2,lam=1e-10):

    V = lda_comp_shrinkage(X,labels,k=k,lam=lam)
    return X@V

#Computes components (discriminating directions) for LDA
def lda_comp(X,labels,k=2):

    within_class,between_class = lda_cov(X,labels)
    vals,V = sparse.linalg.eigsh(within_class,k=100,which='LM')
    vals,P = sparse.linalg.eigsh(V.T@between_class@V,M=V.T@within_class@V,k=k,which='LM')
    V = V@P
    V = V[:,::-1]
    vals = vals[::-1]
    return V

#LDA projection
def lda(X,labels,k=2):

    V = lda_comp(X,labels,k=k)
    return X@V


#Computes principal components
def pca_comp(X,k=2):

    M = (X - np.mean(X,axis=0)).T@(X - np.mean(X,axis=0))

    #Use eigsh to get subset of eigenvectors
    vals, V = sparse.linalg.eigsh(M, k=k, which='LM')
    V = V[:,::-1]
    vals = vals[::-1]

    return vals,V

#PCA projection
def pca(X,k=2,whiten=False):

    vals,V = pca_comp(X,k=k)

    #Now project X onto the 2-D subspace spanned by
    #computing the 2D PCA coorindates of each point in X
    X_pca = X@V
    if whiten:
        print('whiten')
        S = np.diag(vals**(-1/2))
        X_pca = X_pca@S

    return X_pca


#LDA covariance matrices
def lda_cov(X,labels):
    num_classes = np.max(labels)+1
    within_class = np.zeros((X.shape[1],X.shape[1]))
    means = []
    counts = []
    for i in range(num_classes):
        Xs = X[labels==i,:].copy()
        counts += [np.sum(labels==i)]
        m = np.mean(Xs,axis=0)
        means += [m]
        within_class += (Xs-m).T@(Xs-m)

    means = np.array(means)
    counts = np.array(counts)
    Y = (means - np.mean(X,axis=0))*np.sqrt(counts[:,None])
    between_class = Y.T@Y

    return within_class, between_class

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
num = 5   #Number of digits to use
X = data[labels < num] #subset to 0s and 1s
L = labels[labels < num] #corresponding labels

#PCA
Y = pca(X)
plt.figure()
plt.title('PCA')
plt.scatter(Y[:,0],Y[:,1],c=L,s=10)

#LDA
Y = lda_shrinkage(X,L)
plt.figure()
plt.title('LDA via covariance shrinkage')
plt.scatter(Y[:,0],Y[:,1],c=L,s=10)

#LDA
Y = lda(X,L)
plt.figure()
plt.title('LDA via projection onto Sm')
plt.scatter(Y[:,0],Y[:,1],c=L,s=10)
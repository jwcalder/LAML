# %%
"""
# Linear Discriminant Analysis (LDA)

This notebook gives a brief introduction to Linear Discriminant Analysis (LDA). Let us first define some helper functions that will compute LDA and PCA for us.
"""

# %%
import numpy as np
from scipy import sparse

#Computes components (discriminating directions) for LDA
def lda_comp(X,labels,k=2,lam=1e-10):

    within_class,between_class = lda_cov(X,labels)
    within_class += lam*np.eye(X.shape[1])
    vals,V = sparse.linalg.eigsh(between_class,M=within_class,k=k,which='LM')
    V = V[:,::-1]
    vals = vals[::-1]
    return V

#LDA projection
def lda(X,labels,k=2,lam=1e-10):

    V = lda_comp(X,labels,k=k,lam=lam) 
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
"""
We first consider a toy data set in three dimensions
"""

# %%
import matplotlib.pyplot as plt
import numpy as np
plt.ion()

#Toy data
n = 1000
mean = [0,0,0]
cov = [[0.1,0,0], [0,1,0], [0,0,1]]  
X = np.random.multivariate_normal(mean, cov, n)
Y = np.random.multivariate_normal(mean, cov, n) + np.array([1,0,0])
X = np.vstack((X,Y))
L = np.hstack((np.zeros(n),np.ones(n))).astype(int)

#PCA
Y = pca(X)
plt.figure()
plt.title('PCA')
plt.scatter(Y[:,0],Y[:,1],c=L,s=10,vmin=0,vmax=2)

#LDA
Y = lda(X,L)
plt.figure()
plt.title('LDA')
plt.scatter(Y[:,0],Y[:,1],c=L,s=10,vmin=0,vmax=2)

# %%
"""
Let's now run this on MNIST and compare to PCA.
"""

# %%
#pip install -q graphlearning

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
Y = lda(X,L)
plt.figure()
plt.title('LDA')
plt.scatter(Y[:,0],Y[:,1],c=L,s=10)

# %%
"""
## Exercises
1. Try another data set in graphlearning, like 'fashionmnist'.
2. Use LDA as preprocessing for classification via support vector machines (SVM). Try MNIST, FashionMNIST or a data set from sklearn. Make sure to train LDA only on the training data.
3. Rewrite the LDA code so that instead of using covariance shrinkage, we project the data onto the top principal components so that the within covariance matrix is nonsingular.
4. Similar to 3, rewrite the LDA code to use the method in Exercise 4.2 in the LDA section of the course textbook.
"""




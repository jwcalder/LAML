# %%
"""
#Solutions to Laplace Learning Homework Exercise
"""

# %%
pip install -q graphlearning annoy

# %%
import graphlearning as gl
import numpy as np
from scipy import sparse

def spectral_laplace_learning(G,train_ind,train_labels,lam=0.1,k=20):

    n = G.num_nodes
    z = train_labels.copy()
    z[train_labels == 0] = -1
    y = np.zeros(n)
    y[train_ind] = z

    b = np.zeros(n)
    b[train_ind] = 1
    B = sparse.diags(b)
    M = B + lam*G.laplacian()

    vals,P = sparse.linalg.eigsh(M,k=k,which='SM')
    u = P@(np.diag(1/vals)@(P.T@(B@y)))

    return u > 0

# %%
"""
We first consider the two-moons data set. The red stars are the locations of the labeled nodes. See how well you can do with one label per moon.
"""

# %%
import numpy as np
import graphlearning as gl
import matplotlib.pyplot as plt
import sklearn.datasets as datasets

#Draw data randomly and build a k-nearest neighbor graph with k=10 neighbors
n = 500
X,labels = datasets.make_moons(n_samples=n,noise=0.1)
#X,labels = datasets.make_circles(n_samples=n,noise=0.1,factor=0.3)
W = gl.weightmatrix.knn(X,10)
G = gl.graph(W)

#Generate training data
train_ind = gl.trainsets.generate(labels, rate=3)
train_labels = labels[train_ind]

#Poisson Learning
pred_labels = spectral_laplace_learning(G,train_ind,train_labels)

#Compute accuracy
accuracy = gl.ssl.ssl_accuracy(pred_labels, labels, train_ind)
print("Accuracy: %.2f%%"%accuracy)

#Make plots
plt.figure()
plt.scatter(X[:,0],X[:,1], c=pred_labels)
plt.scatter(X[train_ind,0],X[train_ind,1], c='r', marker='*', s=100)
plt.show()

# %%
"""
We can now run an experiment classifying MNIST digits. We first load the dataset and display some images.
"""

# %%
import graphlearning as gl

#Load MNIST data
labels = gl.datasets.load('mnist', labels_only=True)
W = gl.weightmatrix.knn('mnist',10)
G = gl.graph(W)
labels = (labels <= 4).astype(int) #Binary 0--4 vs 5--9

#Generate training data
train_ind = gl.trainsets.generate(labels, rate=100)
train_labels = labels[train_ind]

#Poisson Learning
pred_labels = spectral_laplace_learning(G,train_ind,train_labels)

#Compute accuracy
accuracy = gl.ssl.ssl_accuracy(pred_labels, labels, train_ind)
print("Accuracy: %.2f%%"%accuracy)
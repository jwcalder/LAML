# %%
"""
#k-Means Clustering
This notebook gives some basic examples of k-means clustering, and an application to real data. While sklearn has a k-means clustering function, we will write our own, to make sure we understand the steps.
"""

# %%
import numpy as np
import matplotlib.pyplot as plt

def kmeans(X,k,visualize=False,T=200):
    """
    k-means Clustering

    Args:
        X: nxm array of data, each row is a datapoint
        k: Number of clusters
        visualize: Whether to plot internal iterations
        T: Max number of iterations

    Returns:
        Numpy array of labels obtained by k-means clustering
    """

    #Number of data points
    n = X.shape[0]

    #Randomly choose initial cluster means
    means = X[np.random.choice(n,size=k,replace=False),:]

    #Initialize arrays for distances and labels
    dist = np.zeros((k,n))
    labels = np.zeros((n,))

    #Main iteration for kmeans
    num_changed = 1
    i=0
    while i < T and num_changed > 0:

        #Update labels
        old_labels = labels.copy()
        for j in range(k):
            dist[j,:] = np.sum((X - means[j,:])**2,axis=1)
        labels = np.argmin(dist,axis=0)
        num_changed = np.sum(labels != old_labels)

        #Update means
        for j in range(k):
            means[j,:] = np.mean(X[labels==j,:],axis=0)

        #Iterate counter
        i+=1

        #Plot result (red points are labels)
        if visualize:
            print('Iteration %d'%i)
            plt.scatter(X[:,0],X[:,1], c=labels)
            plt.scatter(means[:,0],means[:,1], c='r')
            plt.pause(0.1)

    return labels

# %%
"""
Let's create some synthetic data and run k-means. Run it several times. Do you every see a poor clustering result?
"""

# %%
import sklearn.datasets as datasets

n = 500
X,L = datasets.make_blobs(n_samples=n, cluster_std=[1,1.5,0.5], random_state=60)
labels = kmeans(X,3,visualize=True)

# %%
"""
##Limits of k-means
k-means has trouble with datasets where clusters are not generally spherical in shape, especially when different clusters have vastly different aspect ratios. An example is given below.
"""

# %%
n = 500
separation = 0.8
X1 = np.random.randn(int(n/2),2)
L1 = np.zeros((n,))
X2 = np.random.randn(int(n/2),2)@np.array([[0.1,0],[0,10]]) + separation*np.array([3,0])
L2 = np.ones((n,))

X = np.vstack((X1,X2))
L = np.hstack((L1,L2))

labels = kmeans(X,2,visualize=True)

# %%
"""
Another interesting example is the famous two-moons.
"""

# %%
import sklearn.datasets as datasets

n=500
X,L = datasets.make_moons(n_samples=n,noise=0.1)

labels = kmeans(X,2,visualize=True)

# %%
"""
##Real data
We now consider using k-means to cluster MNIST digits. Let's install the [Graph Learning](https://github.com/jwcalder/GraphLearning) Python package.
"""

# %%
#pip install -q graphlearning

# %%
"""
Load MNIST data into memory.
"""

# %%
import graphlearning as gl

data, labels = gl.datasets.load('mnist')

# %%
"""
Let's plot some MNIST images.
"""

# %%
gl.utils.image_grid(data, n_rows=10, n_cols=10, title='Some MNIST Images', fontsize=26)

# %%
#Binary clustering problem witih 2 digits
class1 = 0
class2 = 1

#Subset data to two digits
I = labels == class1
J = labels == class2
X = data[I | J,:]
L = labels[I | J]

#Convert labels to 0/1
I = L == class1
L[I] = 0
L[~I] = 1

#kmeans clustering
cluster_labels = kmeans(X, 2)

#Check accuracy
acc1 = np.mean(cluster_labels == L)
acc2 = np.mean(cluster_labels != L)
print('Clustering accuracy = %.2f%%'%(100*max(acc1,acc2)))

#Show images from each cluster
gl.utils.image_grid(X[cluster_labels==0,:], n_rows=10, n_cols=10, title='Cluster 1', fontsize=26)
gl.utils.image_grid(X[cluster_labels==1,:], n_rows=10, n_cols=10, title='Cluster 2', fontsize=26)

# %%
"""
#Exercises
1. Play around with changing the two digits to cluster. Which two digits are most difficult to cluster?
2. Try k-means clustering with more than 2 classes. Try with 3, 4, or with the whole MNIST dataset. Computing accuracy is more challenging, since one has to account for all possible permutations of label values. Use the clustering_purity function below. Show image grids of each cluster.
3. Try applying $k$-means to another data set, say FashionMNIST in graphlearning or one of the real-world data sets in sklearn.dataset.
"""

# %%
import numpy as np
from sklearn import metrics

def purity_score(y_true, y_pred):
    """
    Computes purity of clustering.

    Args:
        y_true: True labels
        y_pred: Predicted labels

    Returns:
        Clustering purity
    """
    contingency_matrix = metrics.cluster.contingency_matrix(y_true, y_pred)
    return 100*np.sum(np.max(contingency_matrix, axis=0)) / np.sum(contingency_matrix)

# %%
for j in range(10):
    gl.utils.image_grid(data[cluster_labels==j,:], n_rows=10, n_cols=10, title='Cluster %d'%j, fontsize=26)

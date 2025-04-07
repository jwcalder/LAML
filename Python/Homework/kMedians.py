# %%
"""
#k-Medians Clustering
This notebook solves the k medians clustering homework problem.
"""

# %%
import numpy as np
import matplotlib.pyplot as plt

def kmedians(X,k,visualize=False,T=200):
    """
    k-medians Clustering

    Args:
        X: nxm array of data, each row is a datapoint
        k: Number of clusters
        visualize: Whether to plot internal iterations
        T: Max number of iterations

    Returns:
        Numpy array of labels obtained by k-medians clustering
    """

    #Number of data points
    n = X.shape[0]

    #Randomly choose initial cluster medians
    medians = X[np.random.choice(n,size=k,replace=False),:]

    #Initialize arrays for distances and labels
    dist = np.zeros((k,n))
    labels = np.zeros((n,))

    #Main iteration for kmedians
    num_changed = 1
    i=0
    while i < T and num_changed > 0:

        #Update labels
        old_labels = labels.copy()
        for j in range(k):
            dist[j,:] = np.sum(np.abs(X - medians[j,:]),axis=1)
        labels = np.argmin(dist,axis=0)
        num_changed = np.sum(labels != old_labels)

        #Update medians
        for j in range(k):
            medians[j,:] = np.median(X[labels==j,:],axis=0)

        #Iterate counter
        i+=1

        #Plot result (red points are labels)
        if visualize:
            print('Iteration %d'%i)
            plt.scatter(X[:,0],X[:,1], c=labels)
            plt.scatter(medians[:,0],medians[:,1], c='r')
            plt.pause(0.1)

    return labels

# %%
"""
Let's create some synthetic data and run k-medians. Run it several times. Do you every see a poor clustering result?
"""

# %%
import sklearn.datasets as datasets

n = 500
X,L = datasets.make_blobs(n_samples=n, cluster_std=[1,1.5,0.5], random_state=60)
labels = kmedians(X,3,visualize=True)

# %%
"""
##Real data
We now consider using k-means to cluster MNIST digits. Let's install the [Graph Learning](https://github.com/jwcalder/GraphLearning) Python package.
"""

# %%
pip install -q graphlearning

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
cluster_labels = kmedians(X, 2)

#Check accuracy
acc1 = np.mean(cluster_labels == L)
acc2 = np.mean(cluster_labels != L)
print('Clustering accuracy = %.2f%%'%(100*max(acc1,acc2)))

#Show images from each cluster
gl.utils.image_grid(X[cluster_labels==0,:], n_rows=10, n_cols=10, title='Cluster 1', fontsize=26)
gl.utils.image_grid(X[cluster_labels==1,:], n_rows=10, n_cols=10, title='Cluster 2', fontsize=26)
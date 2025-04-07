# %%
"""
# Binary Spectral Clustering

We give some examples of binary spectral clustering here on toy and real-world data sets. First we install [GraphLearning](https://github.com/jwcalder/GraphLearning) and [Annoy](https://github.com/spotify/annoy) (for nearest neighbor searches in graphlearning).
"""

# %%
#pip install -q annoy graphlearning

# %%
"""
We implement the k-means algorithm below, to compare against spectral clustering.
"""

# %%
import numpy as np
import matplotlib.pyplot as plt
plt.ion()

def kmeans(X,k,plot_clustering=False,T=200):
    """
    k-means Clustering

    Args:
        X: nxm array of data, each row is a datapoint
        k: Number of clusters
        plot_clustering: Whether to plot final clustering
        T: Max number of iterations

    Returns:
        Numpy array of labels obtained by binary k-means clustering
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
    if plot_clustering:
        plt.scatter(X[:,0],X[:,1], c=labels)
        plt.scatter(means[:,0],means[:,1], c='r')
        plt.title('K-means clustering')

    return labels, means

# %%
"""
Let's now test the algorithm on the two-moons and circles datasets, which k-means does poorly on. Uncomment the make_circles line to try that dataset instead of two moons.
"""

# %%
import sklearn.datasets as datasets
import graphlearning as gl

n=300 #Number of data points

#Two Moons or make circles
X,L = datasets.make_moons(n_samples=n,noise=0.1)
#X,L = datasets.make_circles(n_samples=n,noise=0.075,factor=0.5)
labels, means = kmeans(X,2)
plt.figure()
plt.scatter(X[:,0],X[:,1],c=labels)
plt.scatter(means[:,0],means[:,1],c='red',marker='*',s=200)
plt.title('k-means')

#Build graph and draw
W = gl.weightmatrix.epsilon_ball(X,0.25)
G = gl.graph(W)
G.draw(X=X,c=L,linewidth=0.1)
plt.title('Graph')

#Fiedler vector
v = G.fiedler_vector()
plt.figure()
plt.scatter(X[:,0],X[:,1],c=v)
plt.title('Fiedler Vector')

#Spectral clustering
plt.figure()
plt.scatter(X[:,0],X[:,1],c=v>0)
plt.title('Spectral Clustering')

# %%
"""
Let's now try specral clustering on a real-world graph. We'll use Zachary's karate club graph.
"""

# %%
G = gl.datasets.load_graph('karate')
L = G.labels
v = G.fiedler_vector()
print('True Labels        ',L)
print('Spectral Clustering',(v > 0).astype(int))

# %%
"""
Spectral clustering is only wrong in the classification of two members of the Karate club, members 3 and 9. To see if we can glean any more information from the Fiedler vector, we sort the club members by the value of the Fiedler vector.
"""

# %%
ind = np.argsort(v)
plt.figure()
plt.scatter(range(34),v[ind],c=L[ind])
plt.ylabel('Fiedler vector value')
plt.xlabel('Sorted member number')

# %%
"""
Inspecting the plot, we see that the Fiedler vector does perfectly separate the two groups if we threshold at a value slightly below zero. Inspecting the plot, and a bit of trial and error, yields $0.07$ as a good threshold. If you knew the desired sizes of the two groups, could you find an automatic way to select this threshold?
"""

# %%
print('True Labels        ',L)
print('Spectral Clustering',(v > 0.07).astype(int))

# %%
"""
## Exercise

Try binary spectral clustering on another real-world graph from the [graphlearning package](https://jwcalder.github.io/GraphLearning/datasets.html#graphlearning.datasets.load_graph) For example, `polbooks` is similarly small and easy to work with.
"""

# %%
"""
## Spectral clustering on MNIST

We now experiment with clustering MNIST digits using binary spectral clustering.
"""

# %%
import graphlearning as gl

#Load MNIST data and labels and plot some images
data, labels = gl.datasets.load('MNIST')
gl.utils.image_grid(data, n_rows=20, n_cols=20, title='Some MNIST Images', fontsize=26)

# %%
"""
The code below clusters a pair of MNIST digits. Try different pairs, which are hardest to separate?
"""

# %%
import numpy as np

#Subset data to two digits and convert labels to zeros/ones
digits = (3,8)
I,J = labels == digits[0], labels == digits[1]
X,L = data[I | J,:], (labels[I | J] == digits[1]).astype(int)

#Spectral Clustering (sparse 10-nearest neighbor graph)
W = gl.weightmatrix.knn(X,10)
G = gl.graph(W)
spectral_labels = (G.fiedler_vector() > 0).astype(int)
acc1 = np.mean(spectral_labels == L)
acc2 = np.mean(spectral_labels != L)
print('Spectral clustering accuracy = %.2f%%'%(100*max(acc1,acc2)))

#k-means clustering
kmeans_labels, means = kmeans(X,2)
acc1 = np.mean(kmeans_labels == L)
acc2 = np.mean(kmeans_labels != L)
print('K-means clustering accuracy = %.2f%%'%(100*max(acc1,acc2)))

#Show images from each cluster
gl.utils.image_grid(X[spectral_labels==0,:], n_rows=10, n_cols=10, title='Cluster 1', fontsize=26)
gl.utils.image_grid(X[spectral_labels==1,:], n_rows=10, n_cols=10, title='Cluster 2', fontsize=26)

# %%
"""
## Exercise

Implement the spectral approach to community detection via modularity maximization described in the book. The method is similar to spectral clustering, in that it uses the second eigenvector of a matrix similar to the graph Laplacian (the modularity matrix). How does modularity compare to spectral clustering on the examples in this notebook?
"""

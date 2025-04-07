# %%
"""
# PageRank

This notebook explores the PageRank algorithm. We first install [Graph Learning](https://github.com/jwcalder/GraphLearning), and then write a simple function to compute the PageRank vector with the PageRank (i.e., power) iteration.
"""

# %%
#pip install -q graphlearning

# %%
import graphlearning as gl

def PageRank(W,alpha=0.85,v=None,tol=1e-10):
    """
    PageRank algorithm

    Args:
        W: Weight matrix for graph
        v: Teleportation probability distribution (nx1 array, default is uniform)
        alpha: Probabilty of random walk step (1-alpha is teleportation probability)
        tol: Stopping condition tolerance (default: 1e-10)

    Returns:
        Numpy array containing PageRank vector
    """

    n = W.shape[0]

    D = gl.graph(W).degree_matrix(p=-1)
    P = W.T@D

    #Initialize u and normalize v
    u = np.ones(n)/n
    if v is None:
        v = np.ones(n)
    v = v/np.sum(v)

    #Power iteration until tolerance statisfied
    err = 1
    while err > tol:
        w = alpha*P@u + (1-alpha)*v
        err = np.max(np.absolute(w-u))
        u = w.copy()

    return u

# %%
"""
Let's create a synthetic dataset with 3 pages A,B,C. We'll assume that A and B both have links to C, and C has a self-link to itself. This is described in the weight matrix $W$ constructed below.
"""

# %%
import numpy as np

W = np.array([[0,0,1],[0,0,1],[0,0,1]])
W

# %%
"""
In this graph, node C is the most important node, since all other nodes link to it. Furthermore, there are no links to A or B. We would expect the PageRank of C to be large, and A and B to be small.

Below, we run the PageRank algorithm with uniform teleportation probability distribution. Are the results what you expect? Note in the plot that the GraphLearning package does not draw directed graphs.
"""

# %%
u = PageRank(W,alpha=0.85)
print(u)
gl.graph(W+W.T).draw(c=u)

# %%
"""
We now consider some real world data sets; Zachary's karate club and Krebs' political books graphs.
"""

# %%
import matplotlib.pyplot as plt
plt.ion()

G = gl.datasets.load_graph('karate')
u = PageRank(G.weight_matrix)
X = G.draw(markersize=100,linewidth=0.5,c=u)
plt.title('Karate')

G = gl.datasets.load_graph('polbooks')
u = PageRank(G.weight_matrix)
X = G.draw(markersize=100,linewidth=0.5,c=u)
plt.title('Political Books')

# %%
"""
Finally, let's look at personalized PageRank on each of these graphs.
"""

# %%
import numpy as np

graphs = ['karate','polbooks']

for graph in graphs:
    #Load graph, and choose a random seed node
    G = gl.datasets.load_graph(graph)
    np.random.seed(12)
    v = np.zeros(G.num_nodes)
    v[np.random.randint(0,len(v))] = 1

    #Personalized PageRank
    u = PageRank(G.weight_matrix,v=v)
    u = u / np.max(u)
    G.draw(markersize=100,linewidth=0.5,c=u**(1/5)) #Taking a fractional power to better visualize
    plt.title(graph + ' personalized PageRank')

# %%
"""
## Exercise

Try playing around with different values for the teleportation paramter $\alpha$. 
"""

# %%
"""
##Personalized PageRank on MNIST
We now consider personalized PageRank for finding similar images in the MNIST dataset. Below, we load the MNIST data set and construct a sparse k-nearest neighbor graph.
"""

# %%
import graphlearning as gl
import numpy as np

#Load MNIST labels and results of k-nearest neighbor search
data, labels = gl.datasets.load('mnist')

#Build 10-NN graph
W = gl.weightmatrix.knn('mnist', 10)

# %%
"""
Let's now draw a random digit. Below we will use personalized PageRank to find similar MNIST digits. You can run the code cell below many times to get different random digits.
"""

# %%
import matplotlib.pyplot as plt

#Draw a random image
n = W.shape[0]
rand_ind = np.random.randint(0,high=n)
print(labels[rand_ind])

#Plot the digit
plt.figure()
plt.imshow(np.reshape(data[rand_ind,:],(28,28)), cmap='gray')
plt.axis('off')

# %%
"""
Let's now run personalized PageRank to find similar digits, which we display with image_grid.
"""

# %%
#Localized teleportation distribution
n = W.shape[0]
v = np.zeros(n)
v[rand_ind]=1 #Choose a random image

#Run pagerank
u = PageRank(W,v=v,alpha=0.55)

#Display highest ranked images
ind = np.argsort(-u) #indices to sort u
gl.utils.image_grid(data[ind,:])
print(labels[ind][:100])

# %%
"""
#Exercise

Run personalized PageRank on MNIST using a teleportation distribution focused on one of the digit classes. That is, set $v=1$ on a particular digit (say, $0$), and set $v=0$ elsewhere. What do you expect to see for the highest ranked digits? Compare the highest ranked digits to some random digits from the class you chose. Also plot the lowest ranked digits from the class you chose. They should look like the least typical (worst) examples from that digit.
"""

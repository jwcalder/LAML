# %%
"""
# Introduction to Graphs

This notebook gives an introduction to loading and displaying graphs using the [Graph Learning](https://github.com/jwcalder/GraphLearning) Python package, which we install below.
"""

# %%
#pip install -q graphlearning

# %%
"""
The GraphLearning [documentation](https://jwcalder.github.io/GraphLearning/datasets.html#graphlearning.datasets.load_graph) shows which graphs can be loaded. Here, we load the Zachary's karate club graph, described in class, and visualize the graph and its adjacency matrix.
"""

# %%
import graphlearning as gl
import matplotlib.pyplot as plt
import numpy as np
import sys
np.set_printoptions(threshold=sys.maxsize,linewidth=110) #Disables truncation when printing arrays
plt.ion()

G = gl.datasets.load_graph('karate')

A = G.weight_matrix #Adjacency matrix or weight matrix if a weighted graph
print(A) #Not too useful since A is a sparse matrix
print(A.todense()) #This prints the full matrix

plt.figure(figsize=(10,10))
plt.imshow(A.todense(),cmap='gray') #Displaying graphs as images can be useful

# %%
"""
Some graphs have labels and feature vectors as well. Here, the Karate graph has binary lables indicating the group membership after the club split into two.
"""

# %%
print(G.num_nodes)
print(G.labels)
print(G.features)

# %%
"""
We can us the draw function to visualize the karate club graph.
"""

# %%
G.draw(markersize=100,c=G.labels,linewidth=0.5)
plt.title('Karate club graph')

# %%
"""
Another graph we will use often is PubMed. Pubmed is too large to easily visualize
"""

# %%
G = gl.datasets.load_graph('pubmed')

print(G.num_nodes)
print(G.features.shape)
print(G.labels)
print('Sparsity', 100*G.weight_matrix.count_nonzero()/G.num_nodes**2, '%')
print(G.weight_matrix)

# %%
"""
Another way graphs appear is as similarity graph constructed over data sets. As a simple example, we show the two moons and circles data sets below. This uses the GraphLearning [WeightMatrix](https://jwcalder.github.io/GraphLearning/weightmatrix.html) modele to construct graphs, which can build epsilon ball graphs or k-nearest neighbor graphs.
"""

# %%
from sklearn import datasets

n=300

#Two moons
X,L = datasets.make_moons(n_samples=n,noise=0.1)
W = gl.weightmatrix.epsilon_ball(X,0.25)
G = gl.graph(W)
G.draw(X=X,linewidth=0.5,c=L)
plt.title('Epsilon-ball graph')

W = gl.weightmatrix.knn(X,7)
G = gl.graph(W)
G.draw(X=X,linewidth=0.5,c=L)
plt.title('k-nn graph')

#Circles
X,L = datasets.make_circles(n_samples=n,noise=0.075,factor=0.5)
W = gl.weightmatrix.epsilon_ball(X,0.25)
G = gl.graph(W)
G.draw(X=X,linewidth=0.5,c=L)
plt.title('Epsilon-ball graph')

W = gl.weightmatrix.knn(X,7)
G = gl.graph(W)
G.draw(X=X,linewidth=0.5,c=L)
plt.title('k-nn graph')

# %%
"""
We can also view any triangulated surface as a graph.
"""

# %%
import matplotlib.tri as mtri

ax = plt.figure(figsize=(10,10)).add_subplot(projection='3d')
u = np.linspace(0, 2.0 * np.pi, endpoint=True, num=50)
v = np.linspace(-0.5, 0.5, endpoint=True, num=10)
u, v = np.meshgrid(u, v)
u, v = u.flatten(), v.flatten()
x = (1 + 0.5 * v * np.cos(u / 2.0)) * np.cos(u)
y = (1 + 0.5 * v * np.cos(u / 2.0)) * np.sin(u)
z = 0.5 * v * np.sin(u / 2.0)

# Triangulate parameter space to determine the triangles
tri = mtri.Triangulation(u, v)
edges = tri.edges

ax.scatter(x,y,z,c='black',s=5)
for t in range(edges.shape[0]):
    i,j = edges[t,0],edges[t,1]
    ax.plot([x[i],x[j]],[y[i],y[j]],[z[i],z[j]],c='black',linewidth=0.25)

ax.set_axis_off()
ax.set_zlim(-1, 1)

# %%
"""
We can also view any image as a graph, by simply connecting each pixel to its nearest neighbors. We will see later in the course that there are alternative, and sometimes better, ways to view images as graphs.
"""

# %%
data,labels = gl.datasets.load('mnist')

n = data.shape[0]
p = np.random.permutation(n)
for i in p[:10]:
    img = np.reshape(data[i,:],(28,28))
    W,X = gl.weightmatrix.grid_graph(img,return_xy=True)
    X[:,1] = -X[:,1]
    G = gl.graph(W)
    G.draw(X=X,linewidth=0.5,c=img.flatten())
    plt.title(str(labels[i]))

# %%
"""
Finally, we can construct similarity graphs between pairs of images, like MNIST digits. The GraphLearning package supports this for several different image classification data sets. See the [documentation](https://jwcalder.github.io/GraphLearning/weightmatrix.html#graphlearning.weightmatrix.knn).
"""

# %%
data,labels = gl.datasets.load('mnist')
gl.utils.image_grid(data)

# %%
"""
Below, we build a k-nn graph over the 70000 images in the MNIST data set, with k=10.
"""

# %%
W = gl.weightmatrix.knn('mnist',10)
n = W.shape[0]
print(W.shape)
print('Sparsity',W.count_nonzero()/n**2,'%')
print(W)

# %%
"""
We can display, for example, an image and its nearest neighbors, using the graph structure.
"""

# %%
nn_ind = W[0,:].nonzero()[1]
gl.utils.image_grid(data[nn_ind,:],n_cols=len(nn_ind),n_rows=1)

# %%
"""
Below, we do the same for the Cifar-10 dataset.
"""

# %%
data,labels = gl.datasets.load('cifar10')
gl.utils.color_image_grid(data)

# %%
W = gl.weightmatrix.knn('cifar10',10,metric='simclr')
n = W.shape[0]
print(W.shape)
print('Sparsity',W.count_nonzero()/n**2,'%')
print(W)

# %%
"""
Let's now display some nearest neighbors of Cifar-10 images.
"""

# %%
nn_ind = W[0,:].nonzero()[1]
gl.utils.color_image_grid(data[nn_ind,:],n_cols=len(nn_ind),n_rows=1)

# %%
"""
# Shortest path distances on graphs

This notebook gives some examples of working with and computing distances and shortest paths on toy and real-world data sets. First we install [GraphLearning](https://github.com/jwcalder/GraphLearning).
"""

# %%
#pip install -q graphlearning

# %%
"""
## Computing shortest paths with the GraphLearning package

We first show how to compute the shortest path distance to a point and an optimal path on the toy two moons and circles data sets.
"""

# %%
import graphlearning as gl
import numpy as np
import sklearn.datasets as datasets
import matplotlib.pyplot as plt
plt.ion()

n=300
np.random.seed(1)
X,L = datasets.make_moons(n_samples=n,noise=0.1)
#X,L = datasets.make_circles(n_samples=n,noise=0.075,factor=0.5)
W = gl.weightmatrix.knn(X,7,kernel='uniform')
G = gl.graph(W)

#Compute distance beteween two vertices, here 0,1
#As well as shortest path and distance vector t
i,j = 0,1
d,path,t = G.distance(i,j,return_path=True,return_distance_vector=True)

#Plot distance function
plt.figure()
plt.scatter(X[:,0],X[:,1],c=t)
plt.scatter(X[i,0],X[i,1],c='r',marker='*',s=200,zorder=100)
plt.title('Distance vector')

#Draw graph and shortest path between two vertices (here, 0,1)
G.draw(X=X,linewidth=0.5,c='gray',linecolor='gray')
plt.plot(X[path,0],X[path,1],'ro-',markersize=6,linewidth=1.5)
plt.title('Shortest path')

# %%
"""
## Dynamic programming iteration

We now get into the details of the dynamic programming. We'll show how to compute the distance vector and shortest paths on the karate graph.
"""

# %%
import numpy as np
import graphlearning as gl

#Load graph
G = gl.datasets.load_graph('karate')
m = G.num_nodes

#Choose two vertices to find the shortest path between
pts = (16,15)

#Compute shortest path and distance vector with graphlearning
d,path,t = G.distance(pts[0],pts[1],return_path=True,return_distance_vector=True)
ind = np.argsort(t) #So we can display u in order of increasing distance

#Initialize u to infinity away from the seed point pts[0]
u = np.ones(m)*np.inf
u[pts[0]] = 0
print('Initial u:\n',u[ind])

#Temporary array to hold computations
v = np.zeros(m)

#Dynamic programming iterations loop
err = 1
i=0
while err > 0:

    #Dynamic programming iteration
    for j in range(m):
        nn, w = G.neighbors(j, return_weights=True)
        v[j] = np.min(u[nn] + w**-1)
    v[pts[0]] = 0

    #Start measuring the error u_k-u_{k+1} once infinities are gone
    if np.max(u) < np.inf:
        err = np.max(np.absolute(u-v))

    #Copy v back to u (numpy arrays are pointers, so be careful)
    u = v.copy()

    #Print current u_k
    print('\nIter: %d\n'%i,u[ind])
    i+=1

# %%
"""
We'll now use dynamic programming to find the optimal path.
"""

# %%
p = pts[1]
path = [p]
while p != pts[0]:
    nn, w = G.neighbors(p, return_weights=True)
    j = np.argmin(u[nn] + w**-1)
    p = nn[j]
    path += [p]
path = np.array(path)
print('Optimal path:',path,'\n')

#Draw graph
Y = G.draw(markersize=50,linewidth=0.5)
plt.plot(Y[path,0],Y[path,1],'ro-',markersize=7,linewidth=2)

# %%
"""
## Exercise: PubMed

1. Run the dynamic programming iterations on PubMed to find the distance function to a particular node. How many iterations does it take to converge?
2. Modify the code to use the Gauss-Seidel method. Does Gauss-Seidel converge faster? If so, by how much?
3. In the Gauss-Seidel method, iterate over the nodes of the graph in order of the value of the distance vector, starting from smallest (the seed node) to largest. You can compute the distance vector using GraphLearning for this, and use `numpy.argsort`. Can you get Gauss-Seidel to converge in one iteration? This is the essential idea behind Dijkstra's algorithm.
"""


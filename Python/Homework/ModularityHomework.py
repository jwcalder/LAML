# %%
pip install graphlearning

# %%
import graphlearning as gl
import numpy as np
import matplotlib.pyplot as plt

G = gl.datasets.load_graph('karate')
#G = gl.datasets.load_graph('polbooks')
L = G.labels
d = G.degree_vector()[:,None]
W = G.weight_matrix.toarray()
print(type(W))
M = W - d@d.T/np.sum(d)

#Power iteration
#Shift M to make positive semidefinite
c = np.sqrt(np.sum(W**2)) + np.sum(d**2)/np.sum(d)
A = M + c*np.eye(G.num_nodes)
v = np.random.randn(G.num_nodes)
for i in range(100):
    w = A@v
    v = w/np.linalg.norm(w)

if v[0] < 0:
    v = -v
print('True Labels        ',L)
print('Spectral Clustering',(v < 0).astype(int))

ind = np.argsort(v)
plt.figure()
plt.scatter(range(G.num_nodes),v[ind],c=L[ind])
plt.ylabel('Modularity vector value')
plt.xlabel('Sorted member number')
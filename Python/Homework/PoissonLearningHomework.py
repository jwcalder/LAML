# %%
"""
#Solutions to Poisson Learning Homework Exercise


"""

# %%
pip install -q graphlearning annoy

# %%
import graphlearning as gl
import numpy as np

def poisson_learning(G,train_ind,train_labels,num_iter=1000):

    n = G.num_nodes
    z = train_labels.copy()
    z[train_labels == 0] = -1
    y = np.zeros(n)
    y[train_ind] = z

    n = G.num_nodes
    u = np.zeros(n)
    D = G.degree_matrix(p=-1)
    P = D@G.weight_matrix
    f = D@y
    for i in range(num_iter):
        u = P@u + f

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
#X,labels = datasets.make_moons(n_samples=n,noise=0.1)
X,labels = datasets.make_circles(n_samples=n,noise=0.1,factor=0.3)
W = gl.weightmatrix.knn(X,10)
G = gl.graph(W)

#Generate training data
train_ind = gl.trainsets.generate(labels, rate=3)
train_labels = labels[train_ind]

#Poisson Learning
pred_labels = poisson_learning(G,train_ind,train_labels)

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
pred_labels = poisson_learning(G,train_ind,train_labels)

#Compute accuracy
accuracy = gl.ssl.ssl_accuracy(pred_labels, labels, train_ind)
print("Accuracy: %.2f%%"%accuracy)

# %%
"""
Below is an example of Poisson learning using the GraphLearning package on the full MNIST data set with 10 classes, compared to Laplacian regularized learning.
"""

# %%
import graphlearning as gl

labels = gl.datasets.load('mnist', labels_only=True)
W = gl.weightmatrix.knn('mnist', 10)

num_train_per_class = 10
train_ind = gl.trainsets.generate(labels, rate=num_train_per_class)
train_labels = labels[train_ind]

models = [gl.ssl.laplace(W), gl.ssl.poisson(W)]

for model in models:
    pred_labels = model.fit_predict(train_ind,train_labels)
    accuracy = gl.ssl.ssl_accuracy(pred_labels, labels, train_ind)
    print(model.name + ': %.2f%%'%accuracy)
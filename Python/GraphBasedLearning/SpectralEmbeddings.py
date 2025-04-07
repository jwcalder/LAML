# %%
"""
# Spectral embeddings and spectral clustering

Here, we explore spectral embeddings and spectral clustering on real and toy data sets.
"""

# %%
#pip install graphlearning annoy

# %%
"""
Below we show a spectral embedding of some of the MNIST digits.
"""

# %%
import graphlearning as gl
import numpy as np

#Load MNIST labels and results of k-nearest neighbor search
data, labels = gl.datasets.load('MNIST')

print(data.shape)

#Display some random MNIST images
gl.utils.image_grid(data[np.random.permutation(data.shape[0])],n_rows=20,n_cols=20)

# %%
import matplotlib.pyplot as plt
plt.ion()

#Subset data and labels
X = data[labels <= 2]
Y = labels[labels <= 2]

#Build Graph (sparse k-nearest neighbor graph)
W = gl.weightmatrix.knn(X,10)
G = gl.graph(W)

#Compute eigenvectors of graph Laplacian
vals, vecs = G.eigen_decomp(normalization='normalized', k=20)

#Plot spectral embedding colored by label
#2D plot
plt.figure()
plt.scatter(vecs[:,1],vecs[:,2],c=Y,s=1)
#3D plot
plt.figure()
ax = plt.axes(projection="3d")
ax.scatter3D(vecs[:,1],vecs[:,2],vecs[:,3],c=Y,s=1)

# %%
"""
Spectral clustering simply applies the k-means clustering algorithm to the spectrally embedded points. We use an added step of normalizing the embedded points.
"""

# %%
from scipy import sparse
import sklearn.cluster as cluster

num_clusters = 4 # We choose 4 since there are two clusters of ones.
X_emb = vecs[:,:num_clusters]
norms = np.linalg.norm(X_emb,axis=1)
X_emb = X_emb / norms[:,None] #Normalize rows
kmeans = cluster.KMeans(n_clusters=num_clusters).fit(X_emb)
cluster_labels = kmeans.labels_

for i in range(num_clusters):
    gl.utils.image_grid(X[cluster_labels==i,:],n_rows=20,n_cols=20)

# %%
"""
## Image segmentation

We show here how to use spectral clustering for image segmentation.
"""

# %%
import numpy as np
import graphlearning as gl
import matplotlib.pyplot as plt
from skimage.transform import resize
from sklearn.cluster import KMeans

orig_img = gl.datasets.load_image('cow')
plt.imshow(orig_img)

# %%
"""
We'll subsample the image to speed up processing.
"""

# %%
subsample = 2
img = orig_img[::subsample,::subsample,:].copy() #Subsample to speed up processing

# %%
"""
We now construct a weight matrix taking into account pixel values and pixel locations.
"""

# %%
#Coordinates for pixels in image
m,n,c = img.shape
x,y = np.mgrid[:m,:n]
x,y = x.flatten(),y.flatten()
X = np.vstack((x,y)).T

#Features of image (pixels colors)
F = np.reshape(img,(m*n,3))

#Weight matrix
W = gl.weightmatrix.epsilon_ball(X,10,features=F,epsilon_f=0.15)

# %%
"""
We now compare spectral clustering and k-means clustering.
"""

# %%
num_clusters = 4 #2 cows, grass, sky

colors = [[0,0,0],[1,0,0],[0,1,0],[0,0,1],[1,1,1]]
def color_seg(pred):
    cimg = img.copy()
    for i in range(num_clusters):
        cimg[pred==i,:]=colors[i]
    return cimg

#Kmeans
kmeans = KMeans(n_clusters=num_clusters).fit(F)
kmeans_pred_labels = np.reshape(kmeans.labels_,(m,n))
cimg = color_seg(kmeans_pred_labels)
plt.figure()
plt.imshow(resize(cimg,(subsample*m,subsample*n),order=0))

#Spectral clustering
model = gl.clustering.spectral(W, num_clusters=num_clusters, method='ShiMalik')
pred_labels = np.reshape(model.fit_predict(),(m,n))
cimg = color_seg(pred_labels)
plt.figure()
plt.imshow(resize(cimg,(subsample*m,subsample*n),order=0))

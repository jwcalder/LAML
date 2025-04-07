# %%
"""
#PCA-based image compression

We consider here an application of PCA to image compression. We first load an image and show some of its trucated SVD's (for row-wise compression of the image matrix).
"""

# %%
#pip install -q graphlearning

# %%
"""
Let's load an image and display it.
"""

# %%
import graphlearning as gl
import matplotlib.pyplot as plt
plt.ion()

img = gl.datasets.load_image('chairtoy')
plt.figure()
plt.imshow(img)

#Check data range and shape
print('Pixel intensity range: (%d,%d)'%(img.min(),img.max()))
print(img.shape)

# %%
"""
Now let's display the truncated SVD. Since the $m\times n$ data matrix $X$ below has far more columns than rows, i.e., $m \ll n$, it is expensive to form the $n\times n$ matrix $X^TX$ and find its top eigenvalues. In this case, an SVD is faster, using scipy.sparse.linalg.svds, or we can compute the eigenvectors of $XX^T$, which is a smaller $m\times m$ matrix. In fact the latter is fastest, as shown in the code below, which also compares against full SVD and Eig operations, though these are not always tractable with very large and/or high dimensional data sets.

As an aside, if you inspect the code of svds in scipy, it uses the eigensolver eigsh on $X^TX$, but it does it in such a way that the matrix $X^TX$ is never explicitly computed! The method is similar to the power iteration, using that you can compute $X^TX\mathbf{x} = X^T(X\mathbf{x})$ without ever forming $X^TX$. Thus, if $X$ is $m\times n$, the SVD solver computes $X^TX\mathbf{x}$ in $O(2mn)$ operations, compared to $O(n^2)$ operations by forming $X^TX$ directly. On the other hand, working with $XX^T$ requires only $O(m^2)$ operations to compute $XX^T\mathbf{x}$. As another note, when using svd it is faster to add the flag `full_matrices = False` which truncates to the rank of $X$ automatically.
"""

# %%
from scipy import sparse
import numpy as np
from scipy import linalg
from sklearn.decomposition import PCA
import time

X = np.hstack((img[:,:,0],img[:,:,1],img[:,:,2]))

#How many singular vectors to compute
num_eig = 500

#SVD (order of singular values not guaranteed so we have to sort)
t0 = time.time()
P,S,QT = sparse.linalg.svds(X,k=num_eig)
print('SVD of X time: %.2f seconds'%(time.time()-t0))
ind = np.argsort(-S)
Q = QT[ind,:].T #Scipy returns the SVD transposed

#Compare execution time to eigsh for X^TX
t0 = time.time()
Vals, Q = sparse.linalg.eigsh(X.T@X,k=num_eig,which='LM')
print('X^TX Eigs time: %.2f seconds'%(time.time()-t0))
Q = Q[:,::-1] #Eigenvalues are returned in opposite order

#Compare execution time to eigsh for XX^T
t0 = time.time()
Vals, P = sparse.linalg.eigsh(X@X.T,k=num_eig,which='LM')
Q = X.T@P@np.diag(1/np.sqrt(Vals)) #Convert from left to right singular vectors
print('XX^T Eigs time: %.2f seconds'%(time.time()-t0))
Q = Q[:,::-1] #Eigenvalues are returned in opposite order

#Time for full eig or svd
t0 = time.time()
np.linalg.svd(X,full_matrices=False)
print('Numpy X full svd time: %.2f seconds'%(time.time()-t0))

#Time for full eig or svd
t0 = time.time()
linalg.svd(X,full_matrices=False)
print('Scipy X full svd time: %.2f seconds'%(time.time()-t0))

t0 = time.time()
np.linalg.eigh(X.T@X)
print('X^TX full Eigs time: %.2f seconds'%(time.time()-t0))

t0 = time.time()
np.linalg.eigh(X@X.T)
print('XX^T full Eigs time: %.2f seconds'%(time.time()-t0))

#Compare to sklearn
t0 = time.time()
pca = PCA(n_components=num_eig)
pca.fit(X)
print('sklearn pca time: %.2f seconds'%(time.time()-t0))


for k in [1,5,25,50,100,200]:
    if k <= num_eig:
        Xk = np.clip(X@Q[:,:k]@Q[:,:k].T,0,1)
        imgk = np.stack((Xk[:,:512],Xk[:,512:1024],Xk[:,1024:]),axis=2)
        plt.figure()
        plt.imshow(imgk)
        plt.title('%d Singular Vectors'%k)


# %%
"""
For block-based compression, we'll convet the image to 8x8 patches. The image is 512x512 so this gives 4096 patches, each with 8x8x3=192 numbers (RGB for each pixel).
"""

# %%
m = 8
X = gl.utils.image_to_patches(img,patch_size=(m,m))
print(X.shape)
grid = gl.utils.color_image_grid(X,n_rows=5,n_cols=10)

# %%
"""
Now let's do an SVD (or PCA) on the blocks. We show to top 50 singular vectors, which start out as slowly varying low frequencies, with the later singular vectors capturing fine scale details and texture in the image blocks. 

Here, the $m\times n$ matrix $X$ has far more rows than columns, so $m \gg n$. In this case, it is more efficient to compute $X^TX$, which is a smaller $n\times n$ matrix and compute its eigenvectors, instead of performing an SVD. We perform a full eigendecomposition, since we want all eigenvectors for this simulation. 
"""

# %%
from scipy import sparse
import time

t0 = time.time()
Vals, Q = np.linalg.eigh(X.T@X)
print('Eigh time: %.2f seconds'%(time.time()-t0))
Q_all = Q[:,::-1] #Eigenvalues are returned in opposite order

#SVD for comparison
t0 = time.time()
np.linalg.svd(X,full_matrices=False)
print('Numpy SVD time: %.2f seconds'%(time.time()-t0))

#SVD for comparison
t0 = time.time()
linalg.svd(X,full_matrices=False)
print('Scipy SVD time: %.2f seconds'%(time.time()-t0))

#Compare to sklearn
t0 = time.time()
pca = PCA()
pca.fit(X)
print('sklearn pca time: %.2f seconds'%(time.time()-t0))

P = Q_all.T.copy()
P = P - P.min()
P = P/P.max()
gl.utils.color_image_grid(P,n_rows=5,n_cols=10)

# %%
"""
To compress the image, we project the image blocks onto the top k singular vectors and the reconstruct the image from its blocks.
"""

# %%
for num_comps in [1,5,10,20,40,80]:

    #Get top singular vectors
    Q = Q_all[:,:num_comps]

    #Compress and decompress the image
    comp_X = X@Q
    decomp_X = comp_X@Q.T

    #Compute compression ration
    comp_size = Q.shape[0]*Q.shape[1] + comp_X.shape[0]*comp_X.shape[1]
    comp_ratio = X.shape[0]*X.shape[1]/comp_size

    #Recontruct image from patches for visualization
    img_comp = gl.utils.patches_to_image(decomp_X, (img.shape[0],img.shape[1]), patch_size=(m,m))
    img_comp = np.clip(img_comp,0,1)

    #Print PSNR
    MSE = np.sum((X-decomp_X)**2)/X.shape[0]/X.shape[1]
    PSNR = 10*np.log10(np.max(X)**2/MSE)
    print('Compression ratio: %.1f:1, PSNR: %.1f'%(comp_ratio,PSNR))

    #Plot compressed and difference image
    img_diff = np.clip(img_comp - img + 0.5,0,1)
    plt.figure()
    plt.imshow(np.hstack((img_comp,img_diff)))
    plt.title('Compression Raio: %.1f:1'%comp_ratio)


# %%
"""
## Exercises
1. Play around with different block sizes. What is the best for compression?
2. Compare the PSNR for the original row-based compression with block-based compression. How does the PSNR compare at the same label rates. 
3. Generate a plot of the singular values for the image and the blocks. Which one decays faster? 
4. Project the blocks into two dimensions to visualize the block space. 
5. [Challenging] Write a compression algorithm that chooses the best singular vectors to use for each block, instead of the top $k$. To do this, choose a threshold $\mu>0$, project the image blocks onto all of the singular vectors, and then discard (i.e., set to zero) any coefficient (i.e., PCA coordinate) that is smaller than $\mu$. Reconstruct the image from the truncated blocks, and compute the compression ratio assuming you do not have to store the zeros (the coefficients that were threshold ed to zero), and assume you don't need to store the singular vectors (the setting is that you learn good singular vectors, and then share them between the encoder and decoder, so only the coefficients must be transmitted/stored). How does this compare with the block-based method in this notebook? 
#"""



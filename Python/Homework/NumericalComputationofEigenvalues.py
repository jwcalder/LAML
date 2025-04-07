# %%
"""
# Numerical Computation of Eigenvalues

This notebook covers the power method and orthogonal iteration for the numerical computation of eigenvalues, with application exercises to computing principal directions in PCA on image data sets.
"""

# %%
"""
## Power Method

The power method refers to the iteration
$$\mathbf{x}_{k+1} = \frac{A\mathbf{x}_k}{\|A\mathbf{x}_k \|},$$
which, under certain conditions on the matrix $A$ and initial vector $\mathbf{x}_0$ converges to the dominant eigenvector of $A$.

"""

# %%
"""
### Exercise (small matrix)

Use the power method to compute the top eigenvector of the matrix
$$A = \begin{pmatrix}
2&-1&0&-1\\
-1&2&-1&0\\
0&-1&2&-1\\
-1&0&-1&2
\end{pmatrix}.$$
Check your result against the output of `numpy.linalg.eig`. How many iterations do you need for an accurate result?
"""

# %%
import numpy as np

A = np.array([[2,-1,0,-1],[-1,2,-1,0],[0,-1,2,-1],[-1,0,-1,2]])
print("A=",A)

vals,vecs = np.linalg.eig(A)
print('Eigenvalues=',vals)
print('Eigenvectors=',vecs) #Columns are eigenvalues

#Power method code
x = np.random.randn(4,1) #Random initial vector

#Insert your code here



print('Power Method')
print('Eigenvalue: ',...)
print('Eigenvector: ',...)


# %%
"""
### Exercise (random matrix)

Create a random $n\times n$ positive definite symmetric matrix with $n=100$ and compute the top eigenvector with the power method. Compare to `scipy.sparse.linalg.eigsh`. How many iterations do you need for an accurate result?
"""

# %%
from scipy import sparse

n=100
B = np.random.rand(n,n)
A = B.T@B #Creates a random positive definite Gram matrix
vals,vecs = sparse.linalg.eigsh(A,k=1,which='LM')
print('Eigenvalue=',vals[0]) #Eigenvector too long to print
print('Eigenvector=',vecs.flatten())

#Power method code
x = np.random.randn(n,1) #Random initial vector

#Insert your code here


print('Power Method')
print('Eigenvalue: ',...)
print('Eigenvector: ',...)


# %%
"""
### Exercise (Eigenface)

Use the power method to compute the top principal direction for the Olivetti face dataset. We first install the [GraphLearning](https://github.com/jwcalder/GraphLearning), and then load and display the Olivetti face images.
"""

# %%
#pip install -q graphlearning

# %%
import graphlearning as gl
from sklearn import datasets

ds = datasets.fetch_olivetti_faces()
data = ds['data']
labels = ds['target']

gl.utils.image_grid(data, n_rows=10, n_cols=15, title='Some faces', fontsize=26)
print(data.shape)
print(labels.shape)
print(np.unique(labels))

# %%
"""
Below we compute the covariance matrix and the top principal direction.
"""

# %%
import matplotlib.pyplot as plt
from scipy import sparse

#Centered covariance matrix
mean_face = np.mean(data,axis=0)
X = data - mean_face
M = X.T@X

#Use eigsh to get subset of eigenvectors
#('LM'=largest magnitude, k=1 eigenvectors)
vals, vecs = sparse.linalg.eigsh(M, k=1, which='LM')

#Display the top principal component images
n = len(mean_face)
m = int(np.sqrt(len(mean_face)))
plt.figure()
plt.imshow(np.reshape(mean_face,(m,m)),cmap='gray')
plt.title('Mean Face')

plt.figure()
plt.imshow(np.reshape(vecs[:,0],(m,m)),cmap='gray')
plt.title('Top principal direction')

# %%
"""
Compute the top principal component of the face data set (the top eigenface) using the power method on the covariance matrix $M$. Print the approximate eigenvalue $\mathbf{x}_k^TM\mathbf{x_k}$ and iterate until this stabilizes. How many iterations are required?
"""

# %%
#Power method code
x = np.random.randn(n,1) #Random initial vector

#Insert your code here


plt.figure()
plt.imshow(np.reshape(x,(m,m)),cmap='gray')
plt.title('Top principal direction (power method)')

# %%
"""
### Exercise

Repeat the same exercise except with one of the digits from the MNIST data set.
"""

# %%
import graphlearning as gl

data,labels = gl.datasets.load('mnist')
gl.utils.image_grid(data, n_rows=10, n_cols=15, title='Some MNIST images', fontsize=26)
print(data.shape)
print(labels.shape)
print(np.unique(labels))

#Insert code here
ind = labels == 3 #Use only 3's
mean_digit = np.mean(data[ind,:],axis=0)
X = data[ind,:] - mean_digit
M = X.T@X

#Display the top principal component image
n = len(mean_digit)
m = int(np.sqrt(len(mean_digit)))
plt.figure()
plt.imshow(np.reshape(mean_digit,(m,m)),cmap='gray')
plt.title('Mean Digit')

#Power method code
x = np.random.randn(n,1) #Random initial vector

#Insert your code here



plt.figure()
plt.imshow(np.reshape(x,(m,m)),cmap='gray')
plt.title('Top principal direction (power method)')

# %%
"""
## Orthogonal Iteration

To compute the top $k$ eigenvectors, we use orthogonal iteration, which generalizes the power method to task of computing multiple top eigenvectors. The orthogonal iteration starts from a random $n\times k$ matrix $Q_0$ and iterates
$$Q_{k+1}R_{k+1} = AQ_k,$$
where the left hand side is the QR-factorization of the right hand side. In particular, $Q_k$ for $k\geq 1$ has orthonormal columns and $R_k$ is upper triangular.
"""

# %%
"""
### Exercise (small matrix)

Use the orthogonal iteration to compute all eigenvectors and eigenvalues of the matrix
$$A = \begin{pmatrix}
2&-1&0&-1\\
-1&2&-1&0\\
0&-1&2&-1\\
-1&0&-1&2
\end{pmatrix}.$$
Check your result against the output of `numpy.linalg.eig`. How many iterations do you need for an accurate result? You can use `np.linalg.qr` for QR-factorization.

Recall that the columns of $Q_k$ converge to the top $k$ eigenvectors of $A$, while the diagonal entries of $R$ contain the eigenvalues.
"""

# %%
import numpy as np

A = np.array([[2,-1,0,-1],[-1,2,-1,0],[0,-1,2,-1],[-1,0,-1,2]])
print("A=",A)

vals,vecs = np.linalg.eig(A)
print('Eigenvalues=',vals)
print('Eigenvectors=',vecs) #Columns are eigenvalues

#Orthogonal iteration (Insert Code Here)



print('Orthogonal Iteration')
print('Eigenvalues: ',np.diag(R))
print('Eigenvectors: ',Q)


# %%
"""
### Exercise (Eigenfaces)

Use the orthogonal iteration method to compute the top $k=10$ principal directions (e.g., eigenfaces) for the Olivetti face dataset. Compare to the output of `scipy.sparse.linalg.eigsh`.
"""

# %%
import graphlearning as gl
from sklearn import datasets

ds = datasets.fetch_olivetti_faces()
data = ds['data']
labels = ds['target']

#Centered covariance matrix
mean_face = np.mean(data,axis=0)
X = data - mean_face
M = X.T@X

#Use eigsh to get subset of eigenvectors
#('LM'=largest magnitude, k=10 eigenvectors)
vals, vecs = sparse.linalg.eigsh(M, k=10, which='LM')
vals, P = vals[::-1], vecs[:,::-1] #Returns in opposite order

#Insert orthogonal iteration code to compute Q,R below


#Display the top principal component images
gl.utils.image_grid(P.T, n_rows=1, n_cols=k, title='Top Principal Components (Eigenfaces)', fontsize=26, normalize=True, transpose=False)
gl.utils.image_grid(Q.T, n_rows=1, n_cols=k, title='Orthogonal Iteration (Eigenfaces)', fontsize=26, normalize=True, transpose=False)
gl.utils.image_grid(-Q.T, n_rows=1, n_cols=k, title='Negated Orthogonal Iteration (Eigenfaces)', fontsize=26, normalize=True, transpose=False)

# %%
"""
### Exercise

Repeat the same exercise for one of the MNIST digits.
"""

# %%

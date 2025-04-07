# %%
"""
# Solving linear equations in Numpy
This notebook overviews some basic approaches to solving linear systems and eigenvalue problems in [NumPy](https://NumPy.org/).
"""

# %%
"""
## Solving small linear systems
We can use the `numpy.linalg.solve` command to exactly solve small or moderate size linear systems of full rank. We can check if a system is full rank using `numpy.linalg.matrix_rank`. Below we show how to solve a linear system with matrix
$$A = \begin{pmatrix} 
4 & -1 & 0 & 0\\
-1 & 4 & -1 & 0\\  
0 & -1 & 4 & -1 \\  
0 & 0 & -1 & 4 
\end{pmatrix},
$$
which appears in the linear solvers for certain partial differential equations involving the Laplacian. Solving linear systems is the subject of Chapter 4 in the textbook, and direct solutions are usually computed via QR factorization (Section 4.7) or Gaussian elimination (which we do not cover in the book). 
"""

# %%
import numpy as np

A = np.array([[4,-1,0,0],
              [-1,4,-1,0],
              [0,-1,4,-1],
              [0,0,-1,4]])

k = np.linalg.matrix_rank(A)
print('Rank = ',k)

b = np.ones(4)
x = np.linalg.solve(A,b)
print('Solution = ',x)
print('A*x=',A@x,'b=',b)

# %%
"""
## Solving large sparse linear systems
Direct solvers for linear systems typically struggle with large systems of linear equations. When the matrix $A$ is has some structure, such as symmetric positive definiteness and sparsity (meaning many entries are zero), there are far more efficient solvers based on indirect iterative methods. Below we use the conjugate gradient method (covered in Chapters 6 and 11 of the textbook) to solve a linear system with a matrix $A$ of the same form as above, but much larger. In this case, we use the `scipy` package, which has support for sparse matrices. 
"""

# %%
from scipy import sparse

n = 1000
o = np.ones(n)
A = sparse.diags(4*o) - sparse.diags(o[1:],1) - sparse.diags(o[1:],-1)
print(A)

x0 = np.random.randn(n) #Initialization
b = np.ones(n)
x,_ = sparse.linalg.cg(A,b,x0=x0) #Call conjugate gradient method

print('Solution = ',x)
print('A*x=',A*x,'b=',b)

# %%
"""
Notice the solution is not exact, but is correct up to several decimal places. 

### Exercise
Change the upper diagonal entries of $A$ to be $-2$ instead of $-1$, so that $A$ is not symmetric, and the conjugate gradient method cannot be applied. In this case, the more general method GMRES can still be applied. Write Python code below to solve the same linear system using `sparse.linalg.gmres`.
"""

# %%
from scipy import sparse

# %%
"""
## Solving small eigenvalue problems
We can use the `numpy.linalg.eig` or `numpy.linalg.eigh` (the latter is for symmetric matrices) to compute all of the eigenvectors and eigenvalues of a matrix $A$. We do this below for the $4\times 4$ matrix $A$ introduced above. Eigenvalues and eigenvectors are the subject of Chapter 5 in the textbook.
"""

# %%
import numpy as np

A = np.array([[4,-1,0,0],
              [-1,4,-1,0],
              [0,-1,4,-1],
              [0,0,-1,4]])

vals, vecs = np.linalg.eigh(A)
print('Eigenvalues = ',vals)
print('Eigenvectors = \n',vecs)

# %%
"""
Notice the eigenvalues are returned in ascending order. The columns of the array `vecs` contain the eigenvectors. As an alternative to using a linear solver, we can directly compute $A^{-1}$ using the spectral decomposition and use this to solve the linear system, as is done in the code below. Computing powers of symmetric (or more generally, self-adjoint) matrices using spectral decompositions is covered in Section 5.3 of the textbook). 
"""

# %%
import numpy as np

A = np.array([[4,-1,0,0],
              [-1,4,-1,0],
              [0,-1,4,-1],
              [0,0,-1,4]])

vals, vecs = np.linalg.eigh(A)
Q = vecs
L = np.diag(1/vals)
Ainv = Q@L@Q.T
print(Ainv)

b = np.ones(4)
x = Ainv@b
print('Solution = ',x)
print('A*x=',A@x,'b=',b)

# %%
"""
### Exercise
Use the spectral factorization above to compute the matrix square root $A^{1/2}$ of $A$. Verify the matrix square root is correct by squaring it.  
"""

# %%
"""
## Solving large sparse eigenvalue problems
For large sparse symmetric positive definite matrices (which often show up in applications), it is usually intractable to compute all of the eigenvectors and eigenvalues, and in fact, we are usually only interested in the $k$ top eigenvectors, which are those with the largest eigenvalues (or sometimes the smallest). For this, we can use the `scipy.sparse.eigs` and `scipy.sparse.eigsh` commands (the latter is again for symmetric matrices). These methods employ iterative indirect solvers, like the power method and QR method introduced in the book in Section 5.6.
"""

# %%
from scipy import sparse

n = 1000
o = np.ones(n)
A = sparse.diags(4*o) - sparse.diags(o[1:],1) - sparse.diags(o[1:],-1)

vals,vecs = sparse.linalg.eigsh(A, k=10, which='SM') #find k=10 smallest eigenvalues

print('Eigenvalues=',vals)

# %%
"""
### Exercises

1. Approximate the solution of the large sparse linear system above using a truncated eigenvector decomposition. That is, approximate the inverse matrix by $Q_k\Lambda_k^{-1}Q_k^T$, where we use only the first $k=100$ eigenvectors (in this case use the smallest). Compare your results to the conjugate gradient method. This exercise is related to the truncated singular value decomposition (SVD) of a matrix, which is the best low rank approximation of a matrix, as discussed in Section 8.2 of the textbook.
2. Use `numpy.random.randn` to construct a moderately sized rectangular matrix $X$. Compute the singular value decomposition (SVD) of $X$ by using the code above to find the eigenvector decomposition of $A=X^TX$. Compare your results to `np.linalg.svd`. SVD is the subject of Chapter 5 in the textbook.
3. Repeat exercise 2 with a large sparse matrix $X$ and use `scipy.sparse.eigsh` instead, comparing to `scipy.sparse.svds`. One possible choice for $X$ is the $(n-1)\times n$ matrix with $x_{i,i-1}=-1$, $x_{i,i+1}=1$, and all other entries zero. 
"""



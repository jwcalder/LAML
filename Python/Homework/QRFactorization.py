# %%
"""
# QR Factorization

The QR factorization of a square matrix $A$ produces an orthogonal matrix $Q$ and an upper triangular matrix $R$ such that
$$A = QR.$$
The QR factorization is useful for solving the linear equation $A\mathbf{x}=\mathbf{b}$ by solving the equivalent equation $QR\mathbf{x}=\mathbf{b}$, or rather
$$R\mathbf{x} = Q^T \mathbf{b},$$
where we recall that $Q^T$ is the inverse of $Q$, since $Q$ is orthonormal. As $R$ is upper triangular, we can solve $R\mathbf{x}=Q^T\mathbf{b}$ by back substitution. While we focus on square matrices in this notebook, it is important to note that QR factorization can also be applied to non-square matrices, in which case there are interesting connections to least squares solutions to linear systems.

## QR via Gram-Schmidt

The simplest way to compute a QR factorization is with the Gram-Schmidt algorithm. We assume $A$ is a square $n\times n$ non-singular matrix (i.e., its columns are linearly independent). The main steps of the Gram-Schmidt algorithm are presented in pseudocode below.

Let $\mathbf{a}_1,\mathbf{a}_2,\dots,\mathbf{a}_n$ denote the columns of $A$. We initialize $\mathbf{q}_1=\mathbf{a}_1/r_{11}$, where $r_{11}=\|\mathbf{a}_1\|$ and repeat the steps below for $k=2$ through $k=n$.

1. Compute $r_{jk} = \mathbf{q}_j\cdot \mathbf{a}_k$ for $j \leq k-1$ and $r_{jk}=0$ for $j>k$.
2. Compute $\mathbf{x}_k = \mathbf{a}_k - \sum_{j=1}^{k-1} r_{jk}\mathbf{q}_j$ and $r_{kk} = \|\mathbf{x}_k\|$.
3. Compute the $k$th column of $Q$, given by $\mathbf{q}_k = \frac{\mathbf{x}_k}{r_{kk}}$.

The vectors $\mathbf{q}_k$ form the columns of $Q$, and by definition
$$\mathbf{a}_k = \sum_{j=1}^k r_{jk} \mathbf{q}_j,$$
for all $k$, which is equivalent to the QR factorization statement
$$A = QR.$$

"""

# %%
"""
### Exercise

Write Python code for QR factorization using Gram-Schmidt below. You can use the template in the code below. Try your algorithm on some toy matrices.
"""

# %%
import numpy as np

def QR_GS(A):
    """QR via Gram-Schmidt
    ======
    Produces a QR factorization via standard Gram-Schmidt.
    The algorithm is numerically unstable and may not return orthonormal Q.

    Parameters
    ----------
    A : numpy array, float
        Non-singular matrix to perform QR on. A should be mxn with m >= n
        and all columns linearly independent.

    Returns
    -------
    Q : numpy array, float
        Orthogonal basis.
    R : numpy array, float
        Upper triangular matrix R so that A=QR
    """

    #Get shapes of matrices and initialize Q,R
    m,n = A.shape
    Q = np.zeros((m,n))
    R = np.zeros((n,n))

    #First step 
    R[0,0] = 
    Q[:,0] = 

    #The code you insert below can be vectorized
    #so that only one line is required for each computation. 
    #Alternatively you can add loops.
    for k in range(1,m):
        R[:k,k] =  #Define entries of R
        xk = 
        R[k,k] = 
        Q[:,k] = 

    return Q,R

# %%
"""
As an example, we define the matrix
$$A = \begin{pmatrix}
1 & 1\\
0 & 1
\end{pmatrix}.$$
The QR factorization is $A = I A$, where $I$ is the identity matrix (in fact, we always have $Q=I$ for any $A$ that is already upper triangular). We can verify this in Python code below.
"""

# %%
A = np.array([[1,0],[1,1]])
print("A=\n",A.T)

Q,R = QR_GS(A.T)
print("Q=\n",Q)
print("R=\n",R)

# %%
"""
In the code above, replace $A$ with $A^T$. How does the QR factorization change? Verify the QR factorization for $A^T$ by hand.
"""

# %%
"""
Let us check below that the QR factorization works for randomly generated matrices.
"""

# %%
A = np.random.rand(10,10)
Q,R = QR_GS(A)
print(np.linalg.norm(A - Q@R))

# %%
"""
We see above that the norm $\|A-QR\|$ is close to machine precision, around $10^{-15}$ in this case. We can also check the orthogonality of $Q$ with the code below, which computes the norm $\|I - Q^TQ\|$. Clearly $Q$ is very close to orthogonal, up to machine precision.
"""

# %%
print(np.linalg.norm(Q.T@Q - np.eye(Q.shape[0])))

# %%
"""
### Loss of orthogonality and numerical instabilities

In exact arithmetic, the Gram-Schmidt procedure works to produce a valid QR factorization. However, in the inexact world of floating point arithmetic, the method is numerically unstable and floating point roundoff errors accumulate, leading to a loss of orthogonality in $Q$. When $Q$ is not orthogonal, the definition of $R$ is incorrect and does not lead to a correct factorization.

The instabilities in Gram-Schmidt can be observed for very large matrices, or for ill-conditioned matrices. Our first example is for large matrices.
"""

# %%
A = np.random.rand(1000,1000)
Q,R = QR_GS(A)
print(np.linalg.norm(A - Q@R))
print(np.linalg.norm(Q.T@Q - np.eye(Q.shape[0])))

# %%
"""
Note that both $\|A-QR\|$ and $\|I - Q^TQ\|$ are much larger than the machine precision of around $10^{-15}$ we saw earlier, but they are still quite small.


"""

# %%
"""
### Ill-conditioned matrices

Random matrices are in fact quite easy to compute with as they are often well-conditioned. The situation can be substantially worse for other matrices that are poorly conditioned. An example of an ill-conditioned matrix is the Hilbert matrix
$$A = \begin{pmatrix}
1 & \frac12 & \frac13 & \cdots & \frac1n\\
\frac12 & \frac13 & \frac14 & \cdots & \frac{1}{n+1}\\
\frac13& \frac14 & \frac15 & \cdots & \frac{1}{n+2}\\
\vdots & \vdots & \vdots & \ddots & \vdots\\
\frac1n & \frac{1}{n+1} & \frac{1}{n+2} & \cdots & \frac{1}{2n-1}
\end{pmatrix}.$$

"""

# %%
def hilbert_matrix(n):
    """Hilbert Matrix
    ======
    Returns the nxn Hilbert matrix with (i,j) entry 1/(i+j+1), i,j=0,...,n-1

    Parameters
    ----------
    n : int
        Size of matrix (nxn)

    Returns
    -------
    A : numpy array, float
        Hilbert Matrix
    """

    x = np.arange(n)[:,None]
    A = 1/(x + x.T + 1)
    return A

# %%
"""
Check the performance of your QR algorithm on the Hilbert matrix.
"""

# %%
print('Original Gram-Schmidt')
A = hilbert_matrix(20)
Q,R = QR_GS(A)
print('||A-QR||=',np.linalg.norm(A - Q@R))
print('||I - Q^TQ||',np.linalg.norm(Q.T@Q - np.eye(Q.shape[0])))

# %%
"""
Your code should produce $Q$ and $R$ with $A=QR$ up to machine precision, but the matrix should $Q$ fail to be orthogonal. The loss of orthogonality arises from floating point round-off errors that accumulate. These arise from the step $\mathbf{x}_k = \mathbf{a}_k - \sum_{j=1}^{k-1} r_{jk}\mathbf{q}_j$ when the result $\mathbf{x}_k$ is very small compared to the two terms in the difference (i.e., $\mathbf{a}_k$ is nearly in the span $\mathbf{q}_1,\dots,\mathbf{q_{k-1}}$). This occurs when the matrix $A$ is ill-conditioned.
"""

# %%
"""
### Floating point round-off errors

If this is your first experience thinking about floating point numbers and roundoff errors, try running the code below, which should return 1 (it does not, due to accumulation of floating point roundoff errors, which is particularly bad when operating with very small and very large numbers at the same time).
"""

# %%
n = int(1e6)
b = 1e6
c = b

#In exact arithmetic, the loop below just adds 1 to c, and is the same as c=c+1
#In floating point arithmetic, the errors in c += 1/n accumulate, expecially
#since 1/n is far smaller than c.
for i in range(n):
    c += 1/n

#Since we start at c=b and the loop above should just perform c=c+1
#we should have c=b+1 and so c-b=1. This is not the case in floating point
#arithmetic.
print(c-b)

# %%
"""
Below are some even simpler examples of round-off error.
"""

# %%
print(1+1e-16)
print(0.6 == 0.6)
print(0.1 + 0.2 + 0.3 == 0.6)
print(0.1 + 0.2 + 0.3)

# %%
"""
### Re-orthogonalization

There are several ways to address the numerical instability of Gram--Schmidt for QR factorization. Here, we will use the re-orthogonalization trick, which essentially just repeats the orthogonalization step a second time. The steps are given below.

1. Compute $s_{jk} = \mathbf{q}_j\cdot \mathbf{a}_k$ for $j \leq k-1$.
2. Compute $\mathbf{v} = \mathbf{a}_k - \sum_{j=1}^{k-1} s_{jk}\mathbf{q}_j$.
3. Compute $t_{jk} = \mathbf{q}_j\cdot \mathbf{v}$ for $j \leq k-1$.
4. Compute $\mathbf{x}_k = \mathbf{v} - \sum_{j=1}^{k-1} t_{jk}\mathbf{q}_j$.
5. Set $r_{jk} = s_{jk} + t_{jk}$ for $j \leq k-1$.
6. Set $r_{kk} = \|\mathbf{x}_k\|$ and $\mathbf{q}_k = \frac{\mathbf{x}_k}{r_{kk}}$.

Steps 1-2 are the first orthogonalization, while steps 3-4 are the second one. In exact arithmetic we have $t_{jk}=0$ and steps 3-4 do nothing. In in-exact floating point arithmetic, steps 3-4 correct for a loss of orthogonality in the computation of $\mathbf{v}$.
"""

# %%
"""
### Exercise

Implement the Gram--Schmidt with re-orthogonalization in Python. Use the code template below. You may want to first *vectorize* your original Gram-Schmidt code so it does not have any loops, aside from the outer loop over $k$.
"""

# %%
def QR_GS_RO(A):
    """QR via Gram-Schmidt with Re-orthogonalization
    ======
    Produces a QR factorization via standard Gram-Schmidt.
    The algorithm is numerically unstable and may not return orthonormal Q.

    Parameters
    ----------
    A : numpy array, float
        Non-singular matrix to perform QR on. A should be mxn with m >= n
        and all columns linearly independent.

    Returns
    -------
    Q : numpy array, float
        Orthogonal basis.
    R : numpy array, float
        Upper triangular matrix R so that A=QR
    """

    #Get shapes of matrices and initialize Q,R
    m,n = A.shape
    Q = np.zeros((m,n))
    R = np.zeros((n,n))

    #First step 
    R[0,0] = 
    Q[:,0] = 

    for k in range(1,m):

        #First orthogonalization
        s = 
        v = 

        #Re-orthogonalization
        t = 
        xk = 

        #Set entries of Q and R
        R[:k,k] = s + t
        R[k,k] = np.linalg.norm(xk)
        Q[:,k] = xk/R[k,k]

    return Q,R

# %%
"""
Let's try the method on the Hilbert matrix. If your code is correct, you should find that $Q$ is orthogonal up to machine precision.
"""

# %%
print('\nGram-Schmidt with reorthogonalization')
A = hilbert_matrix(20)
Q,R = QR_GS_RO(A)
print('||A-QR||=',np.linalg.norm(A - Q@R))
print('||I - Q^TQ||',np.linalg.norm(Q.T@Q - np.eye(Q.shape[0])))

# %%
"""
### Additional exercises

"""

# %%
"""
1. QR algorithms in Python packages.
  * Find an implementation of QR factorization in Numpy or Scipy (or any other Python package).
  * Compare the implementation of QR that you found to the original Gram--Schmidt as well as the re-orthogonalized version. Compute both $\|A-QR\|$ and $\|I - Q^TQ\|$ for all three methods. Try random matrices, and ill-conditioned ones.
  * Can you find out what algorithm is used to compute QR for the implementation you found?
  * Compare the run-times of all three algorithms for large matrices, say around $1000\times 1000$.
"""

# %%
import time

A = hilbert_matrix(20)

#Gram-Schmidt QR
print('\nGram-Schmidt')
Q,R = QR_GS(A)
print(np.linalg.norm(A - Q@R,ord=np.inf))
print(np.linalg.norm(Q.T@Q - np.eye(Q.shape[0])))

#Gram-Schmidt QR with re-orthogonalization
print('\nGram-Schmidt with re-orthogonalization')
Q,R = QR_GS_RO(A)
print(np.linalg.norm(A - Q@R,ord=np.inf))
print(np.linalg.norm(Q.T@Q - np.eye(Q.shape[0])))

#Insert your code to compare to Numpy or Scipy
print('\nNumpy or Scipy')

#Computation time
A = np.random.rand(2000,2000)
start_time = time.time()
Q,R = QR_GS(A)
print("\nGram-Schmidt: %s seconds." % (time.time() - start_time))

start_time = time.time()
Q,R = QR_GS_RO(A)
print("\nGram-Schmidt with re-orthogonalization: %s seconds." % (time.time() - start_time))

#Insert your code to compare to Numpy or Scipy
print("\nNumpy or Scipy: %s seconds." % (time.time() - start_time))

# %%
"""
2. Write Python code to solve the linear system $A\mathbf{x}=\mathbf{b}$ with $QR$ factorization. Test your method on some large random matrices. You can use `np.linalg.inv` to compute the inverse of the upper triangular matrix $R$.
"""

# %%
n = 100
A = np.random.rand(n,n)
x_true = np.random.rand(n,1)
b = A@x_true

#Insert your code to solve the linear system Ax=b with your QR code



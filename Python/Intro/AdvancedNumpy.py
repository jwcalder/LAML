# %%
"""
# Advanced NumPy
This notebook overview some more advanced aspects of the [NumPy](https://NumPy.org/) package, such as slicing, logical indexing, and broadcasting. For a more in-depth tutorial, see this [NumPy Tutorial](https://NumPy.org/doc/stable/user/quickstart.html).
"""

# %%
"""
###Slicing arrays
It is often the case that you need to access only part of an array, say, a column of a matrix, or just the first few entries of a vector. NumPy has many useful ways to slice into arrays. Some examples of slicing rows or columns are below.
"""

# %%
import numpy as np

A = np.random.rand(5,3)
print(A)
print(A[:,0]) #First column of A
print(A[0,:]) #First row of A

# %%
"""
To extract only some of the entries in a given column or row of an array, the indexing notation "a:b:c" can be used. Generally this means start indexing at a, increment by c, and stop *before* you get to b. It is important to note that b is not included in the range of indexing.

Some important points: If a is ommitted, it is taken as 0. If b is ommitted, the indexing goes to the end of the array. If any numbers are negative, the array is treated as periodic. Examples are given below. It is a good idea to master this type of indexing, as it is used very often in NumPy.
"""

# %%
import numpy as np

a = np.arange(0,19,1) #Array of numbers from 0 to 18=19-1 going by 1
print(a)
print(a[0:7])
print(a[:7])
print(a[7:])
print(a[10:-2])  #Note the -2 means 2 before the end of the array
print(a[::3])

# %%
"""
We can mix the indexing above with slicing 2D or 3D arrays.
"""

# %%
import numpy as np

A = np.reshape(np.arange(30),(3,10))
print(A)
print(A[:,::2])
print(A[0:2,:])
print(A[0,:])
print(A[:,[3,4]])

# %%
"""
###Logical indexing
It is often useful to index an array logically, by another true/false array. The code below draws a random array, and replaces any entry greater than $0.5$ with $10$, and all other entries are set to zero.
"""

# %%
import numpy as np

A = np.random.rand(3,5)
print(A)

I = A > 0.5  #Boolean true/false
print(I)
A[I] = 10
print(A)
A[~I] = 0
print(A)

# %%
"""
Finally, there are many useful functions in NumPy for computing with arrays, that is, finding the mean or median value of each column, summing all the entries in a matrix, finding the norm of the matrix, etc. Pretty much any function you may want to apply to a matrix has a built-in NumPy function to do the job, and as per the discussion above, this is much faster than writing a loop to do it in Python. Examples below.
"""

# %%
import numpy as np

A = np.reshape(np.arange(30),(3,10))

print(A)
print(np.sum(A))   #Sums all entries in A
print(np.max(A,axis=0))  #Gives sums along axis=0, so it reports column sums
print(np.sum(A,axis=1))  #Row sums

# %%
"""
In the code above, experiment with replacing "sum" by "mean" or "median".
"""

# %%
"""
###Broadcasting
When operating on NumPy arrays, it is sometimes useful to perform elementwise operations on two arrays of different sizes. Say, you want to add a vector to every row of a matrix. NumPy broadcasts automatically, as long as the sizes of the last dimensions match up.
"""

# %%
import numpy as np

A = np.reshape(np.arange(30),(3,10))
print(A)

x = np.arange(10)
print(x.shape)
print(x)
print(A+x) #Adds the row vector of all ones to each row of A
print(A+1)

# %%
"""
Suppose we wanted to add a column vector to each column of A. The code below fails, why?
"""

# %%
import numpy as np

A = np.reshape(np.arange(30),(3,10))
print(A)

x = np.ones((3,))
print(x)
print(A+x) #Adds the row column of all ones to each row of A


# %%
"""
Answer: For broadcasting, the sizes of the last dimensions must match up. Above they are 10 and 3, which are not compatible, whereas in the first example they were 10 and 10. To fix this, you can add a dummy axis to x, to make the last dimensions match up, using np.newaxis.
"""

# %%
print(x)
print(x.shape)
print(x[:,None])
print(x[:,None].shape)
print(A+x[:,None])

# %%
"""
##Exercises

All of the exercises below should be completed with Numpy operations, indexing, slicing, etc., and should not involve any kind of Python loop.

1. Write code to compute the mean row vector for a matrix $X$. Do the same for the mean column vector.
"""

# %%
X = np.random.rand(10,3)
mrow = ??#mean row
mcol = ??#mean column

#To check your code
print(X)
print(mrow)
print(mcol)

# %%
"""
2. Write code to center the rows of the matrix $X$, by subtracting the mean row vector from every row. Do the same for column centering.
"""

# %%
X = np.random.rand(10,3)
mrow = ??#mean row
Xc = ??#Centered X by subtracting mrow

#To check your code
print(X)
print(Xc)

#Compute the mean of rows of Xc to check that they are zero
??

#Do the same for columns

# %%
"""
3. Write code to compute the Euclidean norm of every row of a matrix $X$. Do the same for columns. Your answers should be contained in a vector.
"""

# %%
X = np.random.rand(10,3)
Xrows = ?? #Norms of rows
Xcols = ?? #Norms of columns

print(Xrows)
print(Xrows)

# %%
"""
4. Write code to normalize a matrix $X$ so that all rows are unit vectors. Do the same for columns.
"""

# %%
X = np.random.rand(10,3)
Xrows = ?? #Norms of rows
Xnorm = ?? #Normalize rows

print(X)
print(Xnorm)

#Compute the norms of the rows of X to check they are 1
??

# %%
"""
5. Create the column vector $\mathbf{x} = (0,1,2,3,4)^T$ as an $n\times 1$ array in NumPy using `x = np.arange(5)[:,None]`. Then add $\mathbf{x}$ and its transpose $\mathbf{x}^T$, which is `x.T` in numpy. The results should be a $5\times 5$ matrix. Print and inspect the result. Explain why NumPy allowed you to add an $n\times 1$ array to a $1\times n$ array, and the result, using the rules of [broadcasting](https://NumPy.org/doc/stable/user/basics.broadcasting.html) in NumPy.
"""

# %%


# %%
"""
6. [Challenge] Write a Python program that uses the [Sieve of Eratosthenes](https://en.wikipedia.org/wiki/Sieve_of_Eratosthenes) to find all prime numbers between $2$ and a given integer $n$. In contrast to the question in last lecture's [notebook](https://colab.research.google.com/drive/1R23TOiQ1s8AgCvDk8xbc90fxBu2vYF7D?usp=sharing), here you should use NumPy arrays and NumPy functions. In particular, use only one outer `while` loop, and convert all the inner loops into NumPy operations and functions.
"""

# %%

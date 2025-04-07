# %%
"""
#Introduction to NumPy
This notebook contains an introduction to [NumPy](https://NumPy.org/), a fundamental pacakge for scientific computing in Python. For a more in-depth tutorial, see this [NumPy Tutorial](https://NumPy.org/doc/stable/user/quickstart.html).
"""

# %%
"""
##Python Packages
The power of Python is due to the availability of many useful 3rd party packages. All packages must be explicitly imported in python code, and import statements normally go at the top of a python script.

There are several ways to import packages. Below shows a few ways to import NumPy.
"""

# %%
import numpy
import numpy as np
from numpy import array
from numpy import sin

# %%
"""
The second way "import numpy as np" is the most common. The "as np" gives a shortcut name "np" to numpy, to make coding easier. You can change the shortcut "np" to be anything you like. All functions within NumPy have to be explicitly referenced with code like "numpy.array", "numpy.max", or "np.array" and "np.max", depending on how you imported NumPy.

The third option is used if you want to import just a single function from within NumPy (or another package). With the third option, you have only imported the function "array", and can use it without the "numpy" or "np" prefix. Nothing else from NumPy is imported. The third option is almost never used with NumPy, but is often used for specialty packages when you only need to import a specific function.

Option 4: "from numpy import *" - this imports every NumPy function so one doesn't need the "np." Generally this is not done for NumPy since it is large - more conventional for smaller modules. Note the builtin sum command is overwritten with this command for NumPy. Dangerous to do this with multiple packages at once- can confuse cross-listed functions.
"""

# %%
"""
##NumPy arrays
The basic data type in NumPy is an array.

###Initializing arrays
Arrays can have any number of dimensions, and can be initalized in a variety of ways. The code below shows how to initialize arrays a length 3 one-dimensional array in various ways.
"""

# %%
import numpy as np

a = np.array([1,2,3])
b = np.ones(10)
c = np.zeros((3,))
print(a)
print(b)
print(c)

# %%
"""
For multi-dimensional arrays, we simply add the sizes of the other dimensions. You can have as many dimensions as you like.
"""

# %%
import numpy as np

d = np.array([[1,2,3,4,5],[6,7,8,9,10]])  #2D array of size 2x5, initialized manually
e = np.ones((2,5))          #2D ones array of size 2x5
f = np.zeros((2,5,8))       #3D zeros array of size 2x5x8
g = np.random.rand(2,8)     #Random 2D array of size 2x8
print(d)
print(e)
print(f)
print(g)
print(d.shape)

# %%
"""
Most NumPy construtors take a python tuple, i.e., the (2,5) on the second line above, as input. This is why there are two sets of brackets. This is not always the case, since we can see that np.random.rand takes multiple arguments instead of a python tuple. You will get used to this eventually.

The values of arrays are referenced with square brackets. Python starts with zero as the first index (contrary to, e.g., Matlab, which starts indexing at 1).
"""

# %%
import numpy as np

x = np.random.rand(3,5)
print(x)
print(x[0,1])
print(x[-2,-1])

x[0,0] = np.pi
print(x)

# %%
"""
It is often useful to create a list of numbers increasing at regular intervals. There are two commands for doing this.
"""

# %%
x = np.arange(10)
print(x)

x = np.arange(2,10)
print(x)

x = np.arange(2,10,2)
print(x)

x = np.linspace(0,10,21)
print(x)

# %%
"""
Notice that `np.arange` is not inclusive, so the list stops before the upper limit of 10 in this case, while `np.linspace` is inclusive. The last argument of `np.linspace` is the desired length of the list, while the last argument for `np.arange` is the step size.
"""

# %%
"""
###Operations on NumPy arrays
The advantage of using NumPy arrays instead of Python lists is that NumPy contains very efficient implementations of common operations (e.g., matrix/vector multiplication) with NumPy arrays. These operations are executed with highly optimized compiled C-code.

The code below shows the advantage of using NumPy operations instead of basic Python. This is just for adding two arrays; the difference becomes far larger for more complicated operations, like matrix/vector operations. The moral of this example is to try an "vectorize" all your NumPy code, using built in NumPy functions, instead of using loops.
"""

# %%
import numpy as np
import time

#Let's make two long lists that we wish to add elementwise
n = 300000
A = n*[1]
B = n*[2]
print('A=',end='');print(A)
print('B=',end='');print(B)

#Let's add A and B elementwise using a loop in Python
start_time = time.time()
C = n*[0]
for i in range(n):
    C[i] = A[i] + B[i]
python_time_taken = time.time() - start_time
print('C=',end='');print(C)
print("Python took %s seconds." % python_time_taken)

#Let's convert to NumPy and add using NumPy operations
A = np.array(A)
B = np.array(B)

start_time = time.time()
C = A + B
numpy_time_taken = time.time() - start_time
print("NumPy took %s seconds." % (numpy_time_taken))

print('NumPy was %f times faster.'%(python_time_taken/numpy_time_taken))


# %%
"""
As you can see in the code above, the operation $+$ in NumPy adds two arrays elementwise. The operation $-$ is the same for subtraction. The operation $*$ multiplies two arrays of the same size elementwise, and is *not* matrix multiplication. To perform matrix multiplication, use $@$ in NumPy. Matrices must have compatible sizes to perform matrix multiplication. Some examples are below.
"""

# %%
import numpy as np

A = np.random.rand(3,5)
B = np.random.rand(3,5)

print('A*B=',end='');print(A*B)  #elementwise multiplication
print('A-B=',end='');print(A-B)  #elementwise subtraction

#Examples of matrix multiplication and matrix/vector multiplication
print('A@B.T=',end='');print(A@B.T)   #B.T means the transpose of B
C = np.random.rand(5,7)
D = np.ones((5,))
print('A@C=',end='');print(A@C)
print('A@D=',end='');print(A@D)

# %%
"""
**Warning**: There are some situations in Python where $*$ can refer to matrix/vector multiplication, and this can easily be confusing. We will not see this often (if ever), but it arises when one uses matrix data types, instead of NumPy arrays. There are matrix data types in NumPy and Scipy. See examples below. I prefer to use only NumPy arrays and use @ for matrix multiplication, so it is always clear what operation is intended.
"""

# %%
import numpy as np
import scipy.sparse as sparse

A = np.random.rand(3,3)
x = np.random.rand(3,1)

print(A@x)
print(A*x)

A = np.matrix(A)
print(A*x)

A = sparse.csr_matrix(A)
print(A*x)

# %%
"""
## Computing dot products

There are several ways to compute dot products in Numpy. If the vectors are stored as column vectors, we can use the $x^Ty$, or we can elementwise multiply the vectors and sum.  Alternatively,if the arrays are one dimensional, we can use `np.dot`.
"""

# %%
x = np.random.rand(3,1)
y = np.random.rand(3,1)

print('x =',x)
print('y =',y)

#Different ways to compute the dot product
print('x dot y = ',x.T@y)
print('x dot y = ',np.sum(x*y))

#For one dimensional arrays
x = x.flatten()
y = y.flatten()

print('x =',x)
print('y =',y)
print('x dot y = ',np.dot(x,y))

# %%
"""
The `np.sum` command is very useful in a variety of situations. Other operations, such as `np.mean`, `np.median`, etc., are available as well.

In fact, most mathamatical operations work fine with NumPy arrays, and work elementwise. This includes the power operation $**$, and any special functions in NumPy. Some examples are below.
"""

# %%
import numpy as np

A = np.reshape(np.arange(10),(2,5))

print(A)
print(A**2) #Square all elements in A
print(np.sin(A)) #Apply sin to all elements of A
print(np.sqrt(A)) #Square root of all elements of A
print(A**.5)

# %%
"""
##Exercises

1. Write a function that takes two vectors $\mathbf{x}$ and $\mathbf{y}$, of the same length $n$, and an $n \times n$ matrix $C$, and returns the inner products $\langle \mathbf{x},\mathbf{y}\rangle = \mathbf{x}^TC\mathbf{y}$. Test your function on some small matrices and vectors where you can also check the result by hand.
"""

# %%
def inner_product(x,y,C):

    return x.T@C@y

C = np.array([[2,1],[1,2]])
x = np.array([[1],[2]])
y = np.array([[2],[3]])
print(inner_product(x,y,C))

# %%
"""
2. Write a function that numerically approximates $\pi$ via the series
$$\frac{\pi^2}{6} = \sum_{n=1}^\infty \frac{1}{n^2}.$$
To do this, use `np.arange` to construct the array $(1,2,3,\dots,N)$, for a large integer $N$, and use numpy vectorized operations to approximate the series by the partial sum
$$S_N = \sum_{n=1}^N \frac{1}{n^2}.$$
Then you can approximate $\pi$ as
$$\pi \approx \sqrt{6 S_N}.$$
Your code should not have any loops. How many decimal places of $\pi$ can you approximate this way?
"""

# %%
#This is homework
#def approx_pi(N):
#
#    return ??

# %%
"""
3. Write a function that numerically approximates $\pi$ via the integral expression
$$\pi = 4\int_0^1 \sqrt{1-x^2} \, dx,$$
which is simply the area enclosed by the unit circle. Your function should not use any loops. Use a NumPy array and NumPy functions instead. How many decimal places of $\pi$ can you accurately compute? You should approximate the integral by a Riemann sum
$$\pi \approx 4\sum_{i=1}^n \sqrt{1-x_i^2}\Delta x,$$
where $\Delta x = 1/n$, $n$ is a large integer, and $x_i$ is a point in the interval $[(i-1)\Delta x,i\Delta x]$. Use `np.arange` to construct an numpy array containing $(x_1,\dots,x_n)$. [Challenge: Compare left point, right point, and mid-point rules for choosing $x_i$ in the Riemann sum.]

"""

# %%
def riemann_pi(n):
    x = np.linspace(0,1,n) #Left point
    dx = x[1]-x[0]
    x = x[1:] - dx/2 #Midpoint (remove the -dx/2 for right point)

    return 4*np.sum(np.sqrt(1-x**2))*dx

for p in [1,2,3,4,5,6]:
    n = 10**p
    print(n,riemann_pi(n))

# %%
"""
4. [Very challenging] To date, about 100 trillion digits of $\pi$ have been computed using the [Chudnovsky Algorithm](https://en.wikipedia.org/wiki/Chudnovsky_algorithm), which converges much faster than the methods above. Write a python program to implement the Chudnovsky algorithm and test how many terms you need to get 10 decimal places of accuracy.
"""

# %%

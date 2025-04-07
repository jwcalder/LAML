# %%
"""
#Introduction to Python
This notebook has a brief introduction to python. For something more in-depth, see this [Introduction to Python](https://www.w3schools.com/python/python_intro.asp).

"""

# %%
"""
## Variables
Python variables can hold many different kinds of objects, including integers, floating numbers, lists, strings, numpy arrays, and so on. The basic operations of addition and multiplication depend on which data type you work with. Make sure you understand the examples below.
"""

# %%
"""
Here is an example of numerical variables:
"""

# %%
a = 2
b = 4
print(a+b) #addition
print(a*b) #multiplication
print(a/b) #division
print(a**b)#Exponents a^b

# %%
"""
String variables:
"""

# %%
a = 'hello'
b = ' '
c = 'world'
print(a+b+c)
print(7*(a+b))

# %%
"""
Lists:
"""

# %%
a = [4,'hello world', 3.14159]
b = ['data science', 46]
print(len(a),len(b)) #Length of lists
print(a+b)
print(3*b)

#We can modify lists
a[0] = 6
print(a)

# %%
"""
Tuples: Python tuples are similar to lists but canot be modified. They can be defined using parentheses or without any. When Python functions return multiple arguments, they are usually returned as tuples.
"""

# %%
a = (4,'hello world', 3.14159)
b = 'data science', 46
print(len(a),len(b))
print(a+b)
print(3*b)

#We cannot modify a tuple
a[0] = 3

# %%
"""
##For loop

Loops are simple in python, using the range(n) function, which iterates over $0,1,2,...,n-1$.
"""

# %%
for i in range(3,20,3):
    print(i)
print('done')

# %%
"""
In Python, you can loop over lists very easily.
"""

# %%
x = [1,2,3,'hello','world']
for s in x:
    print(s)

# %%
"""
Often you may want to loop over a list and keep track of the index as well, which is easy to do.
"""

# %%
x = ['this','class','is','about','machine','learning']
for i,s in enumerate(x):
    print(i,s)

# %%
"""
In Python, there is no need to end a loop or conditional statement. The scope is deduced by your indentation. In the code above, the statement print(i) is indented, showing Python that this should be inside the for loop, while print('done') is not indented, which indicates the for loop scope has ended. It doesn't mater how many spaces you use for indentation, as long as you are consistent in each scope.

Try to run the code below. What is wrong? Can you fix it?
"""

# %%
for i in range(10):
    j = 2*i
 print(j)

# %%
"""
We can also use while loops.
"""

# %%
i = 0
while i < 10:
    print(i,end=':')
    i += 1

# %%
"""
##If statements
If statements work similarly to for statements concerning indentation.
"""

# %%
n = 60 #Choose any number you like
if n > 50:
    print('n is larger than 50')
else:
    print('n is less than or equal to 50')


# %%
"""
We can have longer if else statements:
"""

# %%
n = 41 #Choose any number you like
if n > 50:
    print('n is larger than 50')
elif n > 40:
    print('n is larger than 40 but less than or equal to 50')
else:
    print('n is smaller than or equal to 40')


# %%
"""
##Functions
It is easy to define new functions in python.
"""

# %%
def sum(a,b):
    c = a + b
    return c

print(sum(1,3))
print(sum(10,21))
print(sum('hello ','world'))

# %%
"""
## Exercises
These exercises should all be completed with basic Python, without importing any packages (e.g., do not use numpy, scipy, etc).


"""

# %%
"""
1. Write a Python function that approximates $\sin(x)$ using the Taylor expansion $\sin(x) \approx x - \frac{x^3}{3!} + \frac{x^5}{5!} - \frac{x^7}{7!}$. Test your function for simple known values of $\sin(x)$, such as $\sin(0)=0$, $\sin(\pi/4) = \frac{1}{\sqrt{2}}$, $\sin(\pi/2)=1$, and $\sin(\pi)=0$, etc. How good is the approximation?

"""

# %%
def taylor_sin(x):

    return ??

#Test your function (e.g. taylor_sin(3.14159) should be close to zero, taylor_sin(3.14159/2) should be close to 1)
print(taylor_sin(3.14159))
print(taylor_sin(3.14159/2))
print(taylor_sin(3.14159/4)**2)

# %%
"""
2. Write a Python function that computes the square root of a positive number using the Babylonian method. The Babylonian method to compute $\sqrt{S}$ for $S>0$ constructs the sequence $x_n$ by setting $x_0=S$ and iterating
$$x_{n+1} = \frac{1}{2}\left(x_n + \frac{S}{x_n}\right).$$
Your code can take as input a tolerance parameter $\varepsilon>0$, and should iterate until $|x_n^2 - S| \leq \varepsilon$, and then return $x_n$. Test your square root function to make sure it works.

Notes: You can use `abs` for absolute value in Python. Also note the use of the optional argument for `eps`.

"""

# %%
def babylonian_sqrt(S,eps=1e-6):

    return ??

#Test your function for different values of S (e.g., babylonian_sqrt(16))

# %%
"""
3. [Challenge (optional)] Prove that the iteration in the Babylonian method above converges quadratically to the square root of $S$. In particular, show that the error $\epsilon_n = \frac{x_n}{\sqrt{S}}-1$ satisfies
$$\epsilon_{n+1} = \frac{\epsilon_n^2}{2(\epsilon_n+1)}.$$
From this, we get that $\epsilon_n\geq 0$ for $n\geq 1$, and so
$$\epsilon_{n+1} \leq \frac{1}{2}\min\{\epsilon_n^2,\epsilon_n\}.$$
Why does the inequality above guarantee convergence (i.e., that $\epsilon_n\to 0$ as $n\to \infty$)?

"""

# %%
"""
4. Write a Python function that adds two vectors together. Represent your vectors as Python lists.
"""

# %%
def add(x,y):

    return ??

#Test your function (try others)
print(add([1,1,1,1],[2,3,4,7]))

# %%
"""
5. Write a Python function that computes the dot product between two vectors. Represent your vectors as Python lists. [Challenge: Choose an inner product that is not the dot product, and write a python function to compute it.]
"""

# %%
def dot_product(x,y):

    return ??

#Test your dot product (try other examples)
x = [1,2,3]
y = [1,0,1]
print(dot_product(x,y)) #Should be 4

# %%
"""
6. Write a function that multiplies a vector $\mathbf{x}$ (represented by a Python list) by a scalar $a\in \mathbb{R}$. The function should return $a\mathbf{x}$. Make sure to define a new list to store the ouptut (don't overwrite $\mathbf{x}$).
"""

# %%
def scalar_product(x,a):

    return ??

#Test your function

# %%
"""
7. Write a function that computes the Euclidean norm of a vector $\mathbf{x}$ represented as a list. (Hint: Use your dot product and square root functions).
"""

# %%
def norm(x):

    return ??

#Test your function

# %%
"""
8. Write a function that performs Gram-Schmidt on two vectors $\mathbf{u}_1,\mathbf{u}_2$. The output of Gram-Schmidt is two orthonormal vectors $\mathbf{v}_1$ and $\mathbf{v}_2$ defined by
$$\mathbf{v}_1 = \frac{\mathbf{u}_1}{\|\mathbf{u}_1\|}\ \ \text{and} \ \ \mathbf{v_2} = \frac{\mathbf{u}_2 - (\mathbf{u}_2 \cdot \mathbf{v}_1)\mathbf{v}_1}{\|\mathbf{u}_2 - (\mathbf{u}_2 \cdot \mathbf{v}_1)\mathbf{v}_1\|}.$$
"""

# %%
def gram_schmidt(u1,u2):

    return v1,v2

# %%
"""
9. The cross product between two vectors $\mathbf{x}=(x_1,x_2,x_3)$ and $\mathbf{y}=(y_1,y_2,y_3)$ in $\mathbb{R}^3$ is the vector $\mathbf{z} = \mathbf{x}\times \mathbf{y}$ defined by
$$\mathbf{z} = (x_2y_3-x_3y_2,x_3y_1-x_1y_3,x_1y_2-x_2y_1).$$
Write a Python function to compute the cross product and test it on several examples.
"""

# %%
def cross_product(x,y):

    return ??

#Test your cross product function

# %%
"""
10. Write a Python program that uses the [Sieve of Eratosthenes](https://en.wikipedia.org/wiki/Sieve_of_Eratosthenes) to find all prime numbers between $2$ and a given integer $n$.
"""

# %%
def prime_sieve(n):

    return ??

#Test by finding first 100 primes
primes = prime_sieve(100)
for p in primes:
    print(p)

# %%
"""
11. The Fibonacci sequence is given by
$$1,1,2,3,5,8,13,\dots.$$
The general rule is that the next number is the sum of the previous two numbers, i.e., $a_n = a_{n-1} + a_{n-2}$, with $a_0=a_1=1$. The golden ratio is the limit of the ratio of subsequent terms in the Fibonacci sequence
$$\phi = \lim_{n\to \infty}\frac{a_n}{a_{n-1}}.$$
It turns out that $\phi = \frac{1}{2}(1 + \sqrt{5})$ (proving this requires some work). Write a Python program to approximate the golden ratio $\phi$ by computing $N$ terms in the Fibonacci sequence and then computing
$$\phi_N = \frac{a_N}{a_{N-1}}$$
for a large value of $N$. How accurate is your approximation? Try different values for $N$.

"""

# %%
def approx_phi(N):

    return ??


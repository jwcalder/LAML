# %%
"""
#Newton's Method

This notebook gives a brief introduction to Newton's method.

We define a function to optimize. Newton's method converges in one step for quadratic functions, so we need to depart from the example we used for gradient descent, and consider a function with some higher-order non-quadratic terms.
"""

# %%
def f(x):
    return x[0]**2 + 4*x[1]**2 + (x[0] + x[1] + 1)**4 - 4*(x[0]+x[1]) - 1

def grad_f(x):
    T = (x[0] + x[1] + 1)**3
    f0 = 2*x[0] + 4*T - 4
    f1 = 8*x[1] + 4*T - 4
    return np.array([f0,f1])

def hess_f(x):
    T = (x[0] + x[1] + 1)**2
    f00 = 2 + 4*3*T
    f01 = 4*3*T
    f10 = f01
    f11 = 8 + 4*3*T
    return np.array([[f00,f01],[f10,f11]])


# %%
"""
The function we defined is
$$f(x) = x(0)^2 + 4x(1)^2 + (x(0)+x(1) +1)^4 - 4(x(0) + x(1)) - 1,$$
The gradient and Hessian are computed in the code above. This function has its global minimum at $x=(0,0)$, with minimal value $f(0,0)=0$.

We will now compare gradient descent with Newton's method for optimizing this $f$.
"""

# %%
import numpy as np

x_init = np.array([0.1,0.1])  #initial condition
num_steps = 100
alpha = 0.01  #For gradient descent

f_GD = np.zeros(num_steps)
f_NT = np.zeros(num_steps)

x_GD = x_init.copy()
x_NT = x_init.copy()

for i in range(num_steps):

    #Gradient Descent
    x_GD -= alpha*grad_f(x_GD)
    f_GD[i] = max(f(x_GD),0)

    #Newton's Method
    x_NT -= np.linalg.inv(hess_f(x_NT))@grad_f(x_NT)
    f_NT[i] = max(f(x_NT),0)

# %%
"""
Let's compare gradient descent and Newton's method for speed of convergence.
"""

# %%
import matplotlib.pyplot as plt

plt.plot(f_GD, label='Gradient Descent')
plt.plot(f_NT, label='Newton')
plt.yscale('log')
plt.xlabel('Number of steps', fontsize=16)
plt.ylabel('f(x)', fontsize=16)
plt.legend(fontsize=16)

# %%
"""
The difference is very clear; Newton's method gives superlinear convergence. To see the quadratic convergence of Newton's method, we need to set the $y$-axis to a loglog scale.
"""

# %%
plt.plot(-np.log(-np.log(f_GD)), label='Gradient Descent')
plt.xlabel('Number of steps', fontsize=16)
plt.ylabel('-log(-log(f(x)))', fontsize=16)
plt.legend(fontsize=16)

plt.figure()
plt.plot(-np.log(-np.log(f_NT)), label='Newton')
plt.xlabel('Number of steps', fontsize=16)
plt.ylabel('-log(-log(f(x)))', fontsize=16)
plt.legend(fontsize=16)

# %%
"""
#Exercises

1. Try choosing an initial point further away from the minimizer. Can you make Newton's method fail to converge?
2. Try a non-smooth function $f$.
"""

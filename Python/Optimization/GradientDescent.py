# %%
"""
#Gradient Descent

This notebook gives a gentle introduction to gradient descent.

First define the function you wish to minimize, and its gradient
"""

# %%
def f(x):
    return x[0]**2 + 4*x[1]**2 + x[0]*x[1]

def grad_f(x):
    return np.array([2*x[0] + x[1], 8*x[1] + x[0]])

# %%
"""
The function we defined is
$$f(x,y) = x^2 + 4y^2 + xy,$$
whose gradient is
$$\nabla f(x,y) = (2x + y,8y + x)^T.$$
This is a quadratic function with a global minimum at $(0,0)$, with value $f(0,0)=0$. We will now minmimize this function with gradient descent.
"""

# %%
import numpy as np

x = np.array([1.,1.])  #initial condition
num_steps = 100
alpha = 0.2

f_vals = np.zeros(num_steps)
dist_vals = np.zeros(num_steps)

for i in range(num_steps):
    x -= alpha*grad_f(x)
    f_vals[i] = f(x)
    print("Iteration: ",i,': f(x) =',f(x))
    dist_vals[i] = np.linalg.norm(x)

# %%
"""
To see if/how it worked, let's plot the energy and distance to the minimizer over each step of gradient descent.
"""

# %%
import matplotlib.pyplot as plt

plt.plot(f_vals, label='f(x_k)')
plt.plot(dist_vals, label='||x_k||')
plt.xlabel('Number of steps (k)', fontsize=16)
plt.legend(fontsize=16)

# %%
"""
In order to see a convergence rate, we should plot on a log scale.
"""

# %%
plt.plot(f_vals, label='f(x_k)')
plt.plot(dist_vals, label='||x_k||')
plt.yscale('log')
plt.xlabel('Number of steps', fontsize=16)
plt.legend(fontsize=16)

# %%
"""
To check the convergence rate, we can find a line of best fit in the log scale. That is, we expect the error curve is $y = C\mu^k$ for some $C>0$ and $0 < \mu < 1$, where $k$ is the iteration index. To find the convergence rate $\mu$, we take $\log$ on both sides to get $\log(y) = \log(C) + \log(\mu)k$. Thus, we can find a line of best fit of the form $\log(y) = b + mk$, and then $\mu = e^{m}$. 

The code below does this, and we see a rate of $0.4$ for $f(x_k)$, which means the error is reduced by a factor of $0.4$ at each iteration. This rate is the square of the rate for $\|x_k\|$, since $f$ is quadratic near the minimizer.
"""

# %%
k = np.arange(len(f_vals))
m,b = np.polyfit(k,np.log(f_vals),1)
mu = np.exp(m)
print('Convergence rate for f(x_k) (mu) = ', mu)

m,b = np.polyfit(k,np.log(dist_vals),1)
mu = np.exp(m)
print('Convergence rate for x_k (mu) = ', mu)
print('mu^2=',mu**2)

# %%
"""
We can also visualize the path taken by gradient descent. Try playing around with the time step $\alpha$ and see how the path changes.
"""

# %%
import matplotlib.pyplot as plt

x,y = 1,1 #Initial values of x=1 and y=1

plt.text(x,y,f'Initial point=({x},{y})',horizontalalignment='right')
alpha = 0.2
N  = 20
for i in range(N):
    plt.scatter(x,y,c='blue',marker='.')
    x_old,y_old = x,y
    grad = grad_f((x,y))
    x -= alpha*grad[0]
    y -= alpha*grad[1]
    plt.plot([x_old,x],[y_old,y],'cyan',alpha=0.5)

plt.text(x,y-0.1,f'Final point=({x:.2f},{y:.2f})',horizontalalignment='right')


# %%
"""
#Exercises

1. Try increasing the step size $\alpha$. How large can you make $\alpha$ before gradient descent becomes unstable and does not converge? Can you estimate the Lipschitz constant $L$ of the gradient $\nabla f$ this way? (recall that $\alpha \leq \frac{1}{L}$ was our condition for convergence of gradient descent).
2. Try using preconditioned gradient descent. For the preconditioner, use the diagonal of the Hessian matrix.
3. Try minimizing a nonconvex function with several local minima and saddle points. For example, try the function
$$f(x,y) = x^4 - 2x^2 + y^2.$$
"""

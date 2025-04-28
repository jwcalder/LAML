# %%
"""
#Momentum Descent

This notebook gives an introduction to he heavy ball method.

First define the function you wish to minimize, and its gradient
"""

# %%
def f(x):
    return x[0]**4 + x[1]**2

def grad_f(x):
    return np.array([4*x[0]**3,2*x[1]])

# %%
"""
The function we defined is
$$f(x,y) = x^4 + y^2,$$
whose gradient is
$$\nabla f(x,y) = (4x^3,2y).$$
This is convex, but not strongly convex, function with minimizer at $(0,0)$
"""

# %%
import numpy as np

x_gd = np.array([1.,1.])  #initial condition
x_hb = np.array([1.,1.])  #initial condition
y_ns = np.array([1.,1.])  #initial condition
x_ns = np.array([1.,1.])  #initial condition
x_hb_prev = x_hb
num_steps = 20
alpha = 0.05
beta = 0.7

f_vals_gd = np.zeros(num_steps)
f_vals_hb = np.zeros(num_steps)
f_vals_ns = np.zeros(num_steps)

l = 0
for i in range(num_steps):
    #Gradient descent update
    x_gd -= alpha*grad_f(x_gd)
    f_vals_gd[i] = f(x_gd)

    #Heavyball update
    x_temp = x_hb.copy()
    x_hb = x_hb - alpha*grad_f(x_hb) + beta*(x_hb - x_hb_prev)
    x_hb_prev = x_temp
    f_vals_hb[i] = f(x_hb)

    #Nesterov update
    l = (1 + np.sqrt(1 + 4*l**2))/2
    lnext = (1 + np.sqrt(1 + 4*l**2))/2
    ynext = x_ns - alpha*grad_f(x_ns)
    x_ns = ynext + ((l-1)/lnext)*(ynext - y_ns)   #Original Nesterov
    #x_ns = ynext + ((i-1)/(i+2))*(ynext - y_ns)  #Alternative Nesterov
    y_ns = ynext
    f_vals_ns[i] = f(y_ns)

# %%
"""
To see if/how it worked, let's plot the energy and distance to the minimizer over each step of gradient descent.
"""

# %%
import matplotlib.pyplot as plt

plt.plot(f_vals_gd, label='Gradient Descent')
plt.plot(f_vals_hb, label='Heavy Ball')
plt.plot(f_vals_ns, label='Nesterov')
plt.xlabel('Number of steps (k)', fontsize=16)
plt.legend(fontsize=16)

# %%
"""
In order to see a convergence rate, we should plot on a log scale.
"""

# %%
plt.figure(figsize=(10,10))
plt.plot(f_vals_gd, label='Gradient Descent')
plt.plot(f_vals_hb, label='Heavy Ball')
plt.plot(f_vals_ns, label='Nesterov')
plt.yscale('log')
plt.xlabel('Number of steps', fontsize=16)
plt.ylabel('f(x_t) - f(x_*)')
plt.legend(fontsize=16)
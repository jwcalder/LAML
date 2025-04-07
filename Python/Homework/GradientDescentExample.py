# %%
"""
#Gradient Descent Homework Example Solutions

"""

# %%
import numpy as np
import matplotlib.pyplot as plt

def grad_desc(f,grad_f,min_f,x0,num_steps=100,alpha=0.2,title='Gradient Descent'):
    x = np.array(x0).astype(float)  #initial condition

    f_vals = np.zeros(num_steps)
    for i in range(num_steps):
        x -= alpha*grad_f(x)
        f_vals[i] = f(x) - min_f

    print('Output of Gradient Descent=',x)

    plt.figure()
    plt.plot(f_vals, label='f(x_k)')
    plt.yscale('log')
    plt.xlabel('Number of steps', fontsize=16)
    plt.legend(fontsize=16)
    plt.title(title)

    k = np.arange(len(f_vals))
    m,b = np.polyfit(k,np.log(f_vals),1)
    mu = np.exp(m)
    plt.title(title)
    print('Convergence rate for f(x_k) (mu) = ', mu)

# %%
def f1(x):
    return x[0]**2 + 2*x[1]**2

def grad_f1(x):
    return np.array([2*x[0], 4*x[1]])

def f2(x):
    return x[0]**2 + 10*x[1]**2

def grad_f2(x):
    return np.array([2*x[0], 20*x[1]])

def f3(x):
    return np.sin(x[0])*np.sin(x[1])

def grad_f3(x):
    return np.array([np.cos(x[0])*np.sin(x[1]), np.sin(x[0])*np.cos(x[1])])

# %%
grad_desc(f1,grad_f1,0,[1,1],num_steps=20,alpha=0.35,title='f1')
grad_desc(f2,grad_f2,0,[1,1],num_steps=200,alpha=0.09,title='f2')
grad_desc(f3,grad_f3,-1,[1,4],num_steps=200,alpha=0.08,title='f3')

# %%
"""
#Gradient Descent Homework Example Solutions

"""

# %%
import numpy as np
import matplotlib.pyplot as plt

def cons_grad_desc(f,grad_f,min_f,x0,num_steps=100,alpha=0.2,title='Gradient Descent'):
    x = np.array(x0).astype(float)  #initial condition

    B = np.array([[1,-1],[-1,1]])
    f_vals = np.zeros(num_steps)
    for i in range(num_steps):
        x -= alpha*B@grad_f(x)
        f_vals[i] = f(x) - min_f

    print('Output of gradient descent=',x)

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
cons_grad_desc(f1,grad_f1,f1([4*np.pi/3,2*np.pi/3]),[0,2*np.pi],num_steps=20,alpha=0.25,title='f1')
cons_grad_desc(f2,grad_f2,f2([20*np.pi/11,2*np.pi/11]),[0,2*np.pi],num_steps=20,alpha=0.02,title='f2')
cons_grad_desc(f3,grad_f3,-1,[1,2*np.pi-1],num_steps=20,alpha=0.25,title='f3')



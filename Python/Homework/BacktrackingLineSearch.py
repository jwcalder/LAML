# %%
import numpy as np

def F(x):
    return x[0]**3 + 10*x[1]**2
def gF(x):
    return np.array([3*x[0]**2,20*x[1]])
def hF(x):
    return np.array([[6*x[0],0],[0,20]])

# %%
import matplotlib.pyplot as plt

x_gd = [np.array([1.0,1.0])]
alpha = 0.05
T = 100
for i in range(T-1):
    x = x_gd[-1]
    x_gd += [x-alpha*gF(x)]


x_bl = [np.array([1.0,1.0])]
beta = 0.9
gamma = 0.5
T = 100
for i in range(T-1):
    x = x_bl[-1]
    v = -gF(x)
    alpha = 1
    e = F(x + alpha*v) - F(x) + gamma*alpha*np.sum(v**2)
    while e > 0:
        alpha = alpha*beta
        e = F(x + alpha*v) - F(x) + gamma*alpha*np.sum(v**2)
    x_bl += [x+alpha*v]

r_gd = np.linalg.norm(np.array(x_gd),axis=1)
r_bl = np.linalg.norm(np.array(x_bl),axis=1)
it = np.arange(T)
plt.plot(it,r_gd,label='gd')
plt.plot(it,r_bl,label='backtracking')
plt.yscale('log')
plt.legend()

# %%
import matplotlib.pyplot as plt

x_nt = [np.array([1.0,1.0])]
T = 100
for i in range(T-1):
    x = x_nt[-1]
    x_nt += [x-np.linalg.inv(hF(x))@gF(x)]


x_ntbl = [np.array([1.0,1.0])]
beta = 0.9
gamma = 0.5
T = 100
for i in range(T-1):
    x = x_ntbl[-1]
    v = -np.linalg.inv(hF(x))@gF(x)
    alpha = 2
    e = F(x + alpha*v) - F(x) - gamma*alpha*np.sum(gF(x)*v)
    while e > 0:
        alpha = alpha*beta
        e = F(x + alpha*v) - F(x) - gamma*alpha*np.sum(gF(x)*v)
    x_ntbl += [x+alpha*v]

r_nt = np.linalg.norm(np.array(x_nt),axis=1)
r_ntbl = np.linalg.norm(np.array(x_ntbl),axis=1)
it = np.arange(T)
plt.plot(it,r_nt,label='Newton')
plt.plot(it,r_ntbl,label='Backtracking Newton')
plt.yscale('log')
plt.legend()

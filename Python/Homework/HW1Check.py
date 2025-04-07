# %%
"""
Paste your code for all three functions in the code block below.
"""

# %%
import numpy as np

def babylonian_sqrt(S,eps=1e-6):
    ??
    return x

def cross_product(x,y):
    ??
    return z

def approx_pi(N):
    ??
    return pi

# %%
"""
Then run the block below to test. Do not modify anything in the code below. Modify your functions to make it work. "Work" means the error is reasonably close to zero and the cross product is correct. I will test you submissions with this code, except the numbers will change.
"""

# %%
import numpy as np

print('Babylonian Square Root:',end=' ')
try:
    a = babylonian_sqrt(16)
    print('babylonian_sqrt(16)=%f'%a+', Error=%f'%(abs(4-a))+'\n')
except:
    print('babylonian_sqrt not found.\n\n')

print('\nCross Product:',end=' ')
try:
    b = cross_product([1,0,2],[2,1,0])
    print('cross_product([1,0,2],[2,1,0])=[%f,%f,%f]\n'%(b[0],b[1],b[2]))
    print('               True value: [1,0,2]x[2,1,0]=[-2,4,1]\n')
except:
    print('cross_product not found.\n\n')

print('\nApproximate pi:',end=' ')
try:
    pi = approx_pi(1000)
    print('approx_pi(1000)=%f, Error=%f\n'%(pi,abs(pi-np.pi)))
except:
    print('approx_pi not found.\n\n')

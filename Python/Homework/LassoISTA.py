# %%
"""
# Lasso Regression via ISTA

Here we solve the lasso regression problem with the iterative
shrinkage–thresholding algorithm (ISTA) algorithm and apply it to the diabetes data set
"""

# %%
from sklearn import datasets

diabetes = datasets.load_diabetes()
print(diabetes.DESCR)

# %%
"""
The returned object `diabetes` contains the data and target. To see how to access it, we can just print it out.
"""

# %%
print(diabetes)

# %%
"""
We see there are arrays `data` and `target`. Let's give those new names and check their shapes.
"""

# %%
x = diabetes.data
y = diabetes.target

print(x.shape)
print(y.shape)

# %%
"""
Let's plot some of the data against the target to get a feel for the dataset. The plots do not show any clear corellations between individual variables and the target disease progression.
"""

# %%
import matplotlib.pyplot as plt

feature_labels =  ['Age (years)',
                   'Sex',
                   'Body mass index',
                   'Average blood pressure',
                   'TC (total serum cholesterol)',
                   'LDL (low-density lipoproteins)',
                   'HDL (high-density lipoproteins)',
                   'TCH (total cholesterol / HDL)',
                   'LTG (log of serum triglycerides level)',
                   'GLU (blood sugar level)']

for i in [0,2,3,4,5,6,7,8,9]:
    plt.figure()
    plt.scatter(x[:,i],y)
    plt.xlabel(feature_labels[i],fontsize=16)
    plt.ylabel('Disease progression',fontsize=16)

# %%
"""
Let's now split into training and testing sets and run a linear regression, reporting the training and testing error. Play around with the regularization parameter $\lambda$ (`lam`) below.
"""

# %%
from sklearn.model_selection import train_test_split
import numpy as np

#To handle affine data, we extend the features by a constant 1
x = diabetes.data
y = diabetes.target
n = x.shape[0]
x = np.hstack((x,np.ones((n,1))))

#Train/test split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33)

#linear regression on training set
lam = 0.01 #Regularization parameter
m = x_train.shape[1]
A = x_train.T@x_train + lam*np.eye(m)
rank = np.linalg.matrix_rank(A)
if rank != m:
    print("Matrix is singular!")

#Since the system is only 11x11, we don't care that much how Ax=b
#solved. We'll just use numpy.linalg.solve
w = np.linalg.solve(A,x_train.T@y_train)

#print testing and training mean squared error
train_error = np.sqrt(np.mean((y_train - x_train@w)**2))
test_error = np.sqrt(np.mean((y_test - x_test@w)**2))
print('Training error: ',train_error)
print('Testing error: ',test_error)

# %%
"""
We can also print out the weights corresponding to each feature, to understand which are more imporant for the regression/prediction.
"""

# %%
print('\nFeature weights:')
print('================')
for i in range(len(feature_labels)):
    print(feature_labels[i]+': %.2f'%w[i])
print('Offest b: %.2f'%w[10])

# %%
"""
We now implement Lasso regression with the ISTA algorithm.
"""

# %%
import numpy as np

def shrink(x,lam):
    return np.sign(x)*np.maximum(0,np.absolute(x) - lam/2)

X = x_train
Y = y_train

w = np.random.randn(X.shape[1])
alpha = 0.001
lam = 0.1
for i in range(10000):
    lasso_energy = np.sum((X@w - Y)**2 + lam*np.sum(np.abs(w)))
    if i % 1000 == 0:
        print('Iteration =', i, 'Lasso energy =',lasso_energy)
    z = w - 2*alpha*X.T@(X@w-Y)
    w = shrink(z,2*alpha*lam)

#print testing and training mean squared error
train_error = np.sqrt(np.mean((y_train - x_train@w)**2))
test_error = np.sqrt(np.mean((y_test - x_test@w)**2))
print('Training error: ',train_error)
print('Testing error: ',test_error)

print('\nFeature weights:')
print('================')
for i in range(len(feature_labels)):
    print(feature_labels[i]+': %.2f'%w[i])
print('Offest b: %.2f'%w[10])

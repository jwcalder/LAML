# %%
"""
# Support Vector Machines Homework (toy problem)

This notebook solves Exercise 3.3 in Chapter 7.
"""

# %%
from sklearn.svm import SVC
from sklearn import datasets
import numpy as np

X = np.array([[0,0,0],
              [1,-1,1],
              [0,1,1],
              [2,-2,3]])
y = np.array([0,1,0,1])
clf = SVC(kernel='linear')
clf.fit(X,y)

#Print the solution
w = clf.coef_.flatten()
b = -clf.intercept_
print('w=',w)
print('b=',b)
print('Classification function: ',X@w-b)
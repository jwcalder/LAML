# %%
"""
# Support Vector Machines

This notebook is a brief introduction to support vector machines.
"""

# %%
"""
We first train SVMs on some toy blob datasets. For this, we will define a function that helps plot the classification decision regions of an SVM (or any classifier for that matter).
"""

# %%
import numpy as np
import matplotlib.pyplot as plt

#Generates a plot of the classifier decision regions
#corresponding to a classifier trained on the datapoints
#in X, with true labels L, and classifier clf
#All arguments besides X,L,clf are optional and can be ignored
def plot_region(X,L,clf,alpha=0.5,cmap='Paired',cp=np.array([0.5,2.5,6.5,8.5,10.5,11.5]),
                markers = ['o','s','D','^','v','p'],vmin=0,vmax=12,markersize=75,
                linewidths=1.25,markerlinewidths=1):
    x,y = X[:,0],X[:,1]
    f = 0.1*np.maximum(np.max(np.abs(x)),np.max(np.abs(y)))
    xmin, xmax = np.min(x)-f,np.max(x)+f
    ymin, ymax = np.min(y)-f,np.max(y)+f
    plt.figure()
    for i,color in enumerate(np.unique(cp[L])):
        sub = cp[L] == color
        plt.scatter(x[sub],y[sub],zorder=2,c=cp[L][sub],cmap=cmap,edgecolors='black',
                    vmin=vmin,vmax=vmax,linewidths=markerlinewidths,marker=markers[i],s=markersize)

    X,Y = np.mgrid[xmin:xmax:0.01,ymin:ymax:0.01]
    z = clf.predict(np.c_[X.ravel(),Y.ravel()]).reshape(X.shape)
    plt.contourf(X, Y, cp[z],alpha=alpha,cmap=cmap,antialiased=True,vmin=vmin,vmax=vmax)
    plt.contour(X, Y, z, colors='black',linewidths=linewidths,antialiased=True)
    plt.xlim((xmin,xmax))
    plt.ylim((ymin,ymax))

# %%
"""
Let's now generate some blob data, train an SVM classifier using sklearn, and plot the resulting decision regions. For now, we'll use the built-in SVM optimizer in sklearn. In the homwork you'll get a chance to optimize it yourself.

Try a diffrenet number of centers (classes) and try different numbers of points. You can also try different kernels ('rbf','poly'). We haven't discussed those yet in class. The random_state allows you to get the same random clusters every time you run the code, so you can compare different kernels on the same data.
"""

# %%
from sklearn.svm import SVC
from sklearn import datasets

n = 100
X,L = datasets.make_blobs(n_samples=n, centers=3)
clf = SVC(kernel='linear',decision_function_shape='ovr')
clf.fit(X,L)

#Plot the region
plot_region(X,L,clf)

#Print the solution
print('w=',clf.coef_)
print('b=',clf.intercept_)

# %%
"""
We now consdier some real data. We will use the breast cancer classification dataset from sklearn.
"""

# %%
data = datasets.load_breast_cancer()
print(data.DESCR)

# %%
"""
The code below performs a random train/test split, trains an SVM on the training set, and evaluates the performance.
"""

# %%
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split

x = data.data
y = data.target
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33)

#Train SVM
clf = SVC(kernel='linear')
clf.fit(x_train,y_train)

#Training accuracy
y_pred = clf.predict(x_train)
train_acc = np.mean(y_pred == y_train)*100
print('Training Accuracy: %.2f%%'%train_acc)

#Testing accuracy
y_pred = clf.predict(x_test)
test_acc = np.mean(y_pred == y_test)*100
print('Testing Accuracy: %.2f%%'%test_acc)


# %%
"""
### Exercises

1. Play round with different choices of kernels in SVM. The documentation for the function is [here.](https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html)
2. As with linear regression, try running SVM over many different random train/test splits and report the average and standard deviation of accuracy. (You can also use a k-fold cross-validation).
"""

# %%
"""
# Handwritten digit recognition

Here we tackle the problem of handwritten digit recognition with several basic machine learning algorithms. We will first install the [GraphLearning](https://github.com/jwcalder/GraphLearning) Python package, which gives easy access to the MNIST digits dataset.
"""

# %%
#pip install graphlearning -q

# %%
"""
We now load the MNIST dataset and display some of the images.
"""

# %%
import graphlearning as gl

digits,labels = gl.datasets.load('mnist')
print(digits.shape)
print(labels.shape)
gl.utils.image_grid(digits)

# %%
"""
The standard train/test split for MNIST is to take the first 60000 images as training, and the last 10000 as testing. Let's run SVM and k-nearest neighbors on this train/test split. The code will take a long time to run (5-10 minutes).
"""

# %%
from sklearn.svm import SVC

#standard train/test split
x_train = digits[:60000,:]
x_test = digits[60000:,:]
y_train = labels[:60000]
y_test = labels[60000:]

#Train SVM
clf = SVC(kernel='linear')
clf.fit(x_train,y_train)

#Training accuracy
y_pred = clf.predict(x_train)
train_acc = np.mean(y_pred == y_train)*100
print('SVM Training Accuracy: %.2f%%'%train_acc)

#Testing accuracy
y_pred = clf.predict(x_test)
test_acc = np.mean(y_pred == y_test)*100
print('SVM Testing Accuracy: %.2f%%'%test_acc)

# %%
"""
Let's plot some of the misclassified images.
"""

# %%
num_show = 20
img = np.zeros((10*num_show,784))
I = y_pred != y_test
x_wrong = x_test[I]
y_wrong = y_test[I]
for i in range(10):
    I = y_wrong == i
    img[num_show*i:num_show*i + min(num_show,np.sum(I)),:] = x_wrong[I,:][:num_show]

gl.utils.image_grid(img,n_rows=10,n_cols=num_show)

# %%
"""
### Exercise

To speed up the training, try training on a much smaller random subset of the training data `x_train`. You can use `np.random.choice` for this. How does the accuracy vary with the size of the training set? Generate some plots to show the relationship between accuracy and size of the training set.
"""

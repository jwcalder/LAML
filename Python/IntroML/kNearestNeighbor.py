# %%
"""
# k-nearest neighbor classification

This notebook is a brief introduction to support vector machines.
"""

# %%
"""
We first train k-nearest neighbor classifiers on some toy blob datasets. For this, we will define a function that helps plot the classification decision regions.
"""

# %%
import numpy as np
import matplotlib.pyplot as plt

def plot_region(X,L,clf,alpha=0.5,cmap='Paired',cp=np.array([0.5,2.5,6.5,8.5,10.5,11.5]),
                markers = ['o','s','D','^','v','p'],vmin=0,vmax=12,fname=None,markersize=75,
                linewidths=1.25,markerlinewidths=1,res=0.1,train_pts=None):

    plt.figure()
    x,y = X[:,0],X[:,1]
    xmin, xmax = np.min(x),np.max(x)
    ymin, ymax = np.min(y),np.max(y)
    f =0.1*np.maximum(np.max(np.abs(x)),np.max(np.abs(y)))
    xmin -= f
    ymin -= f
    xmax += f
    ymax += f
    c = cp[L]
    c_u = np.unique(c)

    for i,color in enumerate(c_u):
        sub = c==color
        plt.scatter(x[sub],y[sub],zorder=2,c=c[sub],cmap=cmap,edgecolors='black',vmin=vmin,vmax=vmax,linewidths=markerlinewidths,marker=markers[i],s=markersize)
        if train_pts is not None:
            plt.scatter(x[sub & train_pts],y[sub & train_pts],zorder=2,c=np.ones(np.sum(sub&train_pts))*5.5,cmap=cmap,edgecolors='black',vmin=vmin,vmax=vmax,linewidths=markerlinewidths,marker=markers[i],s=markersize)


    X,Y = np.mgrid[xmin:xmax:res,ymin:ymax:res]
    points = np.c_[X.ravel(),Y.ravel()]
    z = clf.predict(points)
    z = z.reshape(X.shape)
    plt.contourf(X, Y, cp[z],alpha=alpha,cmap=cmap,antialiased=True,vmin=vmin,vmax=vmax)

    X,Y = np.mgrid[xmin:xmax:res,ymin:ymax:res]
    points = np.c_[X.ravel(),Y.ravel()]
    if len(np.unique(c)) == 2:

        if hasattr(clf, "decision_function"):
            z = clf.decision_function(points)
        else:
            z = clf.predict_proba(points)
            z = z[:,0] - z[:,1] + 1e-15
        z = z.reshape(X.shape)
        plt.contour(X, Y, z, [0], colors='black',linewidths=linewidths,antialiased=True)
    else:
        z = clf.predict(points)
        z = z.reshape(X.shape)
        plt.contour(X, Y, z, colors='black',linewidths=linewidths,antialiased=True)
    plt.xlim((xmin,xmax))
    plt.ylim((ymin,ymax))

# %%
"""
Let's now generate some blob data, train an k-nearest neighbor classifier using sklearn, and plot the resulting decision regions. Try varying the number of centers (classes), numbers of data points, number of neighbors, and the metric.
"""

# %%
from sklearn.neighbors import KNeighborsClassifier
from sklearn import datasets

n = 100
X,L = datasets.make_blobs(n_samples=n, centers=3)
clf = KNeighborsClassifier(n_neighbors=10, metric='euclidean') #Try 'cosine' or 'cityblock'
clf.fit(X,L)

#Plot the region
plot_region(X,L,clf,res=0.1) #decrease res to get higher resolution of boundary, but can be very slow

# %%
"""
We now consider some real data. We will use the breast cancer classification dataset from sklearn.
"""

# %%
data = datasets.load_breast_cancer()
print(data.DESCR)

# %%
"""
The code below performs a random train/test split, trains a knn classifier on the training set, and evaluates the performance. We also compare against SVM.
"""

# %%
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split

x = data.data
y = data.target
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33)

#Train k-nn classifier
clf = KNeighborsClassifier(n_neighbors=10)
clf.fit(x_train,y_train)

#Training accuracy
y_pred = clf.predict(x_train)
train_acc = np.mean(y_pred == y_train)*100
print('knn Training Accuracy: %.2f%%'%train_acc)

#Testing accuracy
y_pred = clf.predict(x_test)
test_acc = np.mean(y_pred == y_test)*100
print('knn Testing Accuracy: %.2f%%'%test_acc)

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
### Exercises

1. Play round with different metrics and number of neighbors for the k-nearest neighbor classifier.
2. Try running the classification over many different random train/test splits and report the average and standard deviation of accuracy. (You can also use a k-fold cross-validation).
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
The standard train/test split for MNIST is to take the first 60000 images as training, and the last 10000 as testing. Let's run a k-nearest neighbor classifier and a support vector machine (VM) on this train/test split. The code will take a long time to run (5-10 minutes).
"""

# %%
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

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

#Train k-nn classifier
clf = KNeighborsClassifier(n_neighbors=10, metric='cosine')
clf.fit(x_train,y_train)

#Training accuracy
y_pred = clf.predict(x_train)
train_acc = np.mean(y_pred == y_train)*100
print('knn Training Accuracy: %.2f%%'%train_acc)

#Testing accuracy
y_pred = clf.predict(x_test)
test_acc = np.mean(y_pred == y_test)*100
print('knn Testing Accuracy: %.2f%%'%test_acc)

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

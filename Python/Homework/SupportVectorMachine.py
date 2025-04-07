# %%
"""
# Support Vector Machine Homework

In this homework you'll write your own code to train a support vector machine.
"""

# %%
"""
Let's start with the two-point synthetic dataset from class. This is just to check that our code is working properly, using an example where we know the true solution.
"""

# %%
import numpy as np

#Parameters
lamb = 0.1
beta = 1
alpha = 1

#label at z is +1 and -z is -1
#Optimal w = z / ||z||^2 and b=0
z = np.array([1,2])

#Data matrix and labels
X = np.vstack((z,-z))
y = np.array([1,-1])

#Random initialization
w = np.random.randn(2)
b = np.random.randn(1)

for i in range(1000):

    #Insert your code here to compute the gradients and loss. You can
    #Use as many additional lines of code as needed (i.e., don't try too
    #hard to put the whole computation in one line)
    grad_w =
    grad_b =
    loss =

    w -= alpha*grad_w
    b -= alpha*grad_b
    if i % 100 == 0:
        print('Iteration',i,'Loss =',loss,'w =',w,'b =',b,'Prediction =',np.sign(X@w-b))


# %%
"""
Now let's move on to apply our algorithm to real data from the MNIST dataset. We now install [GraphLearning](https://github.com/jwcalder/GraphLearning) and load the MNIST digits.
"""

# %%
#pip install graphlearning -q

# %%
import graphlearning as gl

data,labels = gl.datasets.load('mnist')
gl.utils.image_grid(data,n_rows=25,n_cols=25)

# %%
"""
Let's now create a binary classification problem to classify pairs of MNIST digits.
"""

# %%
import numpy as np

digits = (4,9)

#Subset to pair of digits
mask = (labels == digits[0]) | (labels == digits[1]) #Logical or
X = data[mask,:]
y = labels[mask].astype(float)

#convert to -1,1 labels
y = y -  np.min(y) - 1
y[y>-1] = 1

#We now standardize the data to range of -1 to 1
X -= X.min()
X /= X.max()
X = 2*X-1

# %%
"""
We now perform a train/test split.
"""

# %%
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

# %%
"""
Now we train the svm on the training set, evaluating testing accuracy at each iteration.
"""

# %%
#Size of data
m,n = X_train.shape

#Parameters
lamb = 0.01
beta = 1
alpha = 0.1

#Random initialization
w = np.random.randn(n)
b = np.random.randn(1)

for i in range(1000):

    #Insert your code here to compute the gradients and loss. You can
    #Use as many additional lines of code as needed (i.e., don't try too
    #hard to put the whole computation in one line)
    grad_w =
    grad_b =
    loss =

    w -= alpha*grad_w
    b -= alpha*grad_b

    if i % 100 == 0:
        train_acc = round(100*np.mean(np.sign(X_train@w - b) == y_train),2)
        test_acc = round(100*np.mean(np.sign(X_test@w - b) == y_test),2)
        print('Iteration',i,'Loss =',loss,'Train Accuracy =',train_acc,'Test Accuracy =',test_acc)

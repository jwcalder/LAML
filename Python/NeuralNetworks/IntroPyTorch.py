# %%
"""
#Intro to Pytorch

This is a brief introduction to PyTorch, which is a python package for deep learning. Much of PyTorch is similar to Numpy, except that torch keeps track of gradients for you with a method called *automatic differentiation* or in PyTorch, *autograd*, and it has support for accelerated computations on graphics processing units (GPUs) (in addition to support for various neural network architectures). 

Automatic differentiation uses the chain rule to compute derivatives of any composition of functions. In PyTorch, you simply need to ensure you write every step of the computation using torch functions and modules, and set the requires_grad flag to inform torch of which variables it should track gradients with respect to. This notebook gives an introduction to the autograd feature in Pytorch. The next lecture/notebook will introduce neural networks in PyTorch.

The code below gives a simple example of the autograd feature in Pytorch. This makes training neural networks very easy since we don't need to explicitly compute the gradients with back propagation, etc. Torch also has support for performing computations on GPUs, which we will cover in a later lecture.
"""

# %%
import torch
x = torch.tensor(4.0, requires_grad = True)
z = x**3

z.backward() #Invokes backpropgation to compute gradient with chain rule.

grad = 3*x**2
print('Gradient = ', grad) 
print('Torch Gradient = ', x.grad.data) 

# %%
import torch
x = torch.tensor([1.0,2.0,3.0], requires_grad = True)
z = torch.sum(torch.exp(torch.sin(x**3)))  #You can only call backward on scalars

z.backward() #Invokes backpropgation to compute gradient with chain rule.

grad = 3*x**2*torch.cos(x**3)*torch.exp(torch.sin(x**3))
print('Gradient = ', grad) 
print('Torch Gradient = ', x.grad.data) 

# %%
"""
Try playing around with the examples above to make sure you understand how autograd works. In particular, what is the role of the torch.sum in the second example?
"""

# %%
"""
It is important to point out that gradients get accumulated in PyTorch, which makes the example below fail to give the correct answer.
"""

# %%
import torch
x = torch.tensor(4.0, requires_grad = True)
z = x**3
z.backward() #Invokes backpropgation to compute gradient with chain rule.

#A second example, after already computing the first above
w = x**2
w.backward()

grad = 2*x
print('Gradient = ', grad) 
print('Torch Gradient = ', x.grad.data) 

# %%
"""
Exercise: Fix the code above by reseting the gradient of x immediately before computing w=x**2. This can be done by setting x.grad = None (you can also set x.grad = torch.tensor(0.0), but then you need to know the shape of the tensor).
"""

# %%
"""
## Example: Gradient descent

As a first example, let's use PyTorch to perform gradient descent on a simply toy function. Notice that we will not need to code the gradient, since PyTorch will compute it for us automatically.

\begin{equation}
  f(x,y) = \frac{1}{2} x^2 + \frac{1}{4} y^4
\end{equation}
and the gradient is
\begin{equation}
  \nabla f(x,y) = (x, y^3).
\end{equation}
"""

# %%
#Function you wish to minimize
def f(x,y):
    return (1/2)*x**2 + (1/4)*y**4

# %%
import torch

#Need to define x and y as tensors with requires_grad=True
x = torch.tensor(1.0, requires_grad=True)
y = torch.tensor(2.0, requires_grad=True)

#Time step alpha for gradient descent
alpha = 0.1

#Gradient descent iterations
for t in range(100):
    
    #Zero out gradients before computation, so that we do not carry over gradients
    #from the previous iteration
    x.grad, y.grad = None, None

    #Compute the function you wish to optimize and then call backward
    #Then x.grad and y.grad will be Tensors holding the gradient of the 
    #loss with respect to x and y.
    loss = f(x,y)
    loss.backward()

    #Manually update (x,y) using gradient descent. Wrap in torch.no_grad()
    #to stop autograd from tracking gradients.
    with torch.no_grad():
        #Print current state of gradient descent
        print("(x,y)=(%.3f,%.3f), f(x,y)=%.3f, Grad f(x,y)=(%.5f,%.5f)"%(x,y,f(x,y),x.grad.data,y.grad))

        x -= alpha * x.grad
        y -= alpha * y.grad

# %%
"""
## Example: Polynomial fitting (or regression)

As another example application, we will show how to use PyTorch to fit a low degree polynomial to a given function, that is, given $f(x)$, find $a,b,c,d,e\in \mathbb{R}$ so that

$$f(x) \approx a + bx + cx^2 + dx^3 + ex^4.$$

We'll use the function $f(x)=\sin(x)$, but you can change this to anything else. In this example, we will use an optimizer in PyTorch to handle the gradient descent steps automatically.
"""

# %%
import torch
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
plt.ion()

#Create Tensors to hold input and outputs.
#By default, requires_grad=False, which indicates that we do not need to
#compute gradients with respect to these Tensors during the backward pass.
x = torch.linspace(-np.pi, np.pi, 2000)

# Create a random tensor for the weights. For a third order polynomial, we need
# 5 weights: y = a + b x + c x^2 + d x^3 + e x^4
# Setting requires_grad=True indicates that we want to compute gradients with
# respect to these Tensors during the backward pass.
a = torch.randn(5, requires_grad=True)

#Use an optimizer so we can avoid explicitly coding gradient descent.
#Need to provide a list of the parameters to be optimized over and the learning rate.
#The SGD optimizer performs basic gradient desecent. PyTorch has other more sophisticated
#optimizers that we will use later. 
optimizer = optim.SGD([a], lr=5e-4)  #Learning rate = time step = alpha

for t in range(10**4):

    #Set the gradients to zero (in place of a.grad = None, etc.)
    optimizer.zero_grad()

    #Forward pass: compute predicted y using operations on Tensors.
    y = a[0] + a[1]*x + a[2]*x**2 + a[3]*x**3 + a[4]*x**4

    #Compute the loss using operations on Tensors.
    loss = torch.mean(torch.abs(y - torch.sin(x))**2)

    #Call backward to compute gradients. This call will compute the
    #gradient of loss with respect to all Tensors with requires_grad=True,
    #namely the tensor of coefficients a.
    loss.backward()

    print('Iteration: ',t,'Loss:', loss.item())
    #Take a step of gradient descent. The optimizer handles this for us
    optimizer.step()

b = a.tolist()
print(f'Result: sin(x) ~ %.2f + %.2fx + %.2fx^2 + %.2fx^3 + %.2fx^4'%(b[0],b[1],b[2],b[3],b[4]))
plt.figure()
plt.plot(x,torch.sin(x),label='Sin(x)')
plt.plot(x,y.detach(),label='Polynomial approximation') #Try this without .detach()
plt.legend()

# %%
"""
## Example: Linear classifier

Here, we'll give an example training a linear classifier on some toy data. Our linear classifier will have the form

$$f(\mathbf{x}) = \sigma(\mathbf{x}\cdot \mathbf{w} - b),$$

where $\mathbf{x}\in \mathbb{R}^d$ is the input, $\mathbf{w}\in \mathbb{R}^d$ is the weight, and $b\in \mathbb{R}$ is the bias. The parameters $\mathbf{w}$ and $b$ are learnable, just like with a support vector machine. Here, we'll choose $\sigma$ to be the sigmoid activation function

$$\sigma(t) =  \frac{1}{1 + e^{-t}}.$$

It's role is to squash the values of the classifier to binary 0/1 values. A plot of the Sigmoid activation is shown below.

We will see in the next lecture that this classifier is called a *perceptron*, and is the basic building block of a neural network. For now, we simply recall that this is essentially the same kind of linear classifier we encountered in support vector machines. 
"""

# %%
import numpy as np
import matplotlib.pyplot as plt

t = np.arange(-10,10,0.01)
sigma = 1/(1 + np.exp(-t))
plt.figure()
plt.plot(t,sigma)

# %%
"""
We'll need a simple toy data set to test our linear classifier on. We'll construct the dataset in numpy and then convert to torch.
"""

# %%
import numpy as np
import matplotlib.pyplot as plt
import torch

num_pts = 500 #Total number of points
d = 2 #We are in dimension 2
m = int(num_pts/2) #Number in each class
data = np.random.randn(m,2) - [2,2]
data = np.vstack((data,np.random.randn(m,2) + [4,4]))
target = np.hstack((np.zeros((m,)),np.ones((m,)))).astype(int)

#Scatter plot the points colored by class
plt.figure()
plt.scatter(data[:,0],data[:,1],c=target)

# %%
"""
Now let's train our linear classifier (perceptron) on this synthetic data.
"""

# %%
import torch
import torch.optim as optim
import torch.nn.functional as F

#Convert to torch
data_torch = torch.from_numpy(data).float()
target_torch = torch.from_numpy(target)

#Create random Tensors for weight and bias
w = torch.randn(d, requires_grad=True)
b = torch.randn(1, requires_grad=True)

#We now use the Adam optimizer, which is more efficient than plain gradient descent
optimizer = optim.Adam([w,b], lr=1)  #Learning rate

for i in range(500):
    #Set the gradients to zero (in place of a.grad = None, etc.)
    optimizer.zero_grad()

    #Compute the output of our classifier
    output = torch.sigmoid(data_torch@w - b)

    #Compute the loss using operations on Tensors
    loss = torch.mean((output - target_torch)**2)

    #Print iteration and loss
    print('Iteration: ',i,'Loss:', loss.item())

    #Call backward to compute gradients. This call will compute the
    loss.backward()

    #Take a step of gradient descent
    optimizer.step()

# %%
"""
The loss measures roughly how many points are misclassified. Let's plot the points and decision boundary to see how the classifier performed. We need our plot_region function for this.
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
#Create a Python class so it looks like an sklearn classifier
class Perceptron:
    def __init__(self, w, b):
        self.w = w
        self.b = b
        print('w=',w,'b=',b)

    def decision_function(self, x):
        return x@self.w - self.b
    def predict(self, x):
        return (self.decision_function(x)>0).astype(int)

#Plot classifier
clf = Perceptron(w.detach().numpy(),b.detach().numpy())
plot_region(data,target,clf)

# %%
"""
# Exercises

Here are some python exercises to get familiar with using PyTorch for optimization.
"""

# %%
"""
1. Use PyTorch to find $(x,y)$ that minimize the function

$$f(x,y) = \sin(x^4 + 3xy + y^2 + 2) + x^2 + y^2.$$
"""

# %%


# %%
"""
2. Write PyTorch code to approximate a function $f(x)$ by an $n$-th order trigonometric polynomial of the form
$$p(x) = \sum_{k=0}^{n-1} a_k \sin((k+1) x).$$
Use the polynomial approximation code above to start. Try approximating the function $f(x)=x$.
"""

# %%


# %%
"""
3. Repeat question 2 for a trigonometric polynomial of the form
$$p(x) = \sum_{k=0}^{n-1} a_k \sin\left(\frac{2}{3}(k+1) x\right).$$
Does this do a better job approximating $f(x)=x$ near the endpoints?
"""

# %%


# %%
"""
4. Apply the linear classifier above (i.e., the perceptron) to binary classification of MNIST digits.
"""

# %%


# %%
"""
5. (Homework) Modify the linear classifier code above to implement a soft-margin SVM with the soft-plus regularization. How does the result compare to the perceptron. The key difference is that SVM is maxmizing the margin, while the perceptron code above does not consider the margin of the classifier (though gradient descent has some implicit bias towards good margins). 
"""

# %%



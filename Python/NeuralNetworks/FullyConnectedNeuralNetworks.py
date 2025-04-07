# %%
"""
# Fully Connected Neural Networks

This notebook gives an introduction to training fully connected neural networks in PyTorch. Turn on the GPU in Edit -> Notebook Settings.
"""

# %%
"""
## Function approximation

We first consider approximating a one dimensional function with a neural network.
"""

# %%
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
plt.ion(); 

#The function to learn
a = [5,4,3,2]
k = [3,2*3.14159,10*2.718,15*3.14159]
def myfunc(x):
    y = a[0]*torch.sin(k[0]*x)
    for i in range(1,len(a)):
        y += a[i]*torch.sin(k[i]*x)
        y += a[i]*torch.cos(k[i]*x)
    return y

#Training data on [0,1] and apply myfunc
data = torch.arange(0,1,0.001).unsqueeze(1)  #Unsqueeze makes it 100x1 instead of 1D length 100
target = myfunc(data)

#Plot the function
fig = plt.figure()
plt.plot(data,target,label='Target Function') 

# %%
"""
We now construct our one hidden layer neural network in PyTorch.
"""

# %%
class Net(nn.Module):
    def __init__(self, num_hidden=1000):
        super(Net, self).__init__()
        self.n = num_hidden
        self.fc1 = nn.Linear(1,num_hidden)
        self.fc2 = nn.Linear(num_hidden,1)

    def forward(self, x):
        x = F.sigmoid(self.fc1(x))
        return self.fc2(x)

# %%
"""
The code below trains the network and plots the result. The flag `cuda` controls whether to use the GPU. Notice the data and model must be sent to the GPU, and pulled back to the cpu for plotting and printing. To use the GPU in Colab, go to Edit -> Notebook Settings, and enable the GPU (you'll have to restart the notebook).
"""

# %%
#GPU
cuda = True
use_cuda = cuda and torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
print('Using ',device)

T = 30000 #Number of training iterstions
model = Net(num_hidden=10**3).to(device)
optimizer = optim.Adam(model.parameters(), lr=0.01)  #Learning rate
#optimizer = optim.SGD(model.parameters(), lr=0.001)  #Regular gradient descent (very slow)
data,target = data.to(device), target.to(device)

#Training 
model.train()  #Put into training mode
for i in range(T):

    #Zero the gradients
    optimizer.zero_grad()

    #Pass data through model and loss (gradients get accumulated in the optimizer)
    output = model(data)
    loss = torch.mean((output-target)**2)

    #Back propagation to compute all gradients, and an optimizer step
    loss.backward()
    optimizer.step()

    #Print Loss only every 1000 iterations
    if i % 1000 == 0:
        print('Iteration: %d, Loss: %f'%(i,loss.item()))

#Plot the function and neural network
model.eval()
fig = plt.figure()
plt.plot(data.cpu(),target.cpu(),label='Target Function') 
plt.plot(data.cpu(),output.detach().cpu(),label='Neural Network') 
plt.legend()


# %%
"""
## Toy classification problem

We now consider a simple toy classification problems in two dimensions. 
"""

# %%
import torch
import torch.nn as nn
import torch.nn.functional as F

class ClassifyNet(nn.Module):
    def __init__(self, num_in, num_hidden, num_out):
        super(ClassifyNet, self).__init__()
        self.fc1 = nn.Linear(num_in,num_hidden)
        self.fc2 = nn.Linear(num_hidden,num_out)

    def forward(self, x):
        return F.log_softmax(self.fc2(F.relu(self.fc1(x))), dim=1)

# %%
"""
We now train the model for classification and plot the resulting decision boundary. Here, we do not use the GPU since the data sets are small and training is very quick. We first define our usual plot_region function.
"""

# %%
def plot_region(X,L,clf,alpha=0.5,cmap='Paired',cp=np.array([0.5,2.5,6.5,8.5,10.5,11.5]),markers = ['o','s','D','^','v','p'],vmin=0,vmax=12,markersize=75,linewidths=1.25,markerlinewidths=1,res=0.01,train_pts=None):

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


    X,Y = np.mgrid[xmin:xmax:0.01,ymin:ymax:0.01]
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
import torch.optim as optim
from sklearn import datasets
import matplotlib.pyplot as plt
import numpy as np

#Create a Python class so it looks like an sklearn classifier
class NetWrapper:
    def __init__(self, model):
        self.model = model

    def predict_proba(self, x):
        with torch.no_grad():
            x = torch.from_numpy(x).float()
            p = self.model(x).numpy()
        return p

    def predict(self, x):
        return np.argmax(self.predict_proba(x),axis=1)

#Data set
n = 100
#X,L = datasets.make_blobs(n_samples=n, cluster_std=[1,1.5], centers=2, random_state=1)
X,L = datasets.make_moons(n_samples=n,noise=0.1,random_state=4)
#X,L = datasets.make_circles(n_samples=n,noise=0.1,random_state=4,factor=0.5)

#Setup model and optimizer
num_hidden = 64 #Number of hidden nodes in the one hidden layer
model = ClassifyNet(2,num_hidden,2)
optimizer = optim.Adam(model.parameters(), lr=0.01)  #Learning rates

#Data to torch
data = torch.from_numpy(X).float()
target = torch.from_numpy(L).long()

#Train for 1000 epochs
model.train()
for i in range(1000):
    optimizer.zero_grad()
    loss = F.nll_loss(model(data), target)
    loss.backward()
    optimizer.step()

print('Final Loss=%f'%loss.item())
model.eval()
plot_region(X,L,NetWrapper(model))

# %%
"""
##Application to MNIST digit classification

Our last example is to the classification of MNIST digits. While Torch offers access to MNIST and other datasets, we'll use GraphLearning for now.

"""

# %%
#pip install -q graphlearning

# %%
"""
Let's first load MNIST and display some images.
"""

# %%
import graphlearning as gl

#Load MNIST data
x,y = gl.datasets.load('mnist')

#Display images
gl.utils.image_grid(x,n_rows=16,n_cols=16)
print(x.shape)

# %%
"""
We use a two layer neural network with 64 hidden nodes, and the Adam Optimizer and SGD with batch size 480. You can play around with these parameters and see how the training is affected. We also use the full 60000 training set, but you can experiment with using less training data.
"""

# %%
num_hidden = 64
batch_size = 480

#GPU
cuda = True
use_cuda = cuda and torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

#Training data (select at random from first 600000)
train_size = 60000
train_ind = np.random.permutation(60000)[:train_size]

#Convert data to torch and device
data_train = torch.from_numpy(x[train_ind,:]).float().to(device)
target_train = torch.from_numpy(y[train_ind]).long().to(device)
data_test = torch.from_numpy(x[60000:,:]).float().to(device)
target_test = torch.from_numpy(y[60000:]).long().to(device)

#Setup model and optimizer
model = ClassifyNet(784,num_hidden,10).to(device)
optimizer = optim.Adam(model.parameters(), lr=0.01)  #Learning rates

#Training 
print('Iteration,Testing Accuracy,Training Accuracy')
for i in range(20):

    #Model evaluation
    model.eval()
    with torch.no_grad():
        pred = torch.argmax(model(data_test),axis=1)
        test_accuracy = torch.sum(pred == target_test)/len(pred)
        pred = torch.argmax(model(data_train),axis=1)
        train_accuracy = torch.sum(pred == target_train)/len(pred)
        print(i,test_accuracy.item()*100,train_accuracy.item()*100)

    #Training mode, run data through neural network in mini-batches (SGD)
    model.train()  
    for j in range(0,len(target_train),batch_size):
        optimizer.zero_grad()
        loss = F.nll_loss(model(data_train[j:j+batch_size,:]), target_train[j:j+batch_size])
        loss.backward()
        optimizer.step()

# %%
"""
In order to try to gain some insight into how the network performs, let's plot the weight matrices in the hidden layers. Unfortunately this is not so helpful here.
"""

# %%
I = model.fc1.weight.detach().cpu().numpy()
I = I/np.max(np.absolute(I))/2 + 0.5
img = gl.utils.image_grid(I,n_rows=8,n_cols=8)

# %%
"""
##Exercises
1. Try reducing the train size and see if you can get the network to overfit (which means the training accuracy is much larger than the testing accuracy).
2. Try one of the alternative losses to softmax+negative log likelihood discussed in the lecture slides. How does it compare?
3. Try changing the number of hidden nodes, and the number of layers in the network. How is the accuracy affected?
4. Pick a new classification dataset publicly available online. For example, you can browse [Kaggle](https://www.kaggle.com/) for general data science datasets, [Torch Datasets](https://pytorch.org/vision/stable/datasets.html) for image classification problems, [sklearn datasets](https://scikit-learn.org/stable/datasets.html), or [GraphLearning](https://jwcalder.github.io/GraphLearning/datasets.html#graphlearning.datasets.load). Train a neural network classifier on your new dataset. The code below will get you started with graphlearning.
"""

# %%
import graphlearning as gl

data,labels = gl.datasets.load('signmnist')
gl.utils.image_grid(data)

data,labels = gl.datasets.load('fashionmnist')
gl.utils.image_grid(data)


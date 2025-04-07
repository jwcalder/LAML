# %%
"""
#Stochastic gradient descent

This notebook compares explores stochastic gradient descent for synthetic and real data.
"""

# %%
"""
## Quadratic functions

We first start with the minimization of a quadratic function with SGD. Below we define a quadratic function as the sum of 16 quadratic functions with different linear terms. SGD selects one of these pieces to perform gradient descent on at each iteration. We also define an SGD function to perform SGD for $T$ iterations with time step $\alpha$.
"""

# %%
def f(x,y):
    return (x**2 + y**2)/2

def grad(x,y):
    p = np.random.randint(16)
    theta = 2*np.pi*p/16
    return x + np.cos(theta), y + np.sin(theta)

def sgd(x0,y0,alpha,T):

    x,y = [x0],[y0]
    for i in range(T):
        g = grad(x[-1],y[-1])
        x += [x[-1]-alpha*g[0]]
        y += [y[-1]-alpha*g[1]]

    return np.array(x),np.array(y)

# %%
"""
Let's now test our SGD algorithm and plot the results.
"""

# %%
import matplotlib.pyplot as plt
import numpy as np
plt.ion()

#Plot contours
plt.figure()
x = np.arange(-1, 1, 0.01)
y = np.arange(-1, 1, 0.01)
X, Y = np.meshgrid(x, y)
Z = f(X,Y)
plt.contour(X,Y,Z, np.arange(0,1.1,0.1)**2/2.1, colors='black',linestyles='dashed')

#Run SGD and plot path
alpha = 0.1
T = int(10/alpha)
x,y = sgd(np.sqrt(0.5),np.sqrt(0.5),alpha,T)
plt.plot(x,y,'C0-')

# %%
"""
Try different values for the time step $\alpha$. How does the path taken by SGD change?
"""

# %%
"""
## MNIST Data

We now turn to experiments with real data.
"""

# %%
#pip install -q graphlearning

# %%
"""
Let's load MNIST and display some images.
"""

# %%
import graphlearning as gl

#Load MNIST data
x,y = gl.datasets.load('MNIST',metric='raw')

#Display images
gl.utils.image_grid(x,n_rows=16,n_cols=16)

# %%
"""
We use a simple 3 layer fully connected network
"""

# %%
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

# 3 layer neural network with a ReLU activation function and log_softmax
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(784,48)
        self.fc2 = nn.Linear(48,24)
        self.fc3 = nn.Linear(24,10)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        output = F.log_softmax(x, dim=1)
        return output


# %%
"""
We now train the neural network with full-batch gradient descent and stochastic gradient descent.


"""

# %%
import time

#Training and testing data, converted to torch
train_size = 60000
batch_size = 64
data = torch.from_numpy(x[:train_size,:]).float()
target = torch.from_numpy(y[:train_size]).long()
data_test = torch.from_numpy(x[train_size:,:]).float()
target_test = torch.from_numpy(y[train_size:]).long()

#Setup model
model = Net()     # model that will be trained with full-batch gradient descent
model_SGD = Net() # model that will be trained with SGD
optimizer = optim.Adadelta(model.parameters(), lr=1)
optimizer_SGD = optim.Adadelta(model_SGD.parameters(), lr=1)

#Record loss
loss_iter = []
loss_SGD_iter = []

#Training epochs
num_epochs = 20
for i in range(num_epochs):

    #Accuracy
    model.eval()
    model_SGD.eval()
    with torch.no_grad():
        loss = F.nll_loss(model(data), target)
        loss_iter.append(loss.item())
        loss_SGD = F.nll_loss(model_SGD(data), target)
        loss_SGD_iter.append(loss_SGD.item())

        test_pred = torch.argmax(model(data_test),axis=1)
        accuracy = torch.mean((test_pred == target_test).float())
        test_pred = torch.argmax(model_SGD(data_test),axis=1)
        accuracy_SGD = torch.mean((test_pred == target_test).float())
        print('Epoch:%d, Loss:%f, Accuracy:%.2f, Loss_SGD:%f, Accuracy SGD=%.2f'%(i,loss.item(),accuracy*100,loss_SGD.item(),accuracy_SGD*100))


    #Full batch training
    start_time = time.time()
    model.train()
    optimizer.zero_grad()
    loss = F.nll_loss(model(data), target)
    loss.backward()
    optimizer.step()
    print("Full Batch: --- %s seconds ---" % (time.time() - start_time))

    #SGD training
    start_time = time.time()
    model_SGD.train()
    #Loop over minibatches
    for j in range(0,train_size,batch_size):
        data_minibatch = data[j:j+batch_size,:]
        target_minibatch = target[j:j+batch_size]
        optimizer_SGD.zero_grad()
        loss_SGD = F.nll_loss(model_SGD(data_minibatch), target_minibatch)
        loss_SGD.backward()
        optimizer_SGD.step()
    print("SGD Batch: --- %s seconds ---" % (time.time() - start_time))

plt.figure()
plt.plot(loss_iter,label='Full-batch')
plt.plot(loss_SGD_iter,label='Minibatch SGD')
plt.ylabel('Loss')
plt.xlabel('Training Epochs')
plt.legend()

# %%
"""
We can see SGD converges far more quickly than full batch gradient descent, giving very good accuracy after only one epoch, while full batch gradient descent does not give good results even after 20 epochs.
"""

# %%
"""
## Exercises
1. Try other data sets, like FashionMNIST.
2. Try playing around with the batch size.
"""

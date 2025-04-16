# %%
"""
# Modified Loss Homework
"""

# %%
pip install -q graphlearning

# %%
import graphlearning as gl
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch.optim as optim

def new_loss(x,y,p=2):
    a = torch.arange(x.shape[0])
    norms = torch.linalg.norm(x,dim=1,ord=p)
    loss = -torch.mean(x[a,y]/norms)
    return loss

class ClassifyNet(nn.Module):
    def __init__(self, num_in, num_hidden, num_out):
        super(ClassifyNet, self).__init__()
        self.fc1 = nn.Linear(num_in,num_hidden)
        self.fc2 = nn.Linear(num_hidden,num_out)

    def forward(self, x):
        return self.fc2(F.relu(self.fc1(x)))

#Load MNIST data
x,y = gl.datasets.load('mnist')

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
        loss = new_loss(model(data_train[j:j+batch_size,:]), target_train[j:j+batch_size],p=5)
        loss.backward()
        optimizer.step()
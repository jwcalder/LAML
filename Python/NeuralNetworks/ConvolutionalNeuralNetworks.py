# %%
"""
# Convolutional Neural Networks

This notebook gives an introduction to training convolutional neural networks in PyTorch to classify MNIST digits. Turn on the GPU in Edit -> Notebook Settings.
"""

# %%
"""
Below we construct a small convolutional neural network with 2 convolutional layers with 32 and 64 channels. Between each layer is a 2x2 max pool. After the two convolutions and max poolings, the 28x28 MNIST digits are mapped to 64 5x5 images. The 64*5*5=1600 flattened features are then fed into a one hidden layer fully connected neural network with 1024 hidden neurons. 
"""

# %%
import torch
import torch.nn as nn
import torch.nn.functional as F

class CNN(nn.Module):
    def __init__(self,w=(32,64)):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, w[0], 3, 1)
        self.conv2 = nn.Conv2d(w[0], w[1], 3, 1)
        self.fc1 = nn.Linear(w[1]*5*5, 1024)
        self.fc2 = nn.Linear(1024, 10)

    def forward(self, x):
        # images start out 28x28
        x = self.conv1(x)  #26x26 images
        x = F.relu(x)
        x = F.max_pool2d(x, 2)  #13x13 images
        x = self.conv2(x)  #11x11 images
        x = F.relu(x)
        x = F.max_pool2d(x, 2)  #5x5 images
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output

# %%
"""
Below we define functions to implement 1 training epoch using stochastic gradient descent and compute the test accuracy. These functions make use of PyTorch data loaders, which are convenient ways to load data sets and access minibatches for training.
"""

# %%
def train(model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % 10 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))


def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

# %%
"""
The code below trains the convolutional neural network and plots the result. The flag `cuda` controls whether to use the GPU. Notice the data and model must be sent to the GPU, and pulled back to the cpu for plotting and printing. To use the GPU in Colab, go to Edit -> Notebook Settings, and enable the GPU (you'll have to restart the notebook).

Note below we use the Adadelta optimizer instead of the Adam optimizer, which gives better results in this setting. 
"""

# %%
import torch.optim as optim
from torchvision import datasets, transforms

#Training settings
cuda = True   #Use GPU acceleration (Edit->Notebook Settings and enable GPU)
batch_size = 64
test_batch_size = 1000
learning_rate = 1.0  
epochs = 20

#GPU
use_cuda = cuda and torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

#Train and Test Data Loader Setup
train_kwargs = {'batch_size': batch_size}
test_kwargs = {'batch_size': test_batch_size}

if use_cuda:
    cuda_kwargs = {'num_workers': 1,'pin_memory': True,'shuffle': True}
    train_kwargs.update(cuda_kwargs)
    test_kwargs.update(cuda_kwargs)

transform = transforms.Compose([transforms.ToTensor()])
dataset1 = datasets.MNIST('./data', train=True, download=True,transform=transform)
dataset2 = datasets.MNIST('./data', train=False, transform=transform)
train_loader = torch.utils.data.DataLoader(dataset1,**train_kwargs)
test_loader = torch.utils.data.DataLoader(dataset2, **test_kwargs)

#Set up model, optimizer, and scheduler
model = CNN().to(device)
optimizer = optim.Adadelta(model.parameters(), lr=learning_rate)

for epoch in range(1, epochs + 1):
    train(model, device, train_loader, optimizer, epoch)
    test(model, device, test_loader)

torch.save(model.state_dict(), "mnist_cnn.pt")

# %%
"""
We now consider transfer learning from the MNIST to FashionMNIST data sets. We use the CNN trained on MNIST, but we freeze the convolutional layers and only train the fully connected layers.
"""

# %%
#Training settings
cuda = True   #Use GPU acceleration (Edit->Notebook Settings and enable GPU)
batch_size = 64
test_batch_size = 1000
learning_rate = 1.0  #Learning rate
gamma = 1.0     #Scheduler learning rate step
epochs = 20

#GPU
use_cuda = cuda and torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
print(device)

#Train and Test Data Loader Setup
train_kwargs = {'batch_size': batch_size}
test_kwargs = {'batch_size': test_batch_size}

if use_cuda:
    cuda_kwargs = {'num_workers': 1,'pin_memory': True,'shuffle': True}
    train_kwargs.update(cuda_kwargs)
    test_kwargs.update(cuda_kwargs)

transform = transforms.Compose([transforms.ToTensor()])
dataset1 = datasets.FashionMNIST('./data', train=True, download=True,transform=transform)
dataset2 = datasets.FashionMNIST('./data', train=False, transform=transform)
train_loader = torch.utils.data.DataLoader(dataset1,**train_kwargs)
test_loader = torch.utils.data.DataLoader(dataset2, **test_kwargs)

#Load model trained on MNIST
transfer_model = CNN()
transfer_model.load_state_dict(torch.load("mnist_cnn.pt"))

#Freeze all parameters in the model
for p in transfer_model.parameters():
    p.requires_grad = False

#Replace the fully connected layers with new ones, 
#which will have requires_grad=True by default
transfer_model.fc1 = nn.Linear(1600, 1024)
transfer_model.fc2 = nn.Linear(1024, 10)

#Put model on the device
transfer_model.to(device)

#Optimization
optimizer = optim.Adadelta(transfer_model.parameters(), lr=learning_rate)
for epoch in range(1, epochs + 1):
    train(transfer_model, device, train_loader, optimizer, epoch)
    test(transfer_model, device, test_loader)

# %%
"""
##Exercises
1. Try using the Adam optimizer. Can you get similar results?
2. Try reducing the train size and see if you can get the network to overfit (which means the training accuracy is much larger than the testing accuracy).
3. Try another dataset from Torch, like FashionMNIST or Cifar10. You'll have to change the network for Cifar-10, since the images are 32x32x3 color images.
"""


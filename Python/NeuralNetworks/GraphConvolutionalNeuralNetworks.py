# %%
"""
# Graph Convolutional Neural Networks

This notebook is a brief introduction to graph convolutional neural networks (GCN) in PyTorch.
"""

# %%
#pip install -q graphlearning

# %%
"""
We define a simple GCN, as well as its MLP counterpart that does not use graph information.
"""

# %%
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import graphlearning as gl
from scipy import sparse

def csr_to_torch(W):

    W = W.tocoo()
    values = W.data
    indices = np.vstack((W.row, W.col))

    i = torch.LongTensor(indices)
    v = torch.FloatTensor(values)
    shape = W.shape

    return torch.sparse_coo_tensor(i, v, shape)

class GCN(nn.Module):
    def __init__(self, G, num_hidden=30):
        super(GCN, self).__init__()
        num_in = G.features.shape[1]
        num_classes = len(np.unique(G.labels))
        self.fc1 = nn.Linear(num_in,num_hidden)
        self.fc2 = nn.Linear(num_hidden,num_hidden)
        self.fc3 = nn.Linear(num_hidden,num_classes)

        #Renormalization trick
        W = G.weight_matrix
        H = gl.graph(W + sparse.eye(G.num_nodes))
        D = H.degree_matrix(p=-1)
        A = D*H.weight_matrix
        self.A = csr_to_torch(A).to(device)

    def forward(self, x):
        x = F.relu(self.A@self.fc1(x))
        x = F.relu(self.A@self.fc2(x))
        x = F.log_softmax(self.fc3(x),dim=1)
        return x

    def encode(self, x):
        x = F.relu(self.A@self.fc1(x))
        x = self.A@self.fc2(x)
        return x

class MLP(nn.Module):
    def __init__(self, G, num_hidden=30):
        super(MLP, self).__init__()
        num_in = G.features.shape[1]
        num_classes = len(np.unique(G.labels))
        self.fc1 = nn.Linear(num_in,num_hidden)
        self.fc2 = nn.Linear(num_hidden,num_hidden)
        self.fc3 = nn.Linear(num_hidden,num_classes)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return F.log_softmax(self.fc3(x),dim=1)

# %%
"""
We now run a semi-supervised learning trial on the PubMed data set with GCN compared to MLP.
"""

# %%
import numpy as np
import graphlearning as gl
import torch.optim as optim
from scipy import sparse

#Load Graph
G = gl.datasets.load_graph('pubmed')
W = G.weight_matrix
labels = G.labels
X = G.features
m,n = X.shape

np.random.seed(1) #For reproducibility
train_ind = gl.trainsets.generate(labels, rate=0.003)
train_labels = labels[train_ind]
train_mask = np.zeros(m,dtype=bool)
train_mask[train_ind]=True
test_mask = ~train_mask
test_ind = np.arange(m)[test_mask].astype(int)

#GPU
gpu = True
if gpu:
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
else:
    device = torch.device("cpu")

#Setup model and initial plot
model_GCN = GCN(G,num_hidden=100).to(device)
model_MLP = MLP(G,num_hidden=100).to(device)
optimizer_GCN = optim.Adam(model_GCN.parameters(), lr=0.01)  #Learning rates
optimizer_MLP = optim.Adam(model_MLP.parameters(), lr=0.01)  #Learning rates

#Convert data to torch and device
data = torch.from_numpy(X).float().to(device)
target = torch.from_numpy(labels).long().to(device)
train_set = torch.from_numpy(train_ind).long().to(device)
test_set = torch.from_numpy(test_ind).long().to(device)

for t in range(100):

    optimizer_GCN.zero_grad()
    output_GCN = model_GCN(data)
    loss = F.nll_loss(output_GCN[train_set,:], target[train_set])
    loss.backward()
    optimizer_GCN.step()

    optimizer_MLP.zero_grad()
    output_MLP = model_MLP(data)
    loss = F.nll_loss(output_MLP[train_set,:], target[train_set])
    loss.backward()
    optimizer_MLP.step()

    with torch.no_grad():
        pred = torch.argmax(output_GCN,axis=1)
        GCN_accuracy = 100*torch.sum(pred[test_set] == target[test_set])/len(test_set)
        pred = torch.argmax(output_MLP,axis=1)
        MLP_accuracy = 100*torch.sum(pred[test_set] == target[test_set])/len(test_set)
        print('%d,%f,%f'%(t,GCN_accuracy,MLP_accuracy),flush=True)

# %%
"""
Let's now look at GCN node embeddings for the karate and political books data sets.
"""

# %%
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
plt.ion()

#Load Graph and set features to identity matrix
graph = 'karate' #or 'polbooks'
G = gl.datasets.load_graph(graph)
W = G.weight_matrix
if graph == 'polbooks':
    labels = (G.fiedler_vector() > 0).astype(int)
else:
    labels = G.labels
m = G.num_nodes
n = m
X = np.eye(m)
G.features = X

np.random.seed(1) #For reproducibility
train_ind = gl.trainsets.generate(labels, rate=1)
train_labels = labels[train_ind]
train_mask = np.zeros(m,dtype=bool)
train_mask[train_ind]=True
test_mask = ~train_mask
test_ind = np.arange(m)[test_mask].astype(int)

#Setup model and initial plot
device = torch.device("cpu")
model = GCN(G,num_hidden=50)
optimizer = optim.Adam(model.parameters(), lr=0.01)  #Learning rates

#Convert data to torch and device
data = torch.from_numpy(X).float()
target = torch.from_numpy(labels).long()
train_set = torch.from_numpy(train_ind).long()
test_set = torch.from_numpy(test_ind).long()

for t in range(100):

    optimizer.zero_grad()
    output = model(data)
    loss = F.nll_loss(output[train_set,:], target[train_set])
    loss.backward()
    optimizer.step()

    with torch.no_grad():
        pred = torch.argmax(output,axis=1)
        test_accuracy = 100*torch.sum(pred[test_set] == target[test_set])/len(test_set)
        print('%d,%f'%(t,test_accuracy),flush=True)

with torch.no_grad():
    Y = model.encode(data).numpy()
    pca = PCA(n_components=2)
    Z = pca.fit_transform(Y)
    G.draw(X=Z,c=G.labels,linewidth=0.5)
    plt.scatter(Z[train_ind,0],Z[train_ind,1],c='red',marker='o',s=10,zorder=100)

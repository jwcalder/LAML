# %%
"""
# 2-layer neural net in Numpy
"""

# %%
#pip install -q graphlearning

# %%
import graphlearning as gl
import numpy as np

#Softmax on batches
def softmax(x):
    p = np.exp(x)
    norms = np.linalg.norm(p,axis=1,ord=1)
    return p/norms[:,None]

#Softmax followed by negative log likelihood loss
#Also returns gradient v
def softmax_nll(x,y):
    k = np.arange(x.shape[0])
    v = softmax(x)
    v[k,y] -= 1
    mx = np.max(x,axis=1)
    loss = -x[k,y] + np.log(np.linalg.norm(np.exp(x - mx[:,None]),axis=1,ord=1))
    loss = np.mean(loss)
    return v,loss

#Basic neural net class in Numpy
class NumpyNet():
    def __init__(self, num_in, num_hidden, num_out):
        self.W1 = np.random.randn(num_hidden,num_in)
        self.b1 = np.random.randn(num_hidden,1)
        self.W2 = np.random.randn(num_out,num_hidden)
        self.b2 = np.random.randn(num_out,1)

    def __call__(self, x):
        return self.forward(x)

    #Forward propagation
    def forward(self, x):
        self.x = x.T
        self.p1 = self.W1@self.x + self.b1
        self.z1 = np.maximum(self.p1,0) #ReLU activation
        self.p2 = self.W2@self.z1+self.b2
        self.z2 = self.p2
        return self.z2.T

    #Zero gradients
    def zero_grad(self):
        self.grad_W1 = np.zeros_like(self.W1)
        self.grad_b1 = np.zeros_like(self.b1)
        self.grad_W2 = np.zeros_like(self.W2)
        self.grad_b2 = np.zeros_like(self.b2)

    #Back propagation
    def backward(self,v):
        self.zero_grad()
        bs,d = v.shape

        #Loop over batch to avoid tensors
        for i in range(bs):
            v2 = v[[i],:].T
            z1 = self.z1[:,[i]]
            p1 = self.p1[:,[i]]
            x = self.x[:,[i]]

            #Layer 2
            self.grad_b2 += v2
            self.grad_W2 += v2@z1.T

            #Back propagate
            v1 = self.W2.T@v2

            #Layer 1
            S1 = np.diag(p1.flatten() >= 0)
            self.grad_b1 += S1@v1
            self.grad_W1 += S1@self.W2.T@v2@x.T

        self.grad_W1 /= bs
        self.grad_b1 /= bs
        self.grad_W2 /= bs
        self.grad_b2 /= bs

    #Optimization step
    def step(self,lr):
        self.W1 -= lr*self.grad_W1
        self.W2 -= lr*self.grad_W2
        self.b1 -= lr*self.grad_b1
        self.b2 -= lr*self.grad_b2

#Load MNIST data
x,y = gl.datasets.load('mnist')

num_hidden = 64
batch_size = 128

#Training data (select at random from first 600000)
train_size = 60000
train_ind = np.random.permutation(60000)[:train_size]

#Convert data to torch and device
data_train = x[train_ind,:]
target_train = y[train_ind]
data_test = x[60000:,:]
target_test = y[60000:]

#Setup model and optimizer
model = NumpyNet(784,num_hidden,10)
lr = 1

#Training
print('Iteration,Testing Accuracy,Training Accuracy')
for i in range(100):

    #Test model
    pred = np.argmax(model(data_test),axis=1)
    test_accuracy = np.mean(pred == target_test)
    pred = np.argmax(model(data_train),axis=1)
    train_accuracy = np.mean(pred == target_train)
    print(i,test_accuracy*100,train_accuracy*100)

    #Training mode, run data through neural network in mini-batches (SGD)
    for j in range(0,len(target_train),batch_size):
        xb,yb = data_train[j:j+batch_size,:], target_train[j:j+batch_size]
        v,loss = softmax_nll(model(xb),yb)
        model.backward(v)
        model.step(lr)

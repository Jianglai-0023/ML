import numpy as np
from matplotlib import pyplot as plt

"""
    STEP 1: Define the base class of the neural network layer.
"""

# Base class of the nnet layer
class Layer():
    def __init__(self):
        pass
    
    # Forward propagation function: compute the output by input x
    def forward(self, x):
        raise NotImplementedError
    
    # Backward propagation function: compute dE/dW and dE/dx by node_grad(dE/dy)
    def backward(self, node_grad):
        raise NotImplementedError
    
    # Update function: update the weights by gradients
    def update(self, learning_rate):
        raise NotImplementedError

"""
    STEP 2: Implement the activation functions.
"""

class Sigmoid(Layer):    
    def forward(self, x):
        self.y = 1 / (1 + np.exp(-x))
        return self.y
    
    def backward(self, node_grad):
        return node_grad * (self.y * (1 - self.y))
    
    def update(self, learning_rate):
        pass

class Relu():        
    def forward(self, x):
        self.y=(x>0) * x
        return self.y

    
    def backward(self, node_grad):
        return (self.y>0)*node_grad

       
    
    def update(self, learning_rate):
        pass

class Softmax_Cross_Entropy():
    def forward(self, x):
        bottom=np.sum(np.exp(x))
        # print("##")
        # print(self.bottom)
        self.y=np.exp(x)/bottom
        # print(self.y)
        #[[n1,n2]]
        return self.y
    
    def backward(self, label):
        # print("backward")
        # print(label)# [1,0]
        # print(self.y-label)# [[x,y]]
        return self.y-label

    def update(self, learning_rate):
        pass

"""
    STEP 3: Implement the linear layer.
 """

class Linear(Layer):    
    def __init__(self, size_in, size_out, with_bias):
        self.size_in = size_in
        self.size_out = size_out
        self.with_bias = with_bias
        self.W = self.initialize_weight()
        if with_bias:
            self.b = np.zeros(size_out)
    #定义参数 
    
    def initialize_weight(self):
        epsilon = np.sqrt(2.0 / (self.size_in + self.size_out))
        return epsilon * (np.random.rand(self.size_in, self.size_out) * 2 - 1)
    
    def forward(self, x):
    
       self.x=x.reshape(1,self.size_in)#行向量
       return (x@self.W) + self.b 
        
    
    def backward(self, node_grad):
        node_grad=np.reshape(node_grad,(self.size_out,1))
        self.grandiant_w=(self.x.T@node_grad.T)
        self.grandiant_b=node_grad.T
        return (node_grad.T)@(self.W.T)

    
    def update(self, learning_rate):
        self.W=self.W - self.grandiant_w*learning_rate
        self.b=self.b - self.grandiant_b*learning_rate

"""
    STEP 4: Combine all parts into the MLP.
"""

class MLP():    
    def __init__(self, layer_size, with_bias=True, activation="sigmoid", learning_rate=1):
        assert len(layer_size) >= 2
        self.layer_size = layer_size
        self.with_bias = with_bias
        if activation == "sigmoid":
            self.activation = Sigmoid
        elif activation == "relu":
            self.activation = Relu
        else:
            raise Exception("activation not implemented")
        self.learning_rate = learning_rate
        self.build_model()
        
    def build_model(self):
        self.layers = []
        
        size_in = self.layer_size[0]
        for hu in self.layer_size[1:-1]:#第一个到倒数第一个右开区间
            self.layers.append(Linear(size_in, hu, self.with_bias))
            self.layers.append(self.activation())
            size_in = hu
            
        self.layers.append(Linear(size_in, self.layer_size[-1], self.with_bias))
        self.layers.append(Softmax_Cross_Entropy())
        
    def forward(self, x):
        for layer in self.layers:
            x = layer.forward(x)
        return x
    
    def backward(self, label):
        node_grad = label
        for layer in reversed(self.layers):# 列表的倒序遍历a
            node_grad = layer.backward(node_grad)
            
    def update(self, learning_rate):
        for layer in self.layers:
            layer.update(learning_rate)
            
    def train(self, x, label):#x输入 label输出
        y = self.forward(x)
        self.backward(label)
        self.update(self.learning_rate)
    
    def predict(self, x):
        for layer in self.layers:
            x = layer.forward(x)
        return np.argmax(x)
    
    def loss(self, x, label):# 损失函数
        y = self.forward(x)
        return -np.log(y) @ label #矩阵乘法


"""
    STEP 5: Test
"""

X = np.array([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
])
Y = np.array([
    [0, 1],
    [1, 0],
    [1, 0],
    [0, 1]
])

np.random.seed(1007)
EPOCH = 10000
N = X.shape[0]

mlp = MLP([2, 4, 2], learning_rate=0.1, activation="relu")

loss = np.zeros(EPOCH)
for epoch in range(EPOCH):
    for i in range(N):
        mlp.train(X[i], Y[i])
        
    for i in range(N):
        loss[epoch] += mlp.loss(X[i], Y[i])
        
    loss[epoch] /= N
    
plt.figure()
ix = np.arange(EPOCH)
plt.plot(ix, loss)
#plt.show()
plt.savefig("MLP.png")

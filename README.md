# Neural Network
A neural network is a computational model inspired by the way biological neural networks in the human brain function. It is a key component of machine learning and artificial intelligence systems. Neural networks consist of interconnected nodes, also known as neurons or artificial neurons, organized into layers. These layers typically include an input layer, one or more hidden layers, and an output layer.

Here are some key components and concepts associated with neural networks:

1. Neurons (Nodes): Neurons are the basic processing units in a neural network. Each neuron receives input, performs a computation, and produces an output. The output is typically passed to other neurons in the network.
   
   - $`z = \omega a + b`$

     where $`a`$ is the input data, $`z`$ is output, $`\omega`$ is the weight and $`b`$ is the bias. These are the parameter of one neural.

2. Weights($`\omega`$) and Biases($`b`$): Connections between neurons are represented by weights, which determine the strength of the connection. Each connection has an associated weight that is adjusted during the training process. Biases are additional parameters that are added to the weighted sum of inputs to help the model learn more complex patterns.

3. Activation Function: Neurons often apply an activation function to the weighted sum of their inputs to introduce non-linearity to the model. Common activation functions include sigmoid, hyperbolic tangent (tanh), and rectified linear unit (ReLU).

   - sigmoid: $`z = s (a) = \frac{1}{1+e^{-a}}`$
   
   - $`\bigtriangledown `$sigmoid: $`z = {s}'(a) = s (a)\cdot (1-s (a))`$
   
   - tanh:    $`z = tanh (a) = \frac{e^{a}-e^{-a}}{e^{a}+e^{-a}}`$

   - $`\bigtriangledown `$tanh: $`z = {tanh}'(a) = 1-tanh (a)^{2}`$
   
   - ReLu:    $`z = r (a) = \left\{\begin{matrix} a\ \ a> 0 \\  0\ \ a\leq  0 \end{matrix}\right.`$

   - $`\bigtriangledown `$ReLu: $`z = {r}'(a) = \left\{\begin{matrix} 1\ \ a> 0 \\  0\ \ a\leq 0 \end{matrix}\right.`$

4. Layers: Neural networks are organized into layers, including an input layer where data is fed into the network, one or more hidden layers where computation occurs, and an output layer where the final result is generated.

   $`z^{l} = \sigma (\omega ^{l}a+b^{l})`$
   
   $`z^{l+1} = \sigma (\omega ^{l+1}z^{l}+b^{l+1})`$
   
   $`z^{l+2} = \sigma (\omega ^{l+2}z^{l+1}+b^{l+2})`$
   
   $`z^{n} = \sigma (\omega ^{n}z^{n-1}+b^{n})`$

5. Feedforward and Backpropagation: In the training phase, neural networks use a process called feedforward to make predictions based on input data, and then backpropagation to adjust the weights and biases in order to minimize the difference between the predicted output and the actual target output.

   - Forward: $`z^{n} = \sigma (\omega ^{n}z^{n-1}+b^{n})`$
   
   - backward: $`\delta = {\sigma}'(z^{n})\circ \frac{\partial L}{\partial a^{n}} = \frac{\partial L}{\partial a^{n-1}}`$

6. Loss Function: The performance of a neural network is measured using a loss function, which calculates the difference between the predicted output and the actual target output. The goal during training is to minimize this loss.

    - MSE_loss: $`loss = \frac{1}{2}(y-\hat{y})^{2}`$
      
    - $`\frac{\partial L}{\partial a^{n}}=\bigtriangledown L=\hat{y}-y`$

7. Deep Learning: Neural networks with more than one hidden layer are referred to as deep neural networks. The use of deep neural networks is known as deep learning, and it has proven effective in solving complex tasks such as image and speech recognition.

Neural networks are versatile and can be applied to various tasks, including image and speech recognition, natural language processing, and playing games. They have become a fundamental technology in the field of artificial intelligence, enabling machines to learn from data and make predictions or decisions without being explicitly programmed for a specific task.

# Neural Network on Different Distribution Data
we create the dataset from sklearn:
```ruby
from sklearn.datasets import make_blobs, make_circles, make_classification, make_moons, make_gaussian_quantiles

samples = 200
datasets = [
    make_blobs(n_samples=samples, centers=2, n_features=2, random_state=1),
    make_blobs(n_samples=samples, centers=2, n_features=2, random_state=6),
    make_moons(n_samples=samples, noise=0.15, random_state=0),
    make_circles(n_samples=samples, noise=0.15, factor=0.3, random_state=0),
    make_gaussian_quantiles(n_samples=samples, n_features=2, n_classes=2, random_state=0),
    make_classification(n_samples=samples, n_features=2, random_state=1, n_redundant=0, n_informative=1, n_clusters_per_class=1),
    make_classification(n_samples=samples, n_features=2, random_state=1, n_redundant=0, n_informative=2, n_clusters_per_class=1),
    make_classification(n_samples=samples, n_features=2, random_state=1, n_redundant=0, n_informative=2),
    make_blobs(n_samples=samples, centers=3, n_features=2, random_state=1),
    make_blobs(n_samples=samples, centers=4, n_features=2, random_state=6),
    make_gaussian_quantiles(n_samples=samples, n_features=2, n_classes=3, random_state=0),
    make_gaussian_quantiles(n_samples=samples, n_features=2, n_classes=4, random_state=0),
]
```
<img src="https://github.com/jaja7749/Neural_Network/blob/main/images/different%20distribution.png" width="720">

First of all, we build NN layer class:
```ruby
class NN(Module):
    def __init__(self, in_size, out_size):
        device = (
            "cuda" if torch.cuda.is_available()
            else "mps" if torch.backends.mps.is_available() 
            else "cpu"
            )
        self.device = device
        self.w = torch.randn((out_size, in_size), device=self.device)
        self.b = torch.randn((len(self.w), 1), device=self.device)
    
    def forward(self, a):
        self.a = a.to(self.device)
        self.z = torch.matmul(self.w, self.a) + self.b
        return self.z
    
    def backward(self, dLdz):
        self.dLdz = dLdz.to(self.device)
        self.dLdw = torch.kron(self.dLdz, self.a.T)
        self.dLdb = self.dLdz
        self.dLda = torch.matmul(self.w.T, self.dLdz)
        return self.dLda
    
    def __repr__(self):
        return "NN"
```

Second, build the activation function:
```ruby
class ReLu(Module):
    def forward(self, a):
        self.a = a
        self.z = torch.where(a>0, a, 0.)
        return self.z
    
    def backward(self, dLdz):
        self.dLdz = dLdz
        self.dLda = torch.multiply(torch.where(self.z>0, 1., 0.), dLdz)
        return self.dLda
    
    def __repr__(self):
        return "ReLu"
    
class Sigmoid(Module):
    def forward(self, a):
        self.a = a
        self.z = 1/(1 + torch.exp(-a))
        return self.z
    
    def backward(self, dLdz):
        self.dLdz = dLdz
        self.dLda = torch.multiply(torch.exp(-self.z)/torch.pow(1 + torch.exp(-self.z), 2), dLdz)
        return self.dLda
    
    def __repr__(self):
        return "Sigmoid"
    
class Tanh(Module):
    def forward(self, a):
        self.a = a
        self.z = torch.tanh(a)
        return self.z
    
    def backward(self, dLdz):
        self.dLdz = dLdz
        self.dLda = torch.multiply(1 - torch.pow(torch.tanh(self.z), 2), dLdz)
        return self.dLda
    
    def __repr__(self):
        return "Tanh"
```

Third, we build the loss function:
```ruby
class MSE_loss(Module):
    def __init__(self):
        None
    
    def __call__(self, y, y_pred):
        self.y = y
        self.y_pred = y_pred
        return 0.5 * torch.pow(y - y_pred, 2)
    
    def backward(self):
        return self.y_pred - self.y
    
    def __repr__(self):
        return "MSE_loss"
```

After that we create a class and let model can have muti layers:
```ruby
class Module:
    def __init__(self):
        self.layers = {}
        
    def __call__(self, a):
        for layer_index in self.layers:
            a = self.layers[layer_index].forward(a)
        return a
            
    def backward(self, dLdz):
        for layer_index in reversed(self.layers):
            dLdz = self.layers[layer_index].backward(dLdz)
        
    def update(self, learning_rate=0.01):
        for layer_index in reversed(self.layers):
            try:
                self.layers[layer_index].w -= learning_rate * self.layers[layer_index].dLdw
                self.layers[layer_index].b -= learning_rate * self.layers[layer_index].dLdb
            except AttributeError:
                None
            
    def model_class(self, classes):
        self.classes = classes
        
    def pred(self, a):
        for layer_index in self.layers:
            a = self.layers[layer_index].forward(a)
        return self.classes[torch.argmax(a)]
    
    def add_module(self, layer_index, module):
        if layer_index in self.layers:
            raise ValueError(f"Module with layer index '{layer_index}' already exists.")
        self.layers[layer_index] = module
        
    def __repr__(self):
        repr_string = f"Module(\n"
        for layer_index in self.layers:
            repr_string += f"  ({layer_index}): {self.layers[layer_index]}\n"
        repr_string += ")"
        return repr_string
    
class Sequential(Module):
    def __init__(self, *X):
        super(Sequential, self).__init__()
        for index, layer in enumerate(X):
            if not isinstance(layer, Module):
                raise ValueError(f"Argument at index {index} is not an instance of Module")

            self.add_module(index, layer)
```

Finally, we set the training process:
```ruby
def train(X, Y, model, loss_fn):
    train_loss = 0
    for i in range(len(X)):
        x, y = torch.atleast_2d(X[i].to(device)).T, torch.atleast_2d(Y[i].to(device)).T

        # Forward
        y_pred = model(x)
        loss = loss_fn(y, y_pred)
        train_loss += loss

        # Backpropagation
        dLdz = loss_fn.backward()
        model.backward(dLdz)
        model.update(learning_rate=0.05)

    train_loss /= len(X)
    return train_loss
```

After all of the training let's check our results:

<img src="https://github.com/jaja7749/Neural_Network/blob/main/images/NN%20result2.png" width="720">

# Brief Introduction to Neural Networks 


## Neural networks, setup, architecture definitions

Neural networks consists of $L$ layers labeled 
$l = 1,2 ...L$. In a given $l$ layer there are $n_l$ cells, whose activation outputs are denoted 
$a_j^{l}$ and given by 
$$
	a_j^{l} = f(z^l_{j}),
$$ 
where $f(z^l_{j})$ is known as the activation function whose argument is given by  
$$
	z^l_{j} = \sum_{k=1}^{n_{l-1}} w_{jk}a^{l-1}_k,
$$

here the weight $w^{l}_{jk}$ connects the output $a^{l-1}_k$ of the $k$-th cell in the previous layer 
to the $j$ cell in the $l$-th layer. 

Matrix and vector notation is used to efficiently represent a neural network. 
Considering layer $l$ its activation outputs are contained in a column vector 
$\mathbf{a}^{l} = [a^l_1 ~ a^l_2~...~a^l_{n_l}]^{T}$, given by 


$$\begin{aligned}
	\mathbf{a}^{l} &= f\left( \mathbf{z}^{l} \right) 
\end{aligned}
$$
where the input is
$$\begin{aligned}
	\mathbf{z}^{l}   &= \mathbf{W}^{l} \mathbf{a}^{l-1},
\end{aligned}$$ 
with the weight matrix 

$$
	\mathbf{W}^{l} = 
    \begin{pmatrix} 
    w_{11}^{l} & w_{12}^{l} & \cdots &w_{1n_{l-1}}^{l} \\ 
	w_{21}^{l} & w_{22}^{l} & \cdots & w_{2{n_{l-1}}}^{l}  \\
	\vdots &\vdots  &\vdots \\
	w_{n_{l}1}^{l} & w_{n_{l}2}^{l} &\cdots & w_{n_{l}{n_{l-1}}}^{l} 
	\end{pmatrix}
$$

**Input and output layers.** The activation vector of the input layer consists of the input vector 
$\mathbf{x}$. For each input there is an observation vector $\mathbf{y}$. The activation vector of the 
output layer is denoted by $\mathbf{\hat{y}}$

**Hidden layers.** Hidden layers are those layers connecting the input layer to the output layer.

**Training a neural network.**  A network is trained by adjusting the elements in 
its weight matrices $\mathbf{W}^{l}$ to yield a minimum value in its cost function $\mathrm{C}$. Given 
a set of inputs and observations $\{\mathbf{x},\mathbf{y}\}$ the cost function can defined as 

$$
    \mathrm{C} = \frac{1}{2}\sum_j(y_j - \hat{y}_j)^2
$$

other cost functions could be used. The typical training/optimization technique is gradient descent.  

**Network optmization.** Given a set of known inputs and known observations 
$\{\mathbf{x},\mathbf{y}\}$ a network is optimized by minimizing its cost function $\mathrm{C}$. 
This is typically achieved by first initializing the weights $\mathbf{W}^{l}$. 
Then given a subset (also known as a*batch}) of the inputs 
$\mathbf{x}$ is used to compute the network outputs $\mathbf{\hat{y}}$. 
This step is known as forward propagation. From the computed network outputs and the known outputs 
the cost function is computed, for instance using the squared error 

$$
    \mathrm{C} = \frac{1}{2}\sum_j(y_j - \hat{y}_j)^2
$$


The typical optimization method, i.e. finding minima in the Cost function, is gradient descent. 

**Gradient descent.** (From wikipedia) Gradient descent is a first-order iterative optimization 
algorithm. To find a local minimum of the cost function using gradient descent, 
one takes steps proportional to the negative of the gradient of the function with respect weights at 
the current point:

$$
	\Delta \mathbf{W}^{l} = - \gamma \nabla_{\mathbf{W}^{l}} \mathrm{C},
$$

where proportionality factor $\gamma$ is a positive constant known as the learning rate of the network. 

**Backpropagation.** Backpropagation is the name that gradient descent receives when used to 
train a neural network. It consists in computing the error, for a given set of 
$\{\mathbf{x},\mathbf{y}\}$, and propagating it through the different hidden layers in the network so that 
$\Delta \mathbf{W}^{l}$ can be computed.

The following equations describe a summary of backpropagation, including matrix notation where $\odot$ and $\otimes$
denote element-wise matrix multiplication and outer-product respectively.


1.**Error in output layer**: 
$$
\begin{aligned}
		\delta^{L}_{j} &= \frac{ \partial C}{\partial a^{L}_{j}}f^{'}\left( {z}^{L}_{j}\right) \rightarrow \mathrm{~matrix ~ form:} \\
	\delta^{L} &= \nabla_{\mathbf{a}}C \odot g^{'}\left( \mathbf{h}^{L} \right) 
	% \delta^{L} &=   'np.multiply( \rm{partial\_cost\_partial\_activation*h\_prime})' 
\end{aligned} 
$$
2.**Error in hidden layer**:
$$
\begin{aligned}
	\delta^{l-1}_{k} &= \sum_{j}w^{l}_{jk}\delta^{l}_{j}f^{'}\left( h^{l-1}_k \right) \rightarrow : \\ 
	\delta^{l-1} &= \left(\left(\mathbf{W}^{l}\right)^{T}\delta^{l} \right) \odot f^{'}\left( \mathbf{h}^{l-1}\right) 
\end{aligned}
$$

3.**Rate of change of cost function with respect bias in the network** 
$$
\begin{aligned}
	\frac{\partial C}{\partial b^{l}_{j} } &=& \delta^{l}_{j} \rightarrow : \\
	\frac{\partial C}{\partial \mathbf{b}^{l} } &=& \delta^{l}
\end{aligned}
$$
4.**Rate of change of cost function with respect weights in the network**: 
$$
\begin{aligned}
	\frac{\partial C}{\partial w^{l}_{jk} } &=& \delta^{l}_{j} a^{l-1}_{k}\rightarrow : \\
	\frac{\partial C}{\partial \mathbf{W}} &=& \delta^{l} \otimes \mathbf{a}^{l-1} 
\end{aligned} 
$$

## Algorithm for backpropagation

``` python
# feedforward pass 
# collect all activations and weighted vectors zs
    activations, zs = feedforward(x)
    y_hat = zs[-1]

# backward pass
# gradient descent for weights matrix connecting second-last to output layers 
    delta = self.cost_derivative(y_hat, y) * sigmoid_prime(zs[-1])
    nabla_b[-1] = delta
    nabla_w[-1] = np.outer(delta, activations[-2])

# gradient descent for weights matrix in all inner layers 
    for l in range(2, self.num_layers):
        z = zs[-l]
        sp = sigmoid_prime(z)
        delta = np.dot(weights[-l+1].transpose(), delta) * sp
        nabla_b[-l] = delta
        nabla_w[-l] = np.outer(delta, activations[-l-1])

    weights = [w - learning_rate*nw
                for w, nw in zip(weights, nabla_w)]
    biases  = [b- self.learning_rate*nb
                for b, nb in zip(biases, nabla_b)]
```
## Advanced neural networks
Some problems are better tackled using specialized versions of neural networks, such as in convolutional 
neural networks (CNN) and recurrent-neural networks (RNN). These two types of neural networks are 
implemented taking advantage of known features in the input data such as in images or in time series. 
The idea behind is simple: exploit known structures in the input data to make the learning and inference 
process of the network more efficient.

## Derivation of backpropagation equations
It is all about the chain rule. The goal is to find the weights that minimize the cost function $C$ of the model. This optimization is implemented numerically 
using gradient descent 

$$
	\mathbf{W}^{l} = \mathbf{W}^{l} - \gamma \nabla_{\mathbf{W}^{l}} \mathrm{C},
$$

To get expressions for the rate of change of the cost function with respect weights consider the following.
Starting with weights connecting cells in the second-to-last layer, indexed with $k$, to cells in the output 
layer (index $j$) we obtain 
$$
    \frac{\partial C}{\partial w^{L-1}_{jk}} =  \frac{\partial C}{\partial a^{L}_j} 
                                               \frac{\partial a^{L}_j}{\partial z^{L}_j}
                                               \frac{\partial z^{L}_j}{\partial w_{jk} },
$$

where 
$$
\begin{aligned}
    z^{L}_j &= \sum_k w_{jk}^{L-1} a^{L-1}_k. 
\end{aligned} 
$$ 
Thus 
$$
    \frac{\partial C}{\partial w^{L-1}_{jk}} =  \frac{\partial C}{\partial a^{L}_j} 
                                               \frac{\partial a^{L}_j}{\partial z^{L}_j}
                                               a^{L-1}_k.
$$

Defining 
$$
    \delta_j^{L} = \frac{\partial C}{\partial a^{L}_j} f'(z^{L}_j)
$$
we get 
$$
    \frac{\partial C}{\partial w^{L-1}_{jk}} =  \delta_j^{L} a^{L-1}_k ,
$$    

which corresponds to the elements of the matrix 
$$
    \frac{\partial C}{\partial \mathbf{W}^{L-1}} = \delta^{L} \otimes \mathbf{a}^{L-1}. 
$$

To get an expression for the rate of change of the cost function with respect to coefficients 
connecting cells in the third-to-last layer (indexed with $l$,) to cells
in the second-to-last layer (with index $k$) consider 
$$
\begin{aligned}
    \frac{\partial C}{\partial w^{L-2}_{kl}} &=& \sum_j \frac{\partial C}{\partial a^{L}_j} 
                                               \frac{\partial a^{L}_j}{\partial z^{L}_j}
                                               \frac{\partial z^{L}_j}{\partial w_{kl}^{L-2} } \\
    \frac{\partial C}{\partial w^{L-2}_{kl}} &=& \sum_j \delta_j^{L}  \frac{\partial z^{L}_j}{\partial w_{kl}^{L-2} }                                         
\end{aligned}  
$$  


where the sum running over $j$ is due to the fact that the $ w^{L-2}_{kl} $ weight
in the $L-2$ layer couples to all cells in the $L$ layer by the output $a^{L-1}_k$ of the $k$ cell 
in the $L-1$ layer.   

Note that 
$$
\begin{aligned}
    z^{L}_j &=& \sum_k w_{jk}f(z^{L-1}_k) \\ 
    z^{L}_j &=& \sum_k w_{jk}f(\sum_l w_{kl}^{L-2} a^{L-2}_l)
\end{aligned}  
$$  
thus, following the chain rule one obtains

$$
\begin{aligned}
    \frac{\partial z^{L}_j}{\partial w_{kl}^{L-2} } &=& \sum_k w_{jk}^{L-1} \frac{\partial f(z^{L-1}_j)}{\partial  w_{kl}^{L-2}} \\
    \frac{\partial z^{L}_j}{\partial w_{kl}^{L-2} } &=& \sum_k w_{jk}^{L-1}  f'(z^{L-1}_k) \frac{\partial z^{L-1}_k}{\partial  w_{kl}^{L-2}} \\   
    \frac{\partial z^{L}_j}{\partial w_{kl}^{L-2} } &=& \sum_k w_{jk}^{L-1}  f'(z^{L-1}_k) a^{L-2}_l , \\
\end{aligned}
$$

which can be cast in matrix form as 

$$
    \frac{\partial C}{\partial \mathbf{W}^{L-2}} = \delta^{L-1} \otimes \mathbf{a}^{L-2}. 
$$
where
$$
\begin{aligned}
    \delta^{L-1} &=& \left(\left(\mathbf{W}^{L-1}\right)^{T}\delta^{L} \right) \odot f^{'}\left( \mathbf{z}^{L-1}\right) 
\end{aligned}.
$$

It can be proved that for any inner layer $M$
$$
\begin{aligned}
    \delta^{M} &=& \left(\left(\mathbf{W}^{M}\right)^{T}\delta^{M+1} \right) \odot f^{'}\left( \mathbf{z}^{M}\right), 
\end{aligned}
$$
enabling one to propagate the error in the output to all inner layers. 

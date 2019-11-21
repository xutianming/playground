import torch

"""
A fully-connected ReLU network with one hidden layer, trained to predict y from x
by minimizing squared Euclidean distance.

This implementation uses the nn package from PyTorch to build the network.
PyTorch autograd makes it easy to define computational graphs and take gradients,
but raw autograd can be a bit too low-level for defining complex neural networks;
this is where the nn package can help. The nn package defines a set of Modules,
which you can think of as a neural network layer that has produces output from
input and may have some trainable weights or other state.
"""

#device = torch.device('cpu')
device = torch.device('cuda') # Uncomment this to run on GPU

# N is batch size; D_in is input dimension;
# H is hidden dimension; D_out is output dimension.
N, D_in, H, D_out = 2, 4, 8, 16

# Create random Tensors to hold inputs and outputs
x = torch.randn(N, D_in, device=device)
y = torch.randn(N, D_out, device=device)

# Use the nn package to define our model as a sequence of layers. nn.Sequential
# is a Module which contains other Modules, and applies them in sequence to
# produce its output. Each Linear Module computes output from input using a
# linear function, and holds internal Tensors for its weight and bias.
# After constructing the model we use the .to() method to move it to the
# desired device.
linear1 = torch.nn.Linear(D_in, H, bias=False).to(device)
relu = torch.nn.ReLU().to(device)
linear2 = torch.nn.Linear(H, D_out, bias=False).to(device)

# The nn package also contains definitions of popular loss functions; in this
# case we will use Mean Squared Error (MSE) as our loss function. Setting
# reduction='sum' means that we are computing the *sum* of squared errors rather
# than the mean; this is for consistency with the examples above where we
# manually compute the loss, but in practice it is more common to use mean
# squared error as a loss by setting reduction='elementwise_mean'.
factor = 0.5
loss_fn = torch.nn.MSELoss(reduction='sum')

learning_rate = 1e-4
for t in range(1):
  # Forward pass: compute predicted y by passing x to the model. Module objects
  # override the __call__ operator so you can call them like functions. When
  # doing so you pass a Tensor of input data to the Module and it produces
  # a Tensor of output data.
  print("X=>", x)
  print("Y=>", y)
  l1_out = linear1(x)
  print("Linear1 Out=>", l1_out)
  y_pred = linear2(l1_out)
  print("Y-Pred=>", y_pred)
  error = y_pred - y
  print("Error=>", error)
  print("Error matmul relu_out (linear1 grad)=>", factor * 2 * torch.mm(error.t(), l1_out))
  mse = factor * (error ** 2).sum()
  print("MSE=>", mse)
  # Compute and print loss. We pass Tensors containing the predicted and true
  # values of y, and the loss function returns a Tensor containing the loss.
  loss = factor * loss_fn(y_pred, y)
  print(t, loss.item())
  
  # Zero the gradients before running the backward pass.
  linear1.zero_grad()
  linear2.zero_grad()
  # Backward pass: compute gradient of the loss with respect to all the learnable
  # parameters of the model. Internally, the parameters of each Module are stored
  loss.backward()

  # Update the weights using gradient descent. Each parameter is a Tensor, so
  # we can access its data and gradients like we did before.
  with torch.no_grad():
    print("linear2 param:")
    for param in linear2.parameters():
      print("param=>", param)
      print("param.grad=>", param.grad)
      #param.data -= learning_rate * param.grad
      l1_error = torch.mm(error, param)
      print("l1 error=>", l1_error)
      l1_grad = factor * 2 * torch.mm(l1_error.t(), x)
      print("l1 grad=>", l1_grad)
    print("linear1 param:")
    for param in linear1.parameters():
      print("param=>", param)
      print("param.grad=>", param.grad)
      #param.data -= learning_rate * param.grad

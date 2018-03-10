import torch
from torch.autograd import Variable

N = 64
D_in = 1000
H = 100
D_out = 10

x = Variable(torch.randn(N, D_in))
y = Variable(torch.randn(N, D_out), requires_grad=False)

# Define our model as a sequence of layers
# nn.Sequential is a Module with other Modules that applies them
# in sequence to produce its output
# Each Linear Module uses a linear function. Holds internal Variables
# for weight and bias
model = torch.nn.Sequential(
    torch.nn.Linear(D_in, H),
    torch.nn.ReLU(),
    torch.nn.Linear(H, D_out),
)

# Use Mean Squared Error as loss function
loss_fn = torch.nn.MSELoss(size_average=False)

learning_rate = 1e-4
for t in range(500):
    # Forward pass. Compute predicted y. Modules override __call__ so they can be used as functions.
    y_pred = model(x)

    # Compute and print loss. Pass in predicted and true values for y. The returned variable is the loss
    loss = loss_fn(y_pred, y)
    print(t, loss.data[0])

    # Zero the gradients before running backward pass
    model.zero_grad()

    # Backward pass.
    loss.backward()

    # Update the weights using gradient descent
    for param in model.parameters():
        param.data -= learning_rate * param.grad.data

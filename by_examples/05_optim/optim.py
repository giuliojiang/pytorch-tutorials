import torch
from torch.autograd import Variable

N = 64
D_in = 1000
H = 100
D_out = 10

x = Variable(torch.randn(N, D_in))
y = Variable(torch.randn(N, D_out), requires_grad=False)

model = torch.nn.Sequential(
    torch.nn.Linear(D_in, H),
    torch.nn.ReLU(),
    torch.nn.Linear(H, D_out),
)
# Mean Squared Error
loss_fn = torch.nn.MSELoss(size_average=False)

# Use optim to define an Optimizer to update the weights for us. First argument to Adam tells which Variables it should update
learning_rate = 1e-4
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
for t in range(500):
    # Forward pass: compute predicted y
    y_pred = model(x)

    # Compute loss
    loss = loss_fn(y_pred, y)
    print(t, loss.data[0])

    # Use optimizer to zero all gradients for the variables it will update. By default gradients are accumulated in buffers.
    optimizer.zero_grad()

    # Backward pass
    loss.backward()

    # Step to update parameters
    optimizer.step()

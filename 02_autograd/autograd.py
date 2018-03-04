import torch
from torch.autograd import Variable

dtype = torch.cuda.FloatTensor

N = 64
D_in = 1000
H = 100
D_out = 10

x = Variable(torch.randn(N, D_in).type(dtype), requires_grad=False)
y = Variable(torch.randn(N, D_out).type(dtype), requires_grad=False)

w1 = Variable(torch.randn(D_in, H).type(dtype), requires_grad=True)
w2 = Variable(torch.randn(H, D_out).type(dtype), requires_grad=True)

learning_rate = 1e-6
for t in range(500):
    # Forward pass. Compute predicted y.
    # No need to keep any intermediate variable
    y_pred = x.mm(w1).clamp(min=0).mm(w2)

    # Compute and print loss
    # loss has shape (1,)
    # loss.data is a Tensor of shape (1,)
    # loss.data[0] holds the loss
    loss = (y_pred - y).pow(2).sum()
    print(t, loss.data[0])

    # autograd for backward pass.
    loss.backward()

    # Update weights
    w1.data -= learning_rate * w1.grad.data
    w2.data -= learning_rate * w2.grad.data

    # Manually zero gradients
    w1.grad.data.zero_()
    w2.grad.data.zero_()

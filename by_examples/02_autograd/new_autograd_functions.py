import torch
from torch.autograd import Variable

class MyReLU(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input):
        # receive a tensor
        # return a tensor
        # ctx can be used to store information for backward pass
        ctx.save_for_backward(input)
        return input.clamp(min=0)

    @staticmethod
    def backward(ctx, grad_output):
        # receive gradient of loss with respect to output
        # need to compute gradient with respect to input
        input, = ctx.saved_tensors
        grad_input = grad_output.clone()
        grad_input[input < 0] = 0
        return grad_input

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
    relu = MyReLU.apply

    # Forward pass
    y_pred = relu(x.mm(w1)).mm(w2)

    # Compute and print loss
    loss = (y_pred - y).pow(2).sum()
    print(t, loss.data[0])

    # Use autograd to compute backward pass
    loss.backward()

    # Update weights
    w1.data -= learning_rate * w1.grad.data
    w2.data -= learning_rate * w2.grad.data

    # Zero gradients
    w1.grad.data.zero_()
    w2.grad.data.zero_()

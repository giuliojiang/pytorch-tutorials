import torch
from torch.autograd import Variable

class TwoLayerNet(torch.nn.Module):

    def __init__(self, D_in, H, D_out):
        super(TwoLayerNet, self).__init__()
        self.linear1 = torch.nn.Linear(D_in, H)
        self.linear2 = torch.nn.Linear(H, D_out)

    def forward(self, x):
        h_relu = self.linear1(x).clamp(min=0)
        y_pred = self.linear2(h_relu)
        return y_pred

N = 64
D_in = 1000
H = 100
D_out = 10

# Create random Tensors for inputs and outputs
x = Variable(torch.randn(N, D_in))
y = Variable(torch.randn(N, D_out), requires_grad=False)

# Construct model
model = TwoLayerNet(D_in, H, D_out)

# Loss function and optimizer
criterion = torch.nn.MSELoss(size_average=False)
optimizer = torch.optim.SGD(model.parameters(), lr=1e-4)

for t in range(500):
    y_pred = model(x)

    loss = criterion(y_pred, y)
    print(t, loss.data[0])

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

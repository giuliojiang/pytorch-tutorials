import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        # 1 input image
        # 6 outputs
        # 5x5 square convolutions
        self.conv1 = nn.Conv2d(1, 6, 5) # in channels, out channels, kernel size
        self.conv2 = nn.Conv2d(6, 16, 5)
        # y = Wx + b
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        # Max pooling over a (2, 2) window
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        x = F.max_pool2d(F.relu(self.conv2(x)), (2, 2))
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:] # except batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

net = Net()
print(net)

params = list(net.parameters())
print(len(params))
print(params[0].size())

# Expected input size is 32x32
input = Variable(torch.randn(1, 1, 32, 32))
out = net(input)
print(out)

net.zero_grad()
out.backward(torch.randn(1, 10))

output = net(input)
target = Variable(torch.arange(1, 11))
criterion = nn.MSELoss()

print("target", target)

loss = criterion(output, target)
print("loss", loss)

print("mse loss", loss.grad_fn)
print("linear", loss.grad_fn.next_functions[0][0])
print("relu", loss.grad_fn.next_functions[0][0].next_functions[0][0])

# Backprop ====================================================================

net.zero_grad()
print("conv1.bias.grad before backward")
print(net.conv1.bias.grad)

loss.backward()

print("conv1.bias.grad after backward")
print(net.conv1.bias.grad)

# Update weights ==============================================================

import torch.optim as optim

optimizer = optim.SGD(net.parameters(), lr=0.01)

optimizer.zero_grad()
output = net(input)
loss = criterion(output, target)
loss.backward()
optimizer.step()

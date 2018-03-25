import torch
import torchvision
import torchvision.transforms as transforms
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import labelnet

# =============================================================================
# Load the network

net = torch.load("saved_network")

# =============================================================================
# Test it

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize(
         (0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

trainset = torchvision.datasets.CIFAR10(root="./data", train=True, download=True, transform=transform)

trainloader = torch.utils.data.DataLoader(trainset, batch_size=4, shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root="./data", train=False, download=True, transform=transform)

testloader = torch.utils.data.DataLoader(testset, batch_size=4, shuffle=False, num_workers=2)

classes = ("plane", "car", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck")

dataiter = iter(testloader)
images, labels = dataiter.next()

print("Ground truth {}".format(' '.join("%5s" % classes[labels[j]] for j in range(4))))

outputs = net(Variable(images))

_, predicted = torch.max(outputs.data, 1)

print("Predicted: ", " ".join("%5s" % classes[predicted[j]] for j in range(4)))

correct = 0
total = 0
for data in testloader:
    images, labels = data
    outputs = net(Variable(images))
    _, predicted = torch.max(outputs.data, 1)
    total += labels.size(0)
    correct += (predicted == labels).sum()

print("Accuracy on the 10000 test images: {}".format(100.0 * correct / total))

import torch
import torchvision
import torchvision.transforms as transforms

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize(
         (0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

trainset = torchvision.datasets.CIFAR10(root="./data", train=True, download=True, transform=transform)

trainloader = torch.utils.data.DataLoader(trainset, batch_size=4, shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root="./data", train=False, download=True, transform=transform)

testloader = torch.utils.data.DataLoader(testset, batch_size=4, shuffle=False, num_workers=2)

classes = ("plane", "car", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck")

## Functions to show an image
#import matplotlib.pyplot as plt
#import numpy as np

#def imshow(img):
    #img = img / 2 + 0.5
    #npimg = img.numpy()
    #plt.imshow(np.transpose(npimg, (1, 2, 0)))

## get some random training images
#dataiter = iter(trainloader)
#images, labels = dataiter.next()

## show images
#imshow(torchvision.utils.make_grid(images))
## print labels
#print(' '.join('%5s' % classes[labels[j]] for j in range(4)))

# =============================================================================
# Define a convolution neural network

from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F

import labelnet

net = labelnet.Net()
net.cuda()

# =============================================================================
# Define loss function and optimizer

import torch.optim as optim

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

# =============================================================================
# Train the network

for epoch in range(2):

    running_loss = 0.0

    for i, data in enumerate(trainloader, 0):

        # get the inputs
        inputs, labels = data

        # wrap them in Variable
        inputs = Variable(inputs.cuda())
        labels = Variable(labels.cuda())

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward, backward, optimize
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.data[0]
        if i % 2000 == 1999:
            print("[{}, {}] loss {}".format(epoch, i, running_loss / 2000))
            running_loss = 0.0

print("Finished training")
net = net.cpu()

# =============================================================================
# Test the network on the test data

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

# =============================================================================
# Save network

torch.save(net, "saved_network")

print("Network saved. Complete.")

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
import torch.optim as optim

from Model import CNNclassifier

#hyperparameters
lr = 0.1
batch_size = 512
num_workers = 0
max_epoch = 100
patience = 5 #early stopping patience

#dataset parameters
img_size = 28 
n_channels = 1 
n_classes = 10 

#name for locally saving checkpoint and training stats
directory = 'notransforms'

transform= transforms.Compose([transforms.ToTensor(),
                                     transforms.Normalize((0.5,), (0.5,))])
#many FMNIST models used this normalization

#downloads and loads the dataset using the using torchvision/pytorch
train_set = torchvision.datasets.FashionMNIST('dataset', train = True, transform=transform, download=True)
trainloader = torch.utils.data.DataLoader(train_set, batch_size=batch_size,
                                          shuffle=True, num_workers=num_workers)

test_set = torchvision.datasets.FashionMNIST('dataset', train = False, transform=transform, download=True)
testloader = torch.utils.data.DataLoader(test_set, batch_size=batch_size,
                                          shuffle=True, num_workers=num_workers)

#initialise model
net = CNNclassifier(img_size, n_channels, n_classes) 

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=lr) 
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1, last_epoch=- 1, verbose=False)

#log the training statistics
train_loss = []
test_loss = []
train_accuracy = []
test_accuracy = []
learning_rate = []

#counts the number of epochs without a validation accuracy improvement - basic early stopping
counter = 0 

for epoch in range(max_epoch):  # loop over the dataset multiple times
    training_loss = 0.0
    correct_per_epoch_train = 0.0

    for i, data in enumerate(trainloader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data
        # zero the parameter gradients
        optimizer.zero_grad()
        # forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        training_loss += loss.item()
        #compute accuracy
        max_index = torch.argmax(outputs, dim=1)
        correct_per_batch = (max_index == labels).sum()
        correct_per_epoch_train += correct_per_batch
    
    scheduler.step()
    learning_rate.append((optimizer.param_groups[0]['lr']))
    train_loss.append(training_loss/60000)
    train_accuracy.append(correct_per_epoch_train.item()/60000)

    testing_loss = 0.0
    correct_per_epoch_test = 0.0

    for i, data in enumerate(testloader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        testing_loss += loss.item()
        max_index = torch.argmax(outputs, dim=1)
        correct_per_batch = (max_index == labels).sum()
        correct_per_epoch_test += correct_per_batch

    test_loss.append(testing_loss/10000)
    test_accuracy.append(correct_per_epoch_test.item()/10000)

    if len(test_loss) > 1:
        if round(test_accuracy[-1], 3) < round(max(test_accuracy[:-1]), 3): #check if there is improvement in test acc
            counter += 1
        else: 
            counter = 0
            torch.save(net.state_dict(), 'checkpoint' + directory) #saves a checkpoint

    if counter == patience: #breaks based on early stopping patience set
        break

    print(("Epoch: % s \n Train Accuracy: % s \n Test Accuracy: % s \n" % (len(test_accuracy), train_accuracy[-1], test_accuracy[-1])))

epochs = [i for i in range(1, len(train_loss) + 1)]

plt.subplot(3, 1, 1)
plt.plot(epochs, train_accuracy)
plt.plot(epochs, test_accuracy)
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(["Train", "Test"])

plt.subplot(3, 1, 2)
plt.plot(epochs, train_loss)
plt.plot(epochs, test_loss)
plt.xlabel('Epoch')
plt.ylabel('Cross Entropy Loss')
plt.legend(["Train", "Test"])

plt.subplot(3, 1, 3)
plt.plot(epochs, learning_rate)
plt.xlabel('Epoch')
plt.ylabel('Learning rate')

plt.show()
#saves the plot of training stats
plt.savefig(directory)

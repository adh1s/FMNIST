import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np

from Model import CNNclassifier

img_size = 28
n_channels = 1
n_classes = 10

def output_label(input):
    output_mapping = {
                 0: "T-shirt/Top",
                 1: "Trouser",
                 2: "Pullover",
                 3: "Dress",
                 4: "Coat", 
                 5: "Sandal", 
                 6: "Shirt",
                 7: "Sneaker",
                 8: "Bag",
                 9: "Ankle Boot"
                 }

    input = (label.item() if type(label) == torch.Tensor else label)
    return output_mapping[input]

def visualise_tensor(img):
    img = torch.squeeze(img, dim=0) 
    img = img.cpu().numpy()
    #convert image back to Height,Width,Channels
    img = np.transpose(img, (1,2,0))
    #show the image
    plt.imshow(img)
    plt.show()  

#model to use for inference
directory = 'trained_model'
download = True

model = CNNclassifier(img_size=img_size, n_channels=n_channels, n_classes=n_classes) #configured for F-MNIST
model.load_state_dict(torch.load(directory))
model.eval()

transform_test= transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize((0.5,), (0.5,))])

test_set = torchvision.datasets.FashionMNIST('dataset', train = False, transform=transform_test, download=download)
testloader = torch.utils.data.DataLoader(test_set, batch_size=1,
                                         shuffle=True, num_workers=0)

#picks a random image from the test set - displays image and prints prediction
inputs, classes = next(iter(testloader)) 
outputs = model(inputs)
label = torch.argmax(outputs, dim=1)
class_label = output_label(label)

visualise_tensor(inputs)
print(class_label)

'''
# Gathers a class wise incorrect/correct ratio -> helps to identify worst performing classes

correct_dictionary = {}
incorrect_dictionary = {}

for i, data in enumerate(testloader, 0):
        inputs, labels = data
        outputs = model(inputs)
        max_index = torch.argmax(outputs, dim=1)
        if max_index == labels:
            if labels.item() in correct_dictionary.keys():
                correct_dictionary[labels.item()] += 1
            else:
                correct_dictionary[labels.item()] = 1
        else:
            if labels.item() in incorrect_dictionary.keys():
                incorrect_dictionary[labels.item()] += 1
            else:
                incorrect_dictionary[labels.item()] = 1

# create data
x = ["T-shirt/Top","Trouser","Pullover","Dress","Coat", "Sandal", "Shirt","Sneaker","Bag", "Ankle Boot"]
incorrect_over_correct = []

for i in range(10):
    incorrect_over_correct.append(incorrect_dictionary[i]/correct_dictionary[i])
    
# plot bars in stack manner
plt.bar(x, incorrect_over_correct, color='b')
plt.title('Incorrect/Correct')
plt.show()
'''

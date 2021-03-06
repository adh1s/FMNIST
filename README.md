# FMNIST

Implementation of a CNN classifier for the Fashion MNIST dataset. The CNN uses a standard classifier architecture - two 3x3 convolution layers, followed by three fully connected layers (with dropout and batch normalization). The 'CNNclassifer' class is flexible and can be used with different datasets (with different input dimensions and number of classes). Simple training (using learning rate scheduling and early-stopping) and inference functionality is also provided.

### Environment:

This code was used on a MacOS 11.0.1 (20B50) machine on VScode using the following setup:

* python (3.8.5) with:

  * torch (1.9.0)
  * torchvision (0.10.0)
  * numpy (1.19.2)
  * matplotlib (3.4.2)
  * optional: torchsummary (1.5.1)

### Component Description:

There are three main components:

* Train.py : the main script which trains the CNN - loads the F-MNIST dataset and runs the training of the CNN with early-stopping.
* Model.py : this file implements the architecture of the CNN - a summary (through torchsummary module) of the architecture is provided here too.
* Inference.py : provides a few helper functions to help with inference. Running this file passes a random image from the validation dataset - printing the input image and the associated output class.

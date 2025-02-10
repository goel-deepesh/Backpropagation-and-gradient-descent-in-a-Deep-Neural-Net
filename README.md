# Backpropagation-and-gradient-descent-in-a-Deep-Neural-Net

BACKPROPAGATION

Implemented the forward pass and backward pass for Linear, ReLU, Sigmoid, MSE loss, and BCE loss. Two test cases have been included - test1.py and test2.py - to benchmark the manually computed gradients against autograd.

- Used Python 3.12 and Pytorch 2.6 for the implementation. 
- Put the mlp.py file under the same directory of the test scripts and use the command python TestScriptName.py to check the implementation. Please make sure the file name is mlp.py.

GRADIENT DESCENT

In DeepDream, the paper claims that you can follow the gradient to maximize an energy with respect to the input in order to visualize the input. Given an image classifier, implemented a function that performs optimization on the input (the image), to find the image that most highly represents the class. Find the implementation in the file gd.py.

- Tried to minimize the energy of the class, e.g. maximize the class logit.
- Started with a reasonable starting learning rate of 0.01.
- Used normalize_and_jitter, since the neural network expects a normalized input. Jittering produces more visually pleasing results.

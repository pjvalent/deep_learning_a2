"""
This is the only python file you should edit.
Similar to baseline.py, it should define a neural network model suitable for the machine translation dataset
You must complete the initialize_for function at bottom, which initializes a network and optimizer to be used for training
initialize_for has the following inputs:
    max_len: the maximum sequence length in the dataset
    embeddings: the torch array of embeddings for each token, one row per token
and should return the following outputs:
    net: a torch module that predicts the next output token, based on input and output tokens so far
    opt: a torch optimizer that can be used to optimize the parameters of net
"""
import torch as tr
from utils import *

class Net(tr.nn.Module):

    # Initialize your neural network modules
    def __init__(self, max_len, embeddings):
        super(Net, self).__init__()

        # This is just a placeholder network, as simple as possible
        # Replace with your own network architecture
        self.readout = tr.nn.Linear(embeddings.shape[1], embeddings.shape[1])

    # Forward pass arguments:
    #   inputs: embedded input token sequence
    #   outputs: embedded output token sequence predicted so far
    # Returns:
    #   logits: input to softmax for predicting the next output token
    #   data: any other auxiliary data you want to return
    def forward(self, inputs, outputs):
        
        # Replace with your own network forward pass
        logits = self.readout(inputs)
        data = None
        return logits, data

# Initialize a network and optimizer to be used for training
def initialize_for(max_len, embeddings):

    # Replace with your implementation
    # You can choose a different learning rate, optimizer, etc.
    # Just make sure to return the correct datatypes:
    #    net: a torch module
    #    opt: a torch optimizer
    net = Net(max_len, embeddings)
    opt = tr.optim.Adam(net.parameters(), lr=0.0001)

    return net, opt

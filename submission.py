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

    # Initialize all the sub-modules (multihead attention, etc.)
    def __init__(self, max_len, embeddings):
        super(Net, self).__init__()

        d_model = max_len + embeddings.shape[1]
        num_heads = 1

        self.encoder = MultiHeadAttention(num_heads, d_model, projections="")
        self.precoder = MultiHeadAttention(num_heads, d_model, masked=True, projections="O")
        self.decoder = MultiHeadAttention(num_heads, d_model, projections="")
        self.readout = tr.nn.Linear(d_model, embeddings.shape[1])
        self.lrelu = tr.nn.LeakyReLU()
        self.embeddings = embeddings
        self.positional_encoder = one_hot_positional_encoder(max_len)
        self.max_len = max_len

    # Forward pass arguments:
    #   inputs: embedded input token sequence
    #   outputs: embedded output token sequence predicted so far
    # Returns softmax logits for predicting the next output token
    def forward(self, inputs, outputs):
        Q = K = V = self.positional_encoder(inputs)
        encoded = self.encoder(Q, K, V)
        Q = K = V = self.positional_encoder(outputs)
        precoded = self.precoder(Q, K, V)
        decoded = self.decoder(precoded, encoded, encoded)
        decoded[:,:self.max_len] = 0 # ignore position information at this point
        if self.readout == None:
            logits = decoded[:,-self.embeddings.shape[1]:] @ self.embeddings.t()
        else:
            logits = self.lrelu(self.readout(decoded)) @ self.embeddings.t() # learned read-out connections
        return logits, (encoded, precoded, decoded)

# Initialize a network and optimizer to be used for training
def initialize_for(max_len, embeddings):

    # Replace with your implementation
    # You can choose a different learning rate, optimizer, etc.
    # Just make sure to return the correct datatypes:
    #    net: a torch module
    #    opt: a torch optimizer
    net = Net(max_len, embeddings)
    opt = tr.optim.NAdam(net.parameters(), lr=0.01)

    return net, opt

"""
Do NOT permanently edit this file, it scores your submission and will be replaced by the instructor's copy
"""
import os
import pickle as pk
import random
import numpy as np
import torch as tr
import matplotlib.pyplot as pt

# To ensure reproducibility, don't change these seeds
random.seed(1234567890)
np.random.seed(1234567890)
tr.manual_seed(1234567890)

# Utility for embedding tokens in a vector space
from utils import embed

# Your model is imported here
# import baseline as model
# import reference as model
import submission as model

# Load all the training/validation data
with open("tokens.pkl","rb") as f: tokens = pk.load(f)
with open("lookup.pkl","rb") as f: lookup = pk.load(f)
with open("train.pkl","rb") as f: train = pk.load(f) # training data
with open("valid.pkl","rb") as f: valid = pk.load(f) # validation data

# Initialize the max sequence length, token embeddings, and loss function
max_len = 10 # No sentences with more than 10 words
embeddings = tr.eye(len(tokens)) # one-hot token embeddings
xc = tr.nn.CrossEntropyLoss() # between predicted and actual next token

# Initialize the training iterations
num_reps = 3 # independent repetitions of the experiment
num_iters = 5000 # number of gradient updates per repetition, kept fixed for all students
validation_period = 500 # period between validation accuracy measurements

# placeholders for per-repetition learning curves
train_loss = [[] for _ in range(num_reps)] # loss on training data
valid_accu = [[] for _ in range(num_reps)] # accuracy on validation data

# Run each training repetition
for rep in range(num_reps):
    
    # Initialize a new model and optimizer from your code
    net, opt = model.initialize_for(max_len, embeddings)

    # Run the training iterations
    for i in range(num_iters):

        # Sample a random training example
        src, dst = random.choice(train)
        trg = dst
    
        # Embed the input/output sequences
        inputs = embed(src, max_len, embeddings, lookup)
        outputs = embed(dst, max_len, embeddings, lookup, offset=1)

        # Target tokens at each time-step in index format for cross-entropy loss
        targets = tr.tensor([lookup[token] for token in trg])
    
        # Make the predictions, calculate and save the loss
        logits, _ = net(inputs, outputs)
        loss = xc(logits[:len(targets)], targets)
        train_loss[rep].append(loss.item())

        # Compute the loss gradient and update the model parameters
        opt.zero_grad()
        loss.backward()
        opt.step()
    
        # Measure accuracy on the validation set
        if i % validation_period == 0 or i == num_iters - 1:

            # Don't need gradients for this
            with tr.no_grad():

                # Check correctness on each validation example
                correct = []
                for src, trg in valid:

                    # Embed inputs and predict outputs one at a time
                    inputs = embed(src, max_len, embeddings, lookup)
                    out = [] # initially no outputs have been predicted
                    for t in range(len(trg)):

                        # use predicted outputs so far to predict the next one
                        outputs = embed(out, max_len, embeddings, lookup, offset=1)
                        logits, _ = net(inputs, outputs)
                        prediction = tokens[logits[t].argmax()]
                        out.append(prediction)

                    # Check correctness against target
                    correct.append(tuple(trg) == tuple(out))

            # Compute and save accuracy over entire validation set
            accuracy = np.mean(correct)
            valid_accu[rep].append(accuracy)
        
            print("rep %d, epoch %d: training loss %f, validation accuracy %f" % (rep, i, loss.item(), accuracy))

# Display results
train_loss = np.array(train_loss)
valid_accu = np.array(valid_accu)

print("Average validation regret per update:")
for rep in range(num_reps):
    print("Rep %d:" % rep, (1 - valid_accu[rep]).mean())
print("Average over all reps = %f" % (1 - valid_accu).mean())

pt.subplot(2,1,1)
pt.plot(train_loss.T)
pt.xlabel("Iteration")
pt.ylabel("Training loss")
pt.title("Learning curves (%d reps)" % num_reps)
pt.subplot(2,1,2)
pt.plot(list(range(0, num_iters, validation_period)) + [num_iters-1], valid_accu.T)
pt.xlabel("Iteration")
pt.ylabel("Validation accuracy")
pt.title("Average validation regret = %f" % (1 - valid_accu).mean())
pt.tight_layout()
pt.savefig(model.__name__ + ".png")
pt.show()


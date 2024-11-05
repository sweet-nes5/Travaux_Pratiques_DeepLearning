import torch
from torch import nn as nn
import torch.nn.functional as F
from pub_preliminaries import show_data_L2C2
import matplotlib.pyplot as plt


# Example of concatenation, shuffle and split

labels = torch.cat((torch.zeros(100), torch.ones(100)))
s = torch.randperm(200) # random permutation
labels_shuffled = labels[s]
part1 = labels_shuffled[:150]
part2 = labels_shuffled[150:]

# Accuracy and loss functions

def accuracy(predictions, ys):
    ys = ys.unsqueeze(dim=1) #  ATTENTION: to work with linear
    pred_classes = predictions > 0.5 # fix a threshold 
    success_rate = (pred_classes.to(int) == ys).to(float).mean() # probability of having everything right
    return (success_rate * 100).item()

def loss(predictions, ys):
    ys = ys.unsqueeze(dim=1) # ATTENTION: to work with linear
    return (-(ys * torch.log(predictions) + (1-ys) * torch.log(1 - predictions))).mean()


# Learning with monitoring
import torch.optim as optim
def learning_process(
    model,                  # a nn, possibly of several layers 
    train_data,            # training data inputs
    train_labels,            # training data labels
    validation_inputs,      # validation data inputs
    validation_labels,      # validation data labels
    loss_function,          # loss function
    optimizer,               # an optimizer (learning rate is incorporated), e.g. optim.SGD(model.parameters(), lr=0.1)
    accuracy_function=None, # accuracy function
    n_epochs=10):           # nb of epochs
    history = []
    for epoch in range(n_epochs):    
        predictions = model(train_data)    # notice how model is called  
        loss = loss_function(predictions, train_labels)
        model.zero_grad()
        loss.backward()
        optimizer.step()                    # here model parameters are updated
        # just for monitoring
        if epoch % 100 == 0: 
            acc = accuracy_function(predictions, train_labels)
            loss_val = loss_function(model(validation_inputs), validation_labels)
            acc_val = accuracy_function(model(validation_inputs), validation_labels)
            history.append((epoch, loss.detach().numpy(), acc, loss_val.detach().numpy(), acc_val))
            print(f"epoch {epoch},\t train loss = {loss.detach().numpy():.4f},\t validation loss = {loss_val.detach().numpy():.4f}")        
            print(f"epoch {epoch},\t train accuracy = {acc:.2f},\t validation accuracy = {acc_val:.2f}")

    return history

# Plot monitoring
def ith(l, i):
    return [x[i] for x in l]

def plot_history(h):
    fig, axs = plt.subplots(1, 2)
    axs[0].plot(ith(h,0), ith(h,1))
    axs[0].plot(ith(h,0), ith(h,3), color='red')
    axs[0].set_xlabel('epochs')
    axs[0].set_ylabel('loss')
    axs[1].plot(ith(h,0), ith(h,2))
    axs[1].plot(ith(h,0), ith(h,4), color='red')
    axs[1].set_xlabel('epochs')
    axs[1].set_ylabel('accuracy')
    fig.suptitle('Monitor learning')
    fig.tight_layout()
    plt.show()


# The model using torch.nn.Sequential
model_seq =  nn.Sequential(
    nn.Linear(2, 100),
    nn.ReLU(),
    nn.Linear(100, 1),
    nn.Sigmoid()
)



history = learning_process(model_seq_small, train_data, train_labels, validation_inputs, validation_labels, loss_function=loss, accuracy_function=accuracy, lr=0.1, n_epochs=10000)

show_data_L2C2(train_data, train_labels, model=model_seq_small)

plot_history(history)



###### Spirales

def generate_data_NL2C3(nb_samples, noise=0.0): 
    """
        returns a pair of tensors (inputs,labels) 
        inputs of shape (nb_samples*3, 3)
        labels of shape (nb_samples*3)
        (Code from http://cs231n.stanford.edu)
    """
    inputs = torch.empty(nb_samples * 3, 2)
    labels = torch.empty(nb_samples * 3, dtype=int)
    for j in range(3):  # number of categories
        ix = range(nb_samples * j, nb_samples * (j + 1))
        r = torch.linspace(0.0, 1, nb_samples)
        t = torch.linspace(j * 4, (j + 1) * 4, nb_samples) + torch.randn(nb_samples) * noise  
        inputs[ix] = torch.column_stack((r * torch.sin(t), r * torch.cos(t)))
        labels[ix] = j
    
    return (inputs, labels)

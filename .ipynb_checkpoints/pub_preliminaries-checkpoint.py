import torch
from torch import nn as nn
from torch.nn import functional as F
import numpy as np  # for meshgrid
import matplotlib.pyplot as plt
from torch.nn.modules.linear import Linear

# VISUALISATION FUNCTIONS
# Visualisation data and model 1 feature regression
def show_data_R1(data_inputs, data_outputs, model=None):
    _ = plt.figure()
    ax = plt.axes()
    ax.scatter(data_inputs, data_outputs, c='red')
    if model is not None:
        xs = torch.linspace(0.,1.,20)
        ys = model(xs)
        ax.plot(xs,ys, c='blue')
    plt.show()

# Visualisation loss ◦ model function for 1 feature regression
def show_loss_R1(f):
    _ = plt.figure()
    ax = plt.axes(projection='3d')
    sample_weights  = torch.linspace(0,1,50)   # 50 samples of 1D weights in the [1,-1] interval
    sample_bias = torch.linspace(0,1,50)       # 50 samples of scalar bias in the [1,-1] interval
    W, B = torch.meshgrid(sample_weights, sample_bias)
    E = torch.tensor([[f(w,b) for w,b in zip(ws,bs)] for ws,bs in zip(W, B)]) # 20x20 outputs of f on all possible combinations of above samples
    ax.plot_surface(W, B, E)
    plt.show()

# Visualisation data and model 2 features regression
def show_data_R2(data_inputs, data_outputs, model=None):
    _ = plt.figure()
    ax = plt.axes(projection='3d')
    if model is not None:
        # TODO
        x1s = torch.linspace(0.,1.,20)
        x2s = torch.linspace(0.,1.,20)
        X1, X2 = torch.meshgrid(x1s, x2s)
        Y = torch.tensor([[model(torch.Tensor([x1,x2])) for x1,x2 in zip(x1s,x2s)] for x1s, x2s in zip(X1, X2)]) # 20x20 outputs of f on all possible combinations of above samples
        ax.plot_surface(X1, X2, Y)
    ax.scatter3D(data_inputs[:,0], data_inputs[:,1], data_outputs, c='red')
    plt.show()

# Visualisation data and model 2 features binary regression
def show_data_L2C2(data_inputs, data_labels, model=None):
    zeros = ([x[0] for x,l in zip(data_inputs, data_labels) if l == 0], [x[1] for x,l in zip(data_inputs, data_labels) if l == 0])
    ones = ([x[0] for x,l in zip(data_inputs, data_labels) if l == 1], [x[1] for x,l in zip(data_inputs, data_labels) if l == 1])
    _ = plt.figure()
    ax = plt.axes()
    if model is not None: # for plotting decision boundary of the model
        step = 0.02
        xx, yy = torch.meshgrid(torch.arange(-1.0, 1., step), torch.arange(-1.0, 1., step), indexing='ij')
        labels = model(torch.column_stack((xx.ravel(),yy.ravel())))
        labels = (labels > 0.5)
        plt.pcolormesh(xx, yy, labels.reshape(xx.shape), alpha=0.7, shading='auto')
    ax.scatter(ones[0], ones[1], c='blue', edgecolor="white", linewidth=1)
    ax.scatter(zeros[0], zeros[1], c='red', edgecolor="white", linewidth=1)        
    plt.show()

# Visualisation data and model 2 features ternary regression
def show_data_L2C3(data_inputs, data_labels, model=None):
    zeros = ([x[0] for x,l in zip(data_inputs, data_labels) if l == 0], [x[1] for x,l in zip(data_inputs, data_labels) if l == 0])
    ones = ([x[0] for x,l in zip(data_inputs, data_labels) if l == 1], [x[1] for x,l in zip(data_inputs, data_labels) if l == 1])
    twos = ([x[0] for x,l in zip(data_inputs, data_labels) if l == 2], [x[1] for x,l in zip(data_inputs, data_labels) if l == 2])
    fig, ax = plt.subplots()
    if model is not None: # for plotting state of the model
        step = 0.02
        xx, yy = torch.meshgrid(torch.arange(-1.0, 1.1, step), torch.arange(-1.0, 1.1, step), indexing='ij')
        labels = model(torch.column_stack((xx.ravel(),yy.ravel())))
        labels = labels.argmax(1) 
        plt.pcolormesh(xx, yy, labels.reshape(xx.shape), alpha=0.2, shading='auto')
    ax.scatter(zeros[0], zeros[1], c='purple', edgecolor="white", linewidth=1) 
    ax.scatter(ones[0], ones[1], c='green', edgecolor="white", linewidth=1)
    ax.scatter(twos[0], twos[1], c='goldenrod', edgecolor="white", linewidth=1)    
    plt.show()

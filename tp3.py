# Exercie 1
import torch
import pub_preliminaries as pb
import torch.nn as nn


def generate_data_L2C2(nb_samples):
    data_inputs = torch.rand(nb_samples, 2) * 2 - 1  # convertir dans l'intervalle [-1,1]
    weight1 = -1.5
    intercept = 0.5
    bruit = torch.normal(mean=0.0, std=0.01, size=(nb_samples,))
    data_labels = (weight1 * data_inputs[:, 0] + data_inputs[:, 1] + intercept + bruit) > 0
    return (data_inputs, data_labels)


data_inputs, data_labels = generate_data_L2C2(300)
pb.show_data_L2C2(data_inputs, data_labels)
# Exercice 2
def model_ws(X, weights, bias):
    return torch.sigmoid(torch.matmul(X, weights) + bias)


weights = torch.randn(2, requires_grad=True)
bias = torch.randn(1, requires_grad=True)
model = lambda x: model_ws(x, weights, bias)
pb.show_data_L2C2(data_inputs, data_labels, model=model)
# Exercice 3
def loss_function(actual, predicted):
    epsilon = 1e-7
    prediction = torch.clamp(predicted, epsilon, 1.0 - epsilon)
    return -torch.sum(actual * torch.log(prediction) + (1 - actual) * torch.log(1 - prediction))


def f(weight, bias):
    predicted = model_ws(data_inputs, weight, bias)
    return loss_function(data_labels.long(), predicted)

loss = f(weights, bias)
print("Loss function calculates : ", loss.item())
# Exercice 4
def grad_des(weight, bias, lr, epochs):
    for iter in range(epochs):

        # Forward pass
        loss = f(weight, bias)

        # Backward pass
        loss.backward()

        # Update
        with torch.no_grad():
            weight -= lr * weight.grad
            bias -= lr * bias.grad

            # Prevent accumulation by zeroing the grad
            weight.grad.zero_()
            bias.grad.zero_()

        if iter % 1000 == 0:
            print(f" la perte pour l'epoch {iter} est de : {loss}")
    return weight, bias


lr = 0.001
#epochs = 500000
epochs = 5000
weight_des, bias_des = grad_des(weights, bias, lr, epochs)
#affichage de la classification optimisée
model = lambda x: model_ws(x, weight_des, bias_des)
pb.show_data_L2C2(data_inputs, data_labels, model = model)
# Exercice 5
def generate_data_L2C3(nb_samples):
    data_inputs = torch.rand(nb_samples, 2) * 2 - 1
    bruit = torch.normal(mean=0.0, std=0.01, size=(nb_samples,))
    condition_1 = (3 * data_inputs[:, 0] - data_inputs[:, 1] - 0.2 + bruit) > 0
    condition_2 = (-0.5 * data_inputs[:, 0] - data_inputs[:, 1] + 0.25 + bruit) > 0
    condition_3 = ~ (condition_1 | condition_2)
    # init y
    data_labels = torch.zeros(nb_samples, dtype=torch.long)
    data_labels[condition_1] = 0
    data_labels[condition_2] = 1
    data_labels[condition_3] = 2
    return (data_inputs, data_labels)
data_inputs, data_labels = generate_data_L2C3(500)
pb.show_data_L2C3(data_inputs, data_labels)
# Exercice 6
def model_ws3(X, weights, bias):
    temp = torch.matmul(X, weights) + bias
    return torch.nn.LogSoftmax(dim=1)(temp)

weights_ = torch.randn(2, 3, requires_grad=True)
bias_ = torch.randn(3, requires_grad=True)
model = lambda x: model_ws3(x, weights_, bias_)
pb.show_data_L2C3(data_inputs, data_labels, model = model)
#Exercice 7
def f_(weights, bias):
    loss_fn = nn.NLLLoss()
    predicted = model_ws(data_inputs, weights, bias)
    loss = loss_fn(predicted, data_labels)
    return loss
loss = f_(weights_, bias_)
print("Loss function calculates:", loss.item())
# Exercice 8
def grad_des2(weight, bias, lr, epochs):
    for iter in range(epochs):

        # Forward pass
        loss = f_(weight, bias)

        # Backward pass
        loss.backward()

        # Update
        with torch.no_grad():
            weight -= lr * weight.grad
            bias -= lr * bias.grad

            # Prevent accumulation by zeroing the grad
            weight.grad.zero_()
            bias.grad.zero_()

        if iter % 1000 == 0:
            print(f" la perte pour l'epoch {iter} est de : {loss}")
    return weight, bias


lr = 0.001
#epochs = 500000
epochs = 5000
weight_des, bias_des = grad_des2(weights_, bias_, lr, epochs)
#affichage de la classification optimisée
model = lambda x: model_ws3(x, weight_des, bias_des)
pb.show_data_L2C2(data_inputs, data_labels, model = model)
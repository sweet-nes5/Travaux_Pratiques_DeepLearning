import torch
import matplotlib.pyplot as plt


# exo1
def generate_data_R1(nb_samples):
    data_inputs = torch.rand(nb_samples)
    weight = 0.4
    intercept = 0.5
    bruit = torch.normal(mean=0.0, std=0.01, size=(nb_samples,))
    data_outputs = weight * data_inputs + intercept + bruit
    return data_inputs, data_outputs


inputs, outputs = generate_data_R1(100)
fig2 = plt.figure(num="Regression lineaire avec poids aleatoires")
ax2 = fig2.add_subplot()
ax2.scatter(inputs, outputs, c="red")
plt.xlabel('X')
plt.ylabel('Y')


def model_ws(x, weight, bias):
    return weight * x + bias

def model_ws2(X, weights, bias):
    return torch.matmul(weights, X) + bias

weight = torch.rand(1)
bias = torch.rand(1)
x = torch.linspace(0.,1.,20)
y = model_ws(x, weight, bias)
ax2.plot(x,y,c='blue')
plt.show()

#exo2
def loss_function(output, y):
    return torch.mean((output - y) ** 2)


def f(weight, bias):
    predicted = model_ws(inputs, weight, bias)
    return loss_function(outputs, predicted)

weight_values = torch.linspace(0,1,100)
bias_values = torch.linspace(0,1,100)
weight_mesh, bias_mesh = torch.meshgrid(weight_values, bias_values, indexing='ij')
# calculer la perte pour chaque x
loss_values = torch.zeros_like(weight_mesh)
for i in range(weight_mesh.size(0)):
    for j in range(weight_mesh.size(1)):
        loss_values[i, j] = f(weight_mesh[i, j], bias_mesh[i, j]).item()

fig = plt.figure(num="Gradient descent")
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(weight_mesh, bias_mesh, loss_values)
ax.set_xlabel('Weight')
ax.set_ylabel('Bias')
ax.set_zlabel('Loss')
plt.show()

# exo3
def df_dw(weight, bias): # calcul de la derivée partielle par rapport a weight
    return 2 * inputs * (weight * inputs + bias - outputs)

def df_db(weight, bias) : # calcul de la derivée partielle par rapport au biais
    return 2 * (weight * inputs + bias - outputs)

weight = torch.Tensor([2.0])
bias = torch.Tensor([1.0])
df_dw_moy = torch.mean(df_dw(weight, bias))
df_db_moy = torch.mean(df_db(weight, bias))
print(f"la dérivée moyenne de f par rapport à weight est {df_dw_moy}")
print(f"la dérivée moyenne de f par rapport au bias est {df_db_moy}")

# exo 4
lr = 0.5
epochs = 100
def grad_des(weight, bias, lr, epochs):
    for iter in range(epochs):
        #calcul des dérivées
        df_dw_moy = torch.mean(df_dw(weight, bias))
        df_db_moy = torch.mean(df_db(weight, bias))
        #ensuite on met à jour les parametres
        weight -= lr * df_dw_moy
        bias -= lr * df_db_moy
        loss = f(weight, bias)
        if iter % 10 == 0:
            print(f" la perte pour l'epoch {iter} est de : {loss}")
    return weight, bias


weight_des, bias_des = grad_des(weight, bias, lr, epochs)

# dessiner la droite qui minimise la fonction d'erreur
fig3 = plt.figure(num="Regression lineaire grace à la descente gradient")
ax3 = fig3.add_subplot()
ax3.scatter(inputs, outputs, c="red")
plt.xlabel('X')
plt.ylabel('Y')
x = torch.linspace(0.,1.,20)
y = model_ws(x, weight_des, bias_des)
ax3.plot(x,y,c='blue')
plt.show()

#exo 5
def generate_data_R2(nb_samples):
    X = torch.rand(nb_samples, 2)
    weight1 = 0.4
    weight2 = 0.2
    intercept = 0.5
    bruit = torch.normal(mean=0.0, std=0.01, size=(nb_samples,))
    y = weight1 * X[:,0] + weight2 * X[:, 1] + intercept + bruit
    return X, y

data_inputs2d, data_outputs2d = generate_data_R2(100)
fig1 = plt.figure(num="Visualisation 3D à 2 dimensions")
ax1 = fig1.add_subplot(111, projection='3d')
ax1.scatter3D(data_inputs2d[:, 0], data_inputs2d[:,1], data_outputs2d, c="red")
ax1.set_xlabel('X1')
ax1.set_ylabel('X2')
ax1.set_zlabel('Y')


#exo6


weights = torch.rand(1, 2)
bias_2d = torch.rand(1)
x = torch.linspace(0.,1.,20)
X = torch.vstack([x,x])
y = model_ws2(X, weights, bias_2d)
x1, x2 = torch.meshgrid(torch.linspace(0.,1.,20),torch.linspace(0.,1.,20), indexing='ij')
X2d = torch.vstack([x1, x2])
ax1.plot_surface(x1, x2, y, color="blue")
plt.show()
#loss_value = f(weights, bias)

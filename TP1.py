import torch
import matplotlib.pyplot as plt
from time import perf_counter

# exo1
# t = torch.randint(high=99, size=(100,))
t = torch.arange(100)
m= t.view(10, 10)
print('dernière ligne de m : ', m[-1, :])
print('deuxième colonne de m : ', m[:, 1])
print('sous_matrice de m ayant les lignes de indice impair et colonnes de indice pair : ', m[1::2, ::2])
n = torch.ones(5,5)
n = (m[::2, 1::2] - m[1::2, ::2])
print(n)
#exo 2
"""
- 5 * torch.tensor([1,2,3]) RESULTAT : [5,10,15]
- torch.tensor([1,2,3]) + torch.tensor([[1],[2],[3]]) RESULTAT: les dimensions ne sont pas compatible donc pytorch
fait le broadcasting(etendre la plus petite dimension), le premier tensor est 1D shape(3,) et le 2eme de shape (3,1)
et donc on obtient [ [2] , [4] , [6]]
- torch.tensor([[1,2],[3,4],[5,6]]) + torch.tensor([[1],[2],[3]]) RESULTAT : [ [2,3] , [5,6] ,[8,9] ]
- torch.tensor([[1,2],[3,4]]) + torch.tensor([[1],[2],[3]]) pas possible de faire le broadcasting
"""
#exo 3
t = torch.rand((1000, 2))
min = torch.min(t).item() # devrait etre dans l'intervalle [0,1]
print('la valeur minimal du tensor est : ', min)
max = torch.max(t).item()
print('la valeur maximale du tensor est : ', max)
moyenne = torch.mean(t).item() # devrait etre autour de 0,5 suivant la loi uniforme
print('la valeur moyenne du tensor est : ', moyenne)
ecart = torch.std(t).item() # autour de 0,29
print('la valeur ecart type  du tensor est : ', ecart)
# apres l'affichage je vois qui les valeurs respectent les valeurs de la loi uniforme

u = (torch.rand((1000, 2)) * 2) - 1
# print(torch.max(u).item()) pour verifier que les coef sont dans l'intervalle [-1,1[
# print(torch.min(u).item())
fig, ax = plt.subplots()
ax.scatter(t[:,0], t[:,1], c='blue')
ax.scatter(u[:,0], u[:,1], c='red')
plt.show()
#exo 4
t_norm = torch.normal(mean=0.0, std= 1.0, size=(1000,2))
print("L'ecart type de t_norm: ",torch.std(t_norm).item(),"La moyenne de t_norm : ", torch.mean(t_norm).item())
t1 = torch.normal(mean=1.0, std=2.0, size=(1000,2))
fig1, ax1 = plt.subplots()
ax1.scatter(t_norm[:,0], t_norm[:,1], c='black')
ax1.scatter(t1[:,0], t1[:,1], c='green')
plt.show()
'''l'intervalle change à chaque fois à cause de la generation des nombres aleatoire, mais le centre de
chaque cluster reste toujours autour de la moyenne, et l'ecart type etant la dispertion entre les point 
donc t1 est plus dispersé que t_norm'''
mat1 = torch.normal(mean=0.0, std=1.0, size=(5000,5000))
mat2 = torch.normal(mean=0.0, std=1.0, size=(5000,5000))
start = perf_counter()
torch.mm(mat1,mat2)
stop = perf_counter()
print("Le temps ecroulé à faire la multiplication est :", stop-start, "secondes")
#exo6
def mul_row (t):
    u = t.clone()  #u peut etre creer de differentes manieres mais j'ai pensée à celle ci en premier et c'est la plus facile
    for row in range(t.size(0)): # iterer seulement sur les lignes
        u[row, :] = (row+1) * t[row, :]
    return u

t = torch.full((4, 8), 2.0)
result = mul_row(t)
print(result)

def mul_row_fast(t):
    nb_rows = torch.arange(1, t.size(0) + 1).view(-1, 1) # l'idee est d'avoir un tensor a une colonne avec le meme nombre de ligne que t pour pouvoir faire la mult
    return nb_rows * t

comp_matrice = torch.rand(1000, 500)
slow_start = perf_counter()
slow = mul_row(comp_matrice)
slow_end = perf_counter()
print(f"Le temps necessaire pour faire la multiplication avec mult_row est de  {slow_end-slow_start} secondes")

fast_start = perf_counter()
fast = mul_row_fast(comp_matrice)
fast_end = perf_counter()
print(f"Le temps necessaire pour faire la multiplication avec mult_row_fast est de  {fast_end-fast_start} secondes")

#exo7
def generate_data_R1(nb_samples):
    data_inputs = torch.rand(nb_samples)
    weight = 0.4
    intercept = 0.5
    bruit = torch.normal(mean=0.0, std=0.01, size=(nb_samples,))
    data_outputs = weight * data_inputs + intercept + bruit
    return data_inputs, data_outputs

inputs , outputs = generate_data_R1(100)
fig2, ax2 = plt.subplots()
ax2.scatter(inputs, outputs)
plt.xlabel('X')
plt.ylabel('Y')


weight = torch.linalg.lstsq(outputs.unsqueeze(1), inputs.unsqueeze(1)).solution

def model_ws(x, weight, bias):
    return weight * x + bias

weight = torch.rand(1)
bias = torch.rand(1)
x = torch.linspace(0.,1.,20)
y = model_ws(x, weight, bias)
ax2.plot(x,y,c='blue')
plt.show()
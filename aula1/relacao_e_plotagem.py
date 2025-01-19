import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from sklearn.preprocessing import StandardScaler

from sklearn.cluster import KMeans

x = [20, 27, 21, 37, 46, 53, 55, 47, 52, 32, 39, 41, 39, 48, 48]    # idades
y = [1000, 1200, 2900, 1850, 900, 950, 2000, 2100, 3000, 5900, 4100, 5100, 7000, 5000, 6500] # salarios

grafico = px.scatter(x=x, y=y) # Cria um gráfico de dispersão
#grafico.write_image("meu_grafico.png") # Salva o gráfico em um arquivo

base_salario = np.array(list(zip(x, y))) # Cria uma matriz com as idades e salários

scaler_salario = StandardScaler()  # Cria um objeto para normalização dos dados
base_salario = scaler_salario.fit_transform(base_salario) # Normaliza os dados, pois se os salarios forem muito maior do que a idade, eles podem receber um peso maior durante o calculo

k_means_salario = KMeans(n_clusters=3) # Cria um objeto para o algoritmo de agrupamento KMeans
k_means_salario.fit(base_salario) # Treina o modelo

centroides = k_means_salario.cluster_centers_ # Pega os centroides dos grupos
print(centroides)

print(scaler_salario.inverse_transform(centroides)) # Desnormaliza os centroides

rotulos = k_means_salario.labels_ # Pega os rótulos dos grupos

print(rotulos)

grafico1 = px.scatter(x=base_salario[:,0], y=base_salario[:,1], color=rotulos) # Cria um gráfico de dispersão com os grupos coloridos
grafico1.write_image("grafico_colorido.png") # Salva o gráfico em um arquivo

grafico2 = px.scatter(x=centroides[:,0],y=centroides[:,1],size=[12,12,12])
grafico3 = go.Figure(data=grafico2.data + grafico1.data)

grafico3.write_image("grafico_colorido.png") # Salva o gráfico em um arquivo
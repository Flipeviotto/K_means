import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from sklearn.preprocessing import StandardScaler

from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs # Importa a função make_blobs
x_random, y_random = make_blobs(n_samples=200, centers= 5, random_state=1) # Cria um conjunto de dados, seria aleatório se não tivesse random_state=1
grafico = px.scatter(x=x_random[:,0],y=x_random[:,1]) # Cria um gráfico de dispersão
grafico.write_image(f"simulations\\inicio.png")


for i in range(10):
    kmeans = KMeans(n_clusters=5)
    kmeans.fit(x_random)

    rotulos = kmeans.labels_

    centroides = kmeans.cluster_centers_

    grafico1 = px.scatter(x=x_random[:,0],y=x_random[:,1],color=rotulos)
    grafico2 = px.scatter(x=centroides[:,0],y=centroides[:,1],size=[12,12,12,12,12])
    grafico3 = go.Figure(data =grafico1.data+grafico2.data)
    grafico3.write_image(f"simulations\\results{i}.png")


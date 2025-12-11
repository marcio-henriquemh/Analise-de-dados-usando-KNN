
#importando bibliotecas
import numpy as np
import matplotlib.pyplot as plt 
import pandas as pd
import statistics
import numpy as np
import pandas as pd
from tkinter import *
from tkinter import messagebox
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
dados = pd.read_csv(
    "/home/marciohenrique/UFS/Projeto Analise de Dados/comparecimento_abstencao_2024_SE.csv",sep=';',encoding='latin-1')

print(dados.head())
#calculando metricas

#faixa etária

# Converter faixa etária para número (pegando o primeiro número encontrado)
dados['DS_FAIXA_ETARIA'] = dados['DS_FAIXA_ETARIA'].str.extract(r'(\d+)')
# remove linhas que ficaram sem idade
dados = dados.dropna(subset=['DS_FAIXA_ETARIA'])
# converte para inteiro
dados['DS_FAIXA_ETARIA'] = dados['DS_FAIXA_ETARIA'].astype(int)
contagem_faixa_etaria=dados['DS_FAIXA_ETARIA'].value_counts()
print(contagem_faixa_etaria)
print("Média de idade da faixa etária:",
      statistics.mean(dados['DS_FAIXA_ETARIA']))

###################################################
#Escolaridade
escolaridade=dados['DS_GRAU_ESCOLARIDADE'].value_counts()
print(escolaridade)

print("Média do grau de escolaridade:",
      statistics.mean(dados['CD_GRAU_ESCOLARIDADE']))

####################################################
#cor de raça
contagem_cor_de_raca=dados['DS_COR_RACA'].value_counts()
print(contagem_cor_de_raca)
#

#algoritmo knn para entender a relacao da quantidade de pessoas negras, pardas com grau de instrução
# Seleção das características
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
# Algoritmo KNN para relação cor/raça x grau de instrução

caracteristicas = ["CD_COR_RACA", "DS_FAIXA_ETARIA"]
rotulo = "CD_GRAU_ESCOLARIDADE"  # usando coluna numérica existente

X = dados[caracteristicas]
y = dados[rotulo]

# Normalização
scaler = MinMaxScaler()
X_norm = scaler.fit_transform(X)

# Treino e teste
X_train, X_test, y_train, y_test = train_test_split(
    X_norm, y, test_size=0.3, random_state=42
)

# Modelo KNN
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)

# Previsões
y_pred = knn.predict(X_test)

print("Acurácia:", accuracy_score(y_test, y_pred))
print("Matriz de confusão:")
print(confusion_matrix(y_test, y_pred))


#plotando grafico

# Plotar a contagem em um gráfico de barras
plt.figure(figsize=(10, 6)) # Opcional: ajustar o tamanho da figura
contagem_cor_de_raca.plot(kind='bar', color='skyblue')

plt.title('Distribuição da Cor/Raça')
plt.xlabel('Cor/Raça')
plt.ylabel('Contagem')
plt.xticks(rotation=45) # Opcional: rotacionar os rótulos do eixo X para melhor visualização
plt.show()

#faixa etária
# Plotar a contagem em um gráfico de barras
plt.figure(figsize=(10, 6)) # Opcional: ajustar o tamanho da figura
contagem_faixa_etaria.plot(kind='bar', color='skyblue')

plt.title('Distribuição da faixa etaria')
plt.xlabel('idade')
plt.ylabel('Contagem')
plt.xticks(rotation=45) # Opcional: rotacionar os rótulos do eixo X para melhor visualização
plt.show()

#escolaridade
# Plotar a contagem em um gráfico de barras
plt.figure(figsize=(10, 6)) # Opcional: ajustar o tamanho da figura
escolaridade.plot(kind='bar', color='skyblue')

plt.title('Distribuição da escolaridade')
plt.xlabel('Grau de instrução')
plt.ylabel('Contagem')
plt.xticks(rotation=45) # Opcional: rotacionar os rótulos do eixo X para melhor visualização
plt.show()

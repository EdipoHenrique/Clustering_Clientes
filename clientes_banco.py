# -*- coding: utf-8 -*-

# Análise de Cluster - Segmentação de Clientes Bancários com Base em Comportamento Financeiro
# MBA em Data Science e Analytics USP ESALQ

# Édipo Henrique Teles Leite

#%% Instalando os pacotes

!pip install pandas
!pip install numpy
!pip install matplotlib
!pip install seaborn
!pip install plotly
!pip install scipy
!pip install scikit-learn
!pip install pingouin

#%% Importando os pacotes

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.cluster.hierarchy as sch
import scipy.stats as stats
from scipy.stats import zscore
from scipy.spatial.distance import pdist
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import KMeans
import pingouin as pg
import plotly.express as px 
import plotly.io as pio
pio.renderers.default='browser'

#%% Importando o banco de dados

dados_clientes = pd.read_excel('Clientes_banco.xlsx')
## Fonte: Kaagle - https://www.kaggle.com/datasets/khanmdsaifullahanjar/bank-user-dataset

# Estrutura do banco de dados

print(dados_clientes.info())

# Resumo estatístico das colunas numéricas
resumo = dados_clientes.describe()

# Verificando a presença de valores ausentes
valores_ausentes = dados_clientes.isnull().sum()


#%% Processo de limpeza da base

#Removendo valores ausentes
clientes = dados_clientes.dropna()

# Informações da base
print(clientes.info())

# Verificar e converter variáveis numéricas
clientes['Número_Pagamentos_Atrasados'] = pd.to_numeric(clientes['Número_Pagamentos_Atrasados'], errors='coerce')
clientes['Saldo_Mensal'] = pd.to_numeric(clientes['Saldo_Mensal'], errors='coerce')

#Verificando a conversão das variáveis (Número_Pagamentos_Atrasados e Sald_mensal) para numérico
informações = clientes.info()

#Criando o DataFrame apenas com as variáveis quantitativas que serão analisadas (9 variáveis)
    
variaveis_quantitativas = clientes[['Salário_Mensal', 'Número_Contas_Bancárias', 'Número_Cartão_Crédito', 'Taxa_Juros',
                                          'Atraso_de_data_de_vencimento', 'Número_Pagamentos_Atrasados', 'Consultas_Crédito_Numérico',
                                          'Taxa_Utilização_Crédito', 'Saldo_Mensal']]


# Resumo estatístico das colunas numéricas
resumo = clientes.describe()

# Verificando a presença de valores ausentes
valores_ausentes_quanti = variaveis_quantitativas.isnull().sum()

#Removendo valores ausentes
dados_clientes_cls = variaveis_quantitativas.dropna()

# Verificando a presença de valores ausentes
valores_ausentes_quanti = dados_clientes_cls.isnull().sum()


#%% Removendo os outliers

# Calcular a média e o desvio padrão
mean = dados_clientes_cls.mean()
std = dados_clientes_cls.std()

# Definir um limite (por exemplo, 3 desvios padrão da média)
limite_superior = mean + 3 * std
limite_inferior = mean - 3 * std

# Filtrar os dados para remover outliers
clientes_sem_outliers = dados_clientes_cls[(dados_clientes_cls < limite_superior) & (dados_clientes_cls > limite_inferior)].dropna()

# Resumo estatístico das colunas numéricas
resumo = clientes_sem_outliers.describe()


#%% Padronização dos dados - Z-score

# Padronização por meio do Z-Score para manter em unidades de medidas ou escalas iguais.
clientes_pad = clientes_sem_outliers.apply(zscore, ddof=1)

# Visualizando o resultado do procedimento na média e desvio padrão
print(round(clientes_pad.mean(), 2))
print(round(clientes_pad.std(), 2))


#%% Boxplot com as variáveis originais
#gráfico para identifica os outliers nos dados

plt.figure(figsize=(15, 9))
sns.boxplot(x='variable', y='value', data=pd.melt(clientes_pad))
plt.ylabel('Valores', fontsize=16)
plt.xlabel('Variáveis', fontsize=16)
plt.title('Distribuição das Variáveis', fontsize=18)  # Título para o gráfico
plt.xticks(rotation=45)  # Rotaciona os rótulos do eixo x para melhor legibilidade
plt.show()


#%% Identificação da quantidade de clusters

# Método Elbow para identificação do nº de clusters
## Elaborado com base na "WCSS": distância de cada observação para o centroide de seu cluster
## Quanto mais próximos entre si e do centroide, menores as distâncias internas
## Normalmente, busca-se o "cotovelo", ou seja, o ponto onde a curva "dobra"

elbow = []
K = range(1,5) # ponto de parada pode ser parametrizado manualmente
for k in K:
    kmeanElbow = KMeans(n_clusters=k, init='random', random_state=100).fit(clientes_sem_outliers)
    elbow.append(kmeanElbow.inertia_)
    
plt.figure(figsize=(16,8))
plt.plot(K, elbow, marker='o')
plt.xlabel('Nº Clusters', fontsize=16)
plt.xticks(range(1,5))
plt.ylabel('WCSS', fontsize=16)
plt.title('Método de Elbow', fontsize=16)
plt.show()


clientes_kmeans = KMeans(n_clusters=4, init='random', random_state=100).fit(clientes_sem_outliers)

# Gerando a variável para identificar os clusters gerados

kmeans_clusters = clientes_kmeans.labels_
clientes_sem_outliers['cluster_kmeans'] = kmeans_clusters
clientes_sem_outliers['cluster_kmeans'] = clientes_sem_outliers['cluster_kmeans'].astype('category')


#%% Análise de variância de um fator (ANOVA)

# Saldo_Mensal
pg.anova(dv='Saldo_Mensal', 
         between='cluster_kmeans', 
         data=clientes_sem_outliers,
         detailed=True).T

# Número de Contas
pg.anova(dv='Número_Contas_Bancárias', 
         between='cluster_kmeans', 
         data=clientes_sem_outliers,
         detailed=True).T

# Número de Cartão de Crédito
pg.anova(dv='Número_Cartão_Crédito', 
         between='cluster_kmeans', 
         data=clientes_sem_outliers,
         detailed=True).T

# Taxa de Juros
pg.anova(dv='Taxa_Juros',
         between='cluster_kmeans', 
         data=clientes_sem_outliers,
         detailed=True).T

# Número Pagamentos Atrasadoz
pg.anova(dv='Número_Pagamentos_Atrasados',
         between='cluster_kmeans', 
         data=clientes_sem_outliers,
         detailed=True).T

# Salário Mensal
pg.anova(dv='Salário_Mensal',
         between='cluster_kmeans', 
         data=clientes_sem_outliers,
         detailed=True).T

# Consulta de Crédito
pg.anova(dv='Consultas_Crédito_Numérico',
         between='cluster_kmeans', 
         data=clientes_sem_outliers,
         detailed=True).T
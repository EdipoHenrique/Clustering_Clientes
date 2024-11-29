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

# Verificando a presença de valores ausentes
valores_ausentes = clientes.isnull().sum()

# Informações da base
print(clientes.info())

#%% Criando um novo DataFrame

#Criando o DataFrame apenas com as variáveis que serão avaliadas + Nome do Cliente (7 variáveis)
    
variaveis_quantitativas = clientes[['Nome', 'Número_Contas_Bancárias', 'Número_Cartão_Crédito', 'Taxa_Juros',
                                          'Atraso_de_data_de_vencimento', 'Número_Pagamentos_Atrasados', 'Consultas_Crédito_Numérico']]


# Removendo linhas duplicadas com base na colunas abaixo:
clientes_duplicados = variaveis_quantitativas.drop_duplicates(subset=['Nome'])


# Informações da base
print(clientes_duplicados.info())


# Verificar e converter variáveis numéricas
clientes_duplicados['Número_Pagamentos_Atrasados'] = pd.to_numeric(clientes_duplicados['Número_Pagamentos_Atrasados'], errors='coerce')

#Verificando a conversão das variáveis (Número_Pagamentos_Atrasados) para numérico
print(clientes_duplicados.info())

# Resumo estatístico das colunas numéricas
resumo = clientes_duplicados.describe()

# Verificando a presença de valores ausentes
valores_ausentes = clientes_duplicados.isnull().sum()

#Removendo valores ausentes
clientes = clientes_duplicados.dropna()

# Verificando a presença de valores ausentes
valores_ausentes = clientes.isnull().sum()



#%% Processo de limpeza da base

# Para definir um parâmetro mais realista, podemos considerar dados globais e regionais. 
# No Brasil, por exemplo, o número médio de contas bancárias por pessoa foi de 5,5 contas em 2023. 
# No entanto, para evitar outliers, você pode definir um intervalo mais restrito, como entre 1 e 50 contas 
# ou 1 e 100 contas. Para a análise em questão, será realizado o filtro entre 1 e 100 contas.
# Essa abordagem ajudará a remover valores extremos e manter os dados mais consistentes e representativos.


# Filtrar os dados para manter apenas as linhas em que o número de contas e outras varáveis sejam entre 1 e 100

clientes_filtrado = clientes[
    (clientes['Número_Contas_Bancárias'] >= 1)      & (clientes['Número_Contas_Bancárias'] <= 100) &
    (clientes['Número_Cartão_Crédito'] >= 1)        & (clientes['Número_Cartão_Crédito'] <= 100) &
    (clientes['Taxa_Juros'] >= 1)                   & (clientes['Taxa_Juros'] <= 100) &
    (clientes['Atraso_de_data_de_vencimento'] >= 1) & (clientes['Atraso_de_data_de_vencimento'] <= 100) &
    (clientes['Número_Pagamentos_Atrasados'] >= 1)  & (clientes['Número_Pagamentos_Atrasados'] <= 100) &
    (clientes['Consultas_Crédito_Numérico'] >= 1)   & (clientes['Consultas_Crédito_Numérico'] <= 100)
]


print(clientes_filtrado.info())

# Resumo estatístico das colunas numéricas
resumo = clientes_filtrado.describe()


#%% Criar o DataFrame apenas com as variáveis do estudo

#DataFrame apenas com as variáveis quanti (6 variáveis) - retirando a variável nome
dados_clientes_cls = clientes_filtrado[['Número_Contas_Bancárias', 'Número_Cartão_Crédito', 'Taxa_Juros',
                                          'Atraso_de_data_de_vencimento', 'Número_Pagamentos_Atrasados', 'Consultas_Crédito_Numérico']]


print(dados_clientes_cls.info())

# Verificando a presença de valores ausentes
valores_ausentes = dados_clientes_cls.isnull().sum()


#%% Boxplot com as variáveis originais
#gráfico para identifica os outliers nos dados

plt.figure(figsize=(15, 9))
sns.boxplot(x='variable', y='value', data=pd.melt(dados_clientes_cls))
plt.ylabel('Valores', fontsize=16)
plt.xlabel('Variáveis', fontsize=16)
plt.title('Distribuição das Variáveis', fontsize=18)  # Título para o gráfico
plt.xticks(rotation=45)  # Rotaciona os rótulos do eixo x para melhor legibilidade
plt.show()

#%% Padronização dos dados - Z-score

# Padronização por meio do Z-Score para manter em unidades de medidas ou escalas iguais.
clientes_pad = dados_clientes_cls.apply(zscore, ddof=1)


# Verificar os valores máximos e mínimos dos Z-scores 
print(clientes_pad.max()) 
print(clientes_pad.min())

# Visualizando o resultado do procedimento na média e desvio padrão
print(round(clientes_pad.mean(), 2))
print(round(clientes_pad.std(), 2))


#%% Identificação da quantidade de clusters

# Método Elbow para identificação do nº de clusters
## Elaborado com base na "WCSS": distância de cada observação para o centroide de seu cluster
## Quanto mais próximos entre si e do centroide, menores as distâncias internas
## Normalmente, busca-se o "cotovelo", ou seja, o ponto onde a curva "dobra"

elbow = []
K = range(1,5) # ponto de parada pode ser parametrizado manualmente
for k in K:
    kmeanElbow = KMeans(n_clusters=k, init='random', random_state=100).fit(dados_clientes_cls)
    elbow.append(kmeanElbow.inertia_)
    
plt.figure(figsize=(16,8))
plt.plot(K, elbow, marker='o')
plt.xlabel('Nº Clusters', fontsize=16)
plt.xticks(range(1,5))
plt.ylabel('WCSS', fontsize=16)
plt.title('Método de Elbow', fontsize=16)
plt.show()

# Aplicando o método K-means
clientes_kmeans = KMeans(n_clusters=4, init='random', random_state=100).fit(dados_clientes_cls)

# Gerando a variável para identificar os clusters gerados

kmeans_clusters = clientes_kmeans.labels_
clientes_filtrado['cluster_kmeans'] = kmeans_clusters
clientes_filtrado['cluster_kmeans'] = clientes_filtrado['cluster_kmeans'].astype('category')


#%% Análise de variância de um fator (ANOVA)

#'Número_Contas_Bancárias'
#'Número_Cartão_Crédito', 
#'Taxa_Juros',
#'Atraso_de_data_de_vencimento', 
#'Número_Pagamentos_Atrasados', 
#'Consultas_Crédito_Numérico'

# Número de contas bancárias
pg.anova(dv='Número_Contas_Bancárias', 
         between='cluster_kmeans', 
         data=clientes_filtrado,
         detailed=True).T

# Número de cartões
pg.anova(dv='Número_Cartão_Crédito', 
         between='cluster_kmeans', 
         data=clientes_filtrado,
         detailed=True).T

# Taxa de Juros
pg.anova(dv='Taxa_Juros', 
         between='cluster_kmeans', 
         data=clientes_filtrado,
         detailed=True).T

# Atraso nos vencimentos
pg.anova(dv='Atraso_de_data_de_vencimento',
         between='cluster_kmeans', 
         data=clientes_filtrado,
         detailed=True).T

# Número de pagamentos atrasados
pg.anova(dv='Número_Pagamentos_Atrasados',
         between='cluster_kmeans', 
         data=clientes_filtrado,
         detailed=True).T

# Consulta de Crédito
pg.anova(dv='Consultas_Crédito_Numérico',
         between='cluster_kmeans', 
         data=clientes_filtrado,
         detailed=True).T


#%% 

from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler

# Padronizar os dados - JÁ PADRONIZADOS NA LINHA 153 PELO Z-SCORE
scaler = StandardScaler()
clientes_pad = scaler.fit_transform(clientes)



# Inicialize o modelo DBSCAN
db = DBSCAN(eps=0.5, min_samples=5)

# Ajuste o modelo aos dados padronizados
clusters = db.fit_predict(clientes_pad)

# clusters agora contém os rótulos dos clusters para cada ponto de dado


# Verifique a quantidade de clusters encontrados (excluindo ruído, rotulado como -1)
n_clusters = len(set(clusters)) - (1 if -1 in clusters else 0)

print(f'Número estimado de clusters: {n_clusters}')
print(f'Pontos rotulados como ruído: {(clusters == -1).sum()}')


import matplotlib.pyplot as plt

plt.scatter(clientes_pad[:, 0], clientes_pad[:, 1], c=clusters, cmap='rainbow')
plt.title('DBSCAN Clusters')
plt.show()


#%% Comparar com o K-means

from sklearn.cluster import KMeans

# Ajustar o modelo K-means
kmeans = KMeans(n_clusters=4)  # Suponha que você quer 4 clusters
kmeans_clusters = kmeans.fit_predict(clientes_pad)

# Visualização dos clusters K-means
plt.scatter(clientes_pad[:, 0], clientes_pad[:, 1], c=kmeans_clusters, cmap='rainbow')
plt.title('K-means Clusters')
plt.show()

if isinstance(clientes_pad, pd.DataFrame):
    clientes_pad = clientes_pad.values


import matplotlib.pyplot as plt

plt.scatter(clientes_pad[:, 0], clientes_pad[:, 1], c=kmeans_clusters, cmap='rainbow')
plt.title('K-means Clusters')
plt.xlabel('Primeira variável padronizada')
plt.ylabel('Segunda variável padronizada')
plt.show()


plt.scatter(clientes_pad[:, 0], clientes_pad[:, 1], c=kmeans_clusters, cmap='rainbow', alpha=0.5)
plt.title('K-means Clusters')
plt.xlabel('Primeira variável padronizada')
plt.ylabel('Segunda variável padronizada')
plt.show()


import seaborn as sns

sns.scatterplot(x=clientes_pad[:, 0], y=clientes_pad[:, 1], hue=kmeans_clusters, palette='viridis', alpha=0.5)
plt.title('K-means Clusters')
plt.xlabel('Primeira variável padronizada')
plt.ylabel('Segunda variável padronizada')
plt.show()



plt.figure(figsize=(12, 8))
sns.set_style("whitegrid")

sns.scatterplot(x=clientes_pad[:, 0], y=clientes_pad[:, 1], hue=kmeans_clusters, palette='Set2', alpha=0.7)
plt.title('K-means Clusters')
plt.xlabel('Primeira variável padronizada')
plt.ylabel('Segunda variável padronizada')
plt.show()





#%% PCA

from sklearn.decomposition import PCA

# Reduzir para 2 componentes principais
pca = PCA(n_components=2)
pca_result = pca.fit_transform(dados_clientes_cls)

plt.figure(figsize=(12, 8))
sns.scatterplot(x=pca_result[:, 0], y=pca_result[:, 1], hue=kmeans_clusters, palette='Set2', alpha=0.7)
plt.title('K-means Clusters (PCA)')
plt.xlabel('Primeira componente principal')
plt.ylabel('Segunda componente principal')
plt.show()


#%% Gráfico de Pares: 
# Um pair plot permite visualizar a relação entre cada par de variáveis. 
# É uma maneira eficaz de entender a distribuição dos dados e os clusters em várias dimensões.

import seaborn as sns
import pandas as pd

# Crie um DataFrame com os dados padronizados e os clusters
df = pd.DataFrame(clientes_pad, columns=[
    'Número_Contas_Bancárias',
    'Número_Cartão_Crédito',
    'Taxa_Juros',
    'Atraso_de_data_de_vencimento',
    'Número_Pagamentos_Atrasados',
    'Consultas_Crédito_Numérico'
])
df['Cluster'] = kmeans_clusters

sns.pairplot(df, hue='Cluster', palette='Set2')
plt.suptitle('K-means Clusters (Pair Plot)', y=1.02)
plt.show()


#%%  Gráfico 3D: 


import plotly.express as px

fig = px.scatter_3d(df, x='Número_Contas_Bancárias', y='Número_Cartão_Crédito', z='Taxa_Juros',
                    color='Cluster', opacity=0.7)
fig.update_layout(title='K-means Clusters (3D)', scene=dict(
    xaxis_title='Número_Contas_Bancárias',
    yaxis_title='Número_Cartão_Crédito',
    zaxis_title='Taxa_Juros'
))
fig.show()

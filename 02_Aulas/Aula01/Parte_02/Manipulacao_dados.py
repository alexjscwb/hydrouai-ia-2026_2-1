# -*- coding: utf-8 -*-
"""
Created on Fri Jun 27 15:46:32 2025

@author: afrod
"""

import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import os
from pathlib import Path

#%%

## Definindo o diretório de trabalho

dir_path = Path.cwd() / "02_Shapefiles"

print(dir_path)


#%%

## Importando dados geoespaciais

res_3M = gpd.read_file(dir_path / 'Reservatorio.shp')

bh = gpd.read_file(dir_path / 'bacia.shp') ## Bacia hidrográfica

flu = gpd.read_file(dir_path / 'Flu.gpkg') ## Estações fluviométricas selecionadas

plu = gpd.read_file(dir_path / 'Plu.gpkg') ## Estações Pluviométricas selecionadas

rios = gpd.read_file(dir_path / 'HIDROGRAFIA.shp') ## Hidrografia

#%%

## Conferindo o crs de cada arquivo shp

# Mostrar o CRS original
print("CRS original:")
print("Reservatório:", res_3M.crs)
print("Bacia:", bh.crs)
print("Fluviométricas:", flu.crs)
print("Pluviométricas:", plu.crs)
print("Hidrografia:", rios.crs)

# Definir o CRS alvo: WGS84 (EPSG:4326)
target_crs = "EPSG:4326"

# Reprojetar para WGS84, se necessário
if res_3M.crs != target_crs:
    res_3M = res_3M.to_crs(target_crs)

if bh.crs != target_crs:
    bh = bh.to_crs(target_crs)

if flu.crs != target_crs:
    flu = flu.to_crs(target_crs)

if plu.crs != target_crs:
    plu = plu.to_crs(target_crs)

if rios.crs != target_crs:
    rios = rios.to_crs(target_crs)

# Confirmar reprojeção
print("\nCRS após conversão:")
print("Reservatório:", res_3M.crs)
print("Bacia:", bh.crs)
print("Fluviométricas:", flu.crs)
print("Pluviométricas:", plu.crs)
print("Hidrografia:", rios.crs)

#%%

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines

# Criar o gráfico base
fig, ax = plt.subplots(figsize=(12, 10))

# 1. Bacia
bh.plot(ax=ax, color='lightgreen', edgecolor='black', linewidth=1, zorder=1)

# 2. Reservatório
res_3M.plot(ax=ax, color='darkblue')

# 4. Estações fluviométricas
flu.plot(ax=ax, color='red', markersize=120, marker='o', zorder=3)

# 5. Estações pluviométricas
plu.plot(ax=ax, color='black', markersize=80, marker='^', zorder=4)

# 3. Rios
rios.plot(ax=ax, color='blue', linewidth=1.5, zorder = 2)


# Criar legenda com símbolos
legend_elements = [
    mpatches.Patch(facecolor='lightgreen', edgecolor='black', label='Bacia'),
    mpatches.Patch(color='darkblue', label='Reservatório UHE'),
    mpatches.Patch(color='blue', label='Rios'),
    mlines.Line2D([], [], color='red', marker='o', linestyle='None',
                  markersize=10, label='Est. Fluviométricas'),
    mlines.Line2D([], [], color='black', marker='^', linestyle='None',
                  markersize=10, label='Est. Pluviométricas'),
]

# Ajustes finais
plt.title('Bacia Hidrográfica a montante da UHE Três Marias')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.legend(handles=legend_elements, loc='upper right')
plt.grid(True)
plt.tight_layout()
#plt.savefig('reserv_3m.png', dpi = 300)
plt.show()

#%%

## Upload das séries históricas de precipitação

# Estabelecendo o diretório e carregando os nomes dos arquivos
dir_series_p = Path.cwd() / '01_Dados' / 'Estações_Pluviometricas'

files_p = os.listdir(dir_series_p)

# Filtrando a partir das datas de interesse

start_date = '1999-10-01 00:00:00'
end_date = '2021-09-30 00:00:00'
date = pd.date_range(start_date, end_date, freq = 'D')

# Definindo função para manipulação da consistência dos dados
    # Criando uma função para selecionar as linhas com Nivel de Consistência = 2. Caso não tenha, mantemos o n 1
def selec_nivel_consist(group):
    if (group['NivelConsistencia'] == 2).any():
        return group[group['NivelConsistencia'] == 2]
    else:
        return group.head(1)
    
# Importando e organizando cada estação

all_files_P = pd.DataFrame({'data': date})
for file in files_p:
    
    file_path = dir_series_p / file

    # Verifica tamanho do arquivo
    if os.path.getsize(file_path) <= 1024:  # ~1 KB
        print(f"Arquivo vazio ignorado: {file}")
        continue

    df_p = pd.read_csv(file_path)
    df_p['DataHora'] = pd.to_datetime(df_p['DataHora'])
    df_p['ano'] = df_p['DataHora'].dt.year
    df_p['mes'] = df_p['DataHora'].dt.month
    df_p = df_p.drop(['DataHora'], axis = 1)
    df_p = df_p.melt(id_vars = ['ano', 'mes', 'NivelConsistencia'], var_name = 'dia', value_name = 'P_' + file[:7])
    df_p['dia'] = df_p['dia'].str[-2:]
    df_p['data'] = df_p['ano'].astype(str) + '-' + df_p['mes'].astype(str) + '-' + df_p['dia'].astype(str)
    df_p['data'] = pd.to_datetime(df_p['data'], errors = 'coerce')  # Trata datas inválidas como NaT
    df_p = df_p.drop(['dia','mes','ano'], axis = 1)
    df_p = df_p[['data', 'NivelConsistencia', 'P_' + file[:7]]]
    df_p = df_p.sort_values(
        ['data', 'NivelConsistencia'],
        ascending=[True, False]
    )
    
    df_p = df_p.drop_duplicates(
        subset='data',
        keep='first'
    )   
    df_p = df_p.drop('NivelConsistencia', axis = 1)
    all_files_P = all_files_P.merge(df_p, on = 'data', how = 'left')
    print(file)

#%%

# Diagrama de GANTT Precipitação
import matplotlib.dates as dt

df_P = all_files_P.set_index('data')
# Criar lista de períodos com dados válidos
good_ranges = []
for col_name in df_P.columns:
    col = df_P[col_name][2:]

    # Definir blocos de dados válidos
    start_mark = col.notnull() & col.shift().isnull()  # Início: número precedido por NaN
    end_mark = col.notnull() & col.shift(-1).isnull()  # Fim: número seguido por NaN

    start = col[start_mark].index
    end = col[end_mark].index

    for s, e in zip(start, end):
        good_ranges.append((col_name, s, e))

# Criar DataFrame com os períodos de dados válidos
good_ranges = pd.DataFrame(good_ranges, columns=['gauge', 'start', 'end'])

# Garantir que as datas estão no formato correto
good_ranges['start'] = pd.to_datetime(good_ranges['start'])
good_ranges['end'] = pd.to_datetime(good_ranges['end'])

# Criar um dicionário para converter os nomes das estações em índices
gauge_labels = good_ranges['gauge'].unique()
gauge_map = {gauge: i for i, gauge in enumerate(gauge_labels)}
good_ranges['gauge_index'] = good_ranges['gauge'].map(gauge_map)

# Criar figura
fig, ax = plt.subplots(figsize=(10, 8))

# Criar gráfico de barras horizontais
ax.barh(y=good_ranges['gauge_index'], 
        width=(dt.date2num(good_ranges['end']) - dt.date2num(good_ranges['start'])), 
        left=dt.date2num(good_ranges['start']), 
        color='g', alpha=0.7)

# Configurar rótulos do eixo Y
ax.set_yticks(range(len(gauge_labels)))
ax.set_yticklabels(gauge_labels)

# Ajustar formato do eixo X
ax.set_xlim([dt.date2num(min(good_ranges['start'].min(), pd.Timestamp("1999-10-01"))), 
             dt.date2num(pd.Timestamp("2021-09-30"))])

ax.xaxis_date()  # Formatar eixo X como datas
ax.set_xlabel("Ano")
ax.set_ylabel("Estação")
ax.set_title("Disponibilidade de Dados por Estação")

fig.tight_layout()
plt.show()

#%%

# Eliminando estações com + de X% de falhas diárias (Precipitação)

def missing_values(all_files_dataframe, threshold):
    """
    Remove colunas com mais de `threshold`% de valores ausentes.
    
    Parâmetros:
        all_files_dataframe: DataFrame de entrada
        threshold: porcentagem máxima de dados ausentes permitida por coluna
    
    Retorna:
        DataFrame sem as colunas que excedem o limite de valores ausentes
    """
    # Total de valores ausentes por coluna
    missing_counts = all_files_dataframe.isnull().sum()
    total_rows = all_files_dataframe.shape[0]
    
    # Percentual de valores ausentes
    missing_percent = (missing_counts / total_rows) * 100
    
    # Identificar colunas com mais de `threshold`% de valores faltantes
    cols_to_drop = missing_percent[missing_percent > threshold].index.tolist()
    
    # Remover essas colunas
    cleaned_df = all_files_dataframe.drop(columns=cols_to_drop)
    
    # Exibir informações
    print(f'Colunas removidas (>{threshold}% missing): {cols_to_drop}')
    print(f'Dimensões do novo DataFrame: {cleaned_df.shape}')
    
    return cleaned_df

all_files_P_clean = missing_values(all_files_P, threshold=5)


#%%

## Upload das séries históricas de Vazão

# Estabelecendo o diretório e carregando os nomes dos arquivos
dir_series_q = Path.cwd() / '01_Dados' / 'Estações_Fluviometricas'

files_q = os.listdir(dir_series_q)

# Importando e organizando cada estação

all_files_Q = pd.DataFrame({'data': date})

for file in files_q:
    
    file_path = dir_series_q / file

    # Verifica tamanho do arquivo
    if os.path.getsize(file_path) <= 1024:  # ~1 KB
        print(f"Arquivo vazio ignorado: {file}")
        continue
    
    # Leitura do arquivo
    df_q = pd.read_csv(file_path)
    
    # Organização das datas
    df_q['DataHora'] = pd.to_datetime(df_q['DataHora'])
    df_q['ano'] = df_q['DataHora'].dt.year
    df_q['mes'] = df_q['DataHora'].dt.month
    
    df_q = df_q.drop(['DataHora'], axis=1)
    
    # Transformação para formato longo
    df_q = df_q.melt(
        id_vars=['ano', 'mes', 'NivelConsistencia'],
        var_name='dia',
        value_name='Q_' + file[:8]
    )
    
    # Construção da data
    df_q['dia'] = df_q['dia'].str[-2:]
    
    df_q['data'] = (
        df_q['ano'].astype(str) + '-' +
        df_q['mes'].astype(str) + '-' +
        df_q['dia'].astype(str)
    )
    
    df_q['data'] = pd.to_datetime(
        df_q['data'],
        errors='coerce'
    )
    
    # Seleção das colunas de interesse
    df_q = df_q.drop(['dia', 'mes', 'ano'], axis=1)
    
    df_q = df_q[
        ['data', 'NivelConsistencia', 'Q_' + file[:8]]
    ]
    
    # Prioriza registros com NivelConsistencia = 2
    df_q = df_q.sort_values(
        ['data', 'NivelConsistencia'],
        ascending=[True, False]
    )
    
    # Mantém apenas um registro para cada data
    df_q = df_q.drop_duplicates(
        subset='data',
        keep='first'
    )
    
    # Remove a coluna de consistência
    df_q = df_q.drop(
        'NivelConsistencia',
        axis=1
    )
    
    # Incorpora a estação ao DataFrame geral
    all_files_Q = all_files_Q.merge(
        df_q,
        on='data',
        how='left'
    )
    
    print(file)
#%%

# Eliminando estações com + de X% de falhas diárias (Vazão)

def missing_values(all_files_dataframe, threshold):
    """
    Remove colunas com mais de `threshold`% de valores ausentes.
    
    Parâmetros:
        all_files_dataframe: DataFrame de entrada
        threshold: porcentagem máxima de dados ausentes permitida por coluna
    
    Retorna:
        DataFrame sem as colunas que excedem o limite de valores ausentes
    """
    # Total de valores ausentes por coluna
    missing_counts = all_files_dataframe.isnull().sum()
    total_rows = all_files_dataframe.shape[0]
    
    # Percentual de valores ausentes
    missing_percent = (missing_counts / total_rows) * 100
    
    # Identificar colunas com mais de `threshold`% de valores faltantes
    cols_to_drop = missing_percent[missing_percent > threshold].index.tolist()
    
    # Remover essas colunas
    cleaned_df = all_files_dataframe.drop(columns=cols_to_drop)
    
    # Exibir informações
    print(f'Colunas removidas (>{threshold}% missing): {cols_to_drop}')
    print(f'Dimensões do novo DataFrame: {cleaned_df.shape}')
    
    return cleaned_df

all_files_Q_clean = missing_values(all_files_Q, threshold=5)

#%% Importando estação de interesse (Vazão naturalizada ONS)

# Estabelecendo o diretório
dir_serie_ons = Path.cwd() / '01_Dados' / 'Estação_ONS'

# Criar DataFrame base com todas as datas
df_base = pd.DataFrame({'data': date})

# Importando vazão afluente
df_ons = pd.read_csv(dir_serie_ons / 'Vazão_Afluente.csv')

# Convertendo data
df_ons['data'] = pd.to_datetime(
    df_ons['data'],
    errors='coerce'
)

# Fazer merge com df_ons
df_ons = df_base.merge(
    df_ons,
    on='data',
    how='left'
)

#%% Plotando Série Histórica

plt.figure(figsize=(14, 6))

plt.plot(
    df_ons['data'],
    df_ons['Q_Afluente'],
    label='Vazão Afluente ONS',
    color='blue'
)

# Formatação
plt.title('Série Histórica da Vazão Afluente - ONS')
plt.xlabel('Data')
plt.ylabel('Vazão (m³/s)')
plt.grid(True)
plt.legend()
plt.tight_layout()

plt.show()

#%% 

## Justanto todos os conjuntos de dados

from functools import reduce

dfs_to_merge = [df_ons, all_files_P_clean, all_files_Q_clean]

all_files_final = pd.DataFrame({'data': date})
all_files_final = reduce(lambda left, right: left.merge(right, on='data', how='left'), [all_files_final] + dfs_to_merge)

#%% Visualização de dados das outras estações

import matplotlib.dates as mdates

# all_files_final.reset_index(inplace=True)

all_files_final['data'] = pd.to_datetime(all_files_final['data'])

# Diretório para salvar as figuras
dir_fig = Path.cwd() / 'Figuras'

# Cria a pasta caso ela não exista
dir_fig.mkdir(parents=True, exist_ok=True)

# Plotagem de série histórica (e.g., P_2044053)
plt.figure(figsize=(12, 6))

plt.plot(
    all_files_final['data'],
    all_files_final['P_2044053'],
    label='P_2044053'
)

# Melhorando o eixo X
plt.gca().xaxis.set_major_formatter(
    mdates.DateFormatter('%Y-%m-%d')
)

plt.gca().xaxis.set_major_locator(
    mdates.AutoDateLocator()
)

plt.xticks(rotation=45)
plt.xlabel('Data')
plt.ylabel('P (mm)')
plt.title('Time Series Plot - P_2044053')
plt.legend()
plt.tight_layout()

# Salvando a figura
plt.savefig(
    dir_fig / 'P_2044053.png',
    dpi=300
)

plt.show()
#%%

# Análise de correlação e definição de estação Plu e Flu para a modelagem

"""
    
    Com a análise de correlação, iremos optar por estações que apresentam maior relação com a estação de interesse (ONS).
    Além disso, optaremos por uma boa representatividade espacial (pegando os principais rios)
    
"""

# 1. Calcular a matriz de correlação (excluindo a coluna 'data')
corr_matrix = all_files_final.drop(columns='data').corr()

# 2. Selecionar correlações com a coluna Q_afluente
corr_afluente = corr_matrix['Q_Afluente']

# 3. Filtrar apenas correlações maiores que X (excluindo a própria coluna Q_afluente)
high_corr_cols = corr_afluente[corr_afluente > 0.3].drop('Q_Afluente').index.tolist()

# 4. Criar novo DataFrame com apenas essas colunas e a coluna Q_afluente
selected_df = all_files_final[['data', 'Q_Afluente'] + high_corr_cols]

## Plotando e avaliando espacialmente

columns = selected_df.columns[2:]

est = columns.str.extract(r'_(\d+)$')[0] # Seleciona apenas os números encontrados após o primeiro _

## Selecionandos as estações no arquivo shapefile

flu['CODIGO'] = flu['CODIGO'].astype(str)
flu_selecionadas = flu[flu['CODIGO'].isin(est)]

plu['CODIGO'] = plu['CODIGO'].astype(str)
plu_selecionadas = plu[plu['CODIGO'].isin(est)]

## Plotando

# Criar o gráfico base
fig, ax = plt.subplots(figsize=(12, 10))

# 1. Bacia
bh.plot(ax=ax, color='lightgreen', edgecolor='black', linewidth=1)

# 2. Reservatório
res_3M.plot(ax=ax, color='darkblue')

# 3. Rios
rios.plot(ax=ax, color='blue', linewidth=1.5)

# 4. Estações fluviométricas
flu_selecionadas.plot(ax=ax, color='red', markersize=80, marker='o')

# 5. Estações pluviométricas
plu_selecionadas.plot(ax=ax, color='black', markersize=80, marker='^')

# Criar legenda com símbolos
legend_elements = [
    mpatches.Patch(facecolor='lightgreen', edgecolor='black', label='Bacia'),
    mpatches.Patch(color='darkblue', label='Reservatório UHE'),
    mpatches.Patch(color='blue', label='Rios'),
    mlines.Line2D([], [], color='red', marker='o', linestyle='None',
                  markersize=10, label='Est. Fluviométricas'),
    mlines.Line2D([], [], color='black', marker='^', linestyle='None',
                  markersize=10, label='Est. Pluviométricas'),
]

# Ajustes finais
plt.title('Bacia Hidrográfica a montante da UHE Três Marias')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.legend(handles=legend_elements, loc='upper right')
plt.grid(True)
plt.tight_layout()
plt.show()
#%%

# Avaliando de acordo com a representatividade espacial, optamos por selecionar as seguintes estações

est_final = [
    '1944059', '1944063', '1945002', '1945008', '1946000',
    '2044008', '2044016', '2044024', '2044042', '2045002',
    '2046013', '40032000', '40050000', '40060001', '40100000',
    '40185000', '40330000', '40800001', '40850000', '40930000'
]

# Selecionando no shapefile

## Selecionandos as estações no arquivo shapefile

flu['CODIGO'] = flu['CODIGO'].astype(str)
flu_selecionadas_final = flu[flu['CODIGO'].isin(est_final)]

plu['CODIGO'] = plu['CODIGO'].astype(str)
plu_selecionadas_final = plu[plu['CODIGO'].isin(est_final)]

## Plotando

# Criar o gráfico base
fig, ax = plt.subplots(figsize=(12, 10))

# 1. Bacia
bh.plot(ax=ax, color='lightgreen', edgecolor='black', linewidth=1)

# 2. Reservatório
res_3M.plot(ax=ax, color='darkblue')

# 3. Rios
rios.plot(ax=ax, color='blue', linewidth=1.5)

# 4. Estações fluviométricas
flu_selecionadas_final.plot(ax=ax, color='red', markersize=80, marker='o')

# 5. Estações pluviométricas
plu_selecionadas_final.plot(ax=ax, color='black', markersize=80, marker='^')

# Criar legenda com símbolos
legend_elements = [
    mpatches.Patch(facecolor='lightgreen', edgecolor='black', label='Bacia'),
    mpatches.Patch(color='darkblue', label='Reservatório UHE'),
    mpatches.Patch(color='blue', label='Rios'),
    mlines.Line2D([], [], color='red', marker='o', linestyle='None',
                  markersize=10, label='Est. Fluviométricas'),
    mlines.Line2D([], [], color='black', marker='^', linestyle='None',
                  markersize=10, label='Est. Pluviométricas'),
]

# Ajustes finais
plt.title('Bacia Hidrográfica a montante da UHE Três Marias')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.legend(handles=legend_elements, loc='upper right')
plt.grid(True)
plt.tight_layout()
# Adicionando o código das estações no mapa
for x, y, label in zip(flu_selecionadas_final.geometry.x, flu_selecionadas_final.geometry.y, flu_selecionadas_final['CODIGO']):
    ax.text(x, y, label, fontsize=12, ha='right', va='bottom')

for x, y, label in zip(plu_selecionadas_final.geometry.x, plu_selecionadas_final.geometry.y, plu_selecionadas_final['CODIGO']):
    ax.text(x, y, label, fontsize=12, ha='left', va='bottom')
    
plt.show()

#%% Selecionando do arquivos dataframe apenas as estações escolhidas

# Verificar as colunas disponíveis no DataFrame
colunas_sh = all_files_final.columns

# Identificar colunas que terminam com os códigos da lista est_final
colunas_selecionadas = [
    col for col in colunas_sh
    if any(col.endswith('_' + codigo) for codigo in est_final)
]

# Adicionar as colunas 'data' e 'Q_afluente'
colunas_finais = ['data', 'Q_Afluente'] + colunas_selecionadas

# Selecionar o subconjunto final do DataFrame
df_final_selecionado = all_files_final[colunas_finais]
print(df_final_selecionado.head())

#%%# Análise exploratória

## Detect Outliers using IQR Method
def detect_outliers(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers = df[(df[column] < lower_bound) | (df[column] > upper_bound)]
    return outliers

# Aplicar para colunas numéricas
numerical_columns = df_final_selecionado.select_dtypes(include=['float64']).columns
outliers_summary = {col: detect_outliers(df_final_selecionado, col) for col in numerical_columns}

# Apresentar contagem de outliers

outlier_counts = {col: len(outliers_summary[col]) for col in numerical_columns}
print("Outlier Counts by Column:")
print(outlier_counts)

# %% Visualização dos dados
import seaborn as sns

# Boxplot para outliers de precipitação
plt.figure(figsize=(12, 6))
sns.boxplot(data=df_final_selecionado[numerical_columns[1:12]])
plt.xticks(rotation=45)
plt.title('Boxplot Precipitação Bacia')
plt.show()

# Boxplot para outliers de vazão
plt.figure(figsize=(12, 6))
sns.boxplot(data=df_final_selecionado[numerical_columns[12:]])
plt.xticks(rotation=45)
plt.title('Boxplot Vazão Bacia')
plt.show()

# Boxplot para outliers Q_ONS
plt.figure(figsize=(12, 6))
sns.boxplot(data=df_final_selecionado[numerical_columns[0]])
plt.xticks(rotation=45)
plt.title('Boxplot Vazão ONS')
plt.show()

#%%

# Salvando os arquivos shapefile e as séries históricas

# Diretório de saída
dir_saida = Path.cwd() / '04. Dados finais'

# Cria a pasta caso ela ainda não exista
dir_saida.mkdir(parents=True, exist_ok=True)


## Série histórica

# Salvando como CSV
df_final_path = dir_saida / 'series_selecionadas.csv'

df_final_selecionado.to_csv(
    df_final_path,
    index=False
)


## Shapefiles

flu_shp_path = dir_saida / 'flu_selecionadas.shp'
plu_shp_path = dir_saida / 'plu_selecionadas.shp'

flu_selecionadas_final.to_file(
    flu_shp_path,
    encoding='utf-8'
)

plu_selecionadas_final.to_file(
    plu_shp_path,
    encoding='utf-8'
)
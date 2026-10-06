"""Curso Hydro-UAI — Dia 2
Análise exploratória e seleção de variáveis.

Etapas:
1. Correlação de Pearson para horizontes T+1, T+3 e T+7 dias
2. Correlação cruzada entre estações fluviométricas e Q_Afluente
3. Informação Mútua para horizontes T+1, T+3 e T+7 dias
4. PCA
"""

# %% 1. Importar bibliotecas
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.feature_selection import mutual_info_regression
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


# %% 2. Definir arquivo de entrada e parâmetros
PASTA = Path(__file__).resolve().parent
ARQUIVO = PASTA / "series_preenchidas.csv"

# Horizontes de previsão avaliados (dias)
HORIZONTES = [1, 3, 7]

# Maior lag testado na correlação cruzada
MAX_LAG = 15


# %% 3. Carregar e organizar o dataset
df = pd.read_csv(ARQUIVO)

df["data"] = pd.to_datetime(df["data"])
df = df.sort_values("data").set_index("data")

# Identificação automática das variáveis
colunas_precipitacao = [c for c in df.columns if c.startswith("P_")]
colunas_vazao = [
    c for c in df.columns
    if c.startswith("Q_") and c != "Q_Afluente"
]

# Variáveis disponíveis no instante t
preditores = colunas_vazao + colunas_precipitacao

print("Período:", df.index.min(), "até", df.index.max())
print("Número de observações:", len(df))
print("Estações pluviométricas:", len(colunas_precipitacao))
print("Estações fluviométricas:", len(colunas_vazao))


# %% 4. Correlação de Pearson para T+1, T+3 e T+7
# Para cada horizonte T, calcula:
#
# Corr[X(t), Q_Afluente(t + T)]
#
# Assim, todas as variáveis são avaliadas como possíveis preditoras
# da vazão afluente futura.

pearson_horizontes = pd.DataFrame(index=preditores)

for T in HORIZONTES:
    target = df["Q_Afluente"].shift(-T)

    pearson_horizontes[f"T+{T}"] = [
        df[variavel].corr(target)
        for variavel in preditores
    ]

print("\nCorrelação de Pearson com a Q_Afluente futura:")
print(pearson_horizontes)


# %% 5. Visualizar Pearson nos três horizontes
# O heatmap facilita a comparação simultânea entre variáveis e horizontes.

plt.figure(figsize=(8, 9))

sns.heatmap(
    pearson_horizontes,
    annot=True,
    fmt=".2f",
    cmap="coolwarm",
    center=0,
    vmin=-1,
    vmax=1
)

plt.xlabel("Horizonte de previsão")
plt.ylabel("Variável em t")
plt.title("Pearson com Q_Afluente futura")
plt.tight_layout()
plt.show()


# %% 6. Pearson por horizonte - ranking das variáveis
# Um gráfico para cada horizonte facilita observar como a relação linear
# se altera à medida que aumenta a antecedência da previsão.

for T in HORIZONTES:

    serie = pearson_horizontes[f"T+{T}"].sort_values(ascending=False)

    plt.figure(figsize=(9, 6))

    sns.barplot(
        x=serie.values,
        y=serie.index
    )

    plt.axvline(0, color="black", linewidth=0.8)
    plt.xlabel("Correlação de Pearson")
    plt.ylabel("Variável em t")
    plt.title(f"Pearson com Q_Afluente(t+{T})")
    plt.xlim(-1, 1)
    plt.tight_layout()
    plt.show()


# %% 7. Correlação cruzada - estações fluviométricas x Q_Afluente
# Aqui a pergunta é diferente da análise anterior.
#
# Para cada estação fluviométrica, calcula:
#
# Corr[Q_estacao(t - lag), Q_Afluente(t)]
#
# Lag positivo significa que relacionamos uma vazão observada antes
# na estação com a vazão afluente atual.
#
# Exemplo:
# lag = 3 -> Corr[Q_estacao(t-3), Q_Afluente(t)]

lags = range(0, MAX_LAG + 1)

correlacao_cruzada = pd.DataFrame(index=lags)

for estacao in colunas_vazao:
    correlacao_cruzada[estacao] = [
        df[estacao].shift(lag).corr(df["Q_Afluente"])
        for lag in lags
    ]

correlacao_cruzada.index.name = "lag_dias"

print("\nCorrelação cruzada:")
print(correlacao_cruzada)


# %% 8. Visualizar correlação cruzada
plt.figure(figsize=(11, 7))

for estacao in colunas_vazao:
    plt.plot(
        correlacao_cruzada.index,
        correlacao_cruzada[estacao],
        marker="o",
        label=estacao
    )

plt.axhline(0, color="black", linewidth=0.8)
plt.xlabel("Lag (dias)")
plt.ylabel("Correlação de Pearson")
plt.title("Correlação cruzada: estações fluviométricas x Q_Afluente")
plt.legend(title="Estação", bbox_to_anchor=(1.02, 1), loc="upper left")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()


# %% 9. Lag de maior correlação para cada estação
resumo_lags = []

for estacao in colunas_vazao:
    lag_max = correlacao_cruzada[estacao].idxmax()
    corr_max = correlacao_cruzada.loc[lag_max, estacao]

    resumo_lags.append({
        "estacao": estacao,
        "lag_maior_correlacao_dias": lag_max,
        "correlacao_maxima": corr_max
    })

resumo_lags = pd.DataFrame(resumo_lags)
resumo_lags = resumo_lags.sort_values(
    "correlacao_maxima",
    ascending=False
)

print("\nLag de maior correlação por estação:")
print(resumo_lags.to_string(index=False))


# %% 10. Informação Mútua para T+1, T+3 e T+7
# Para cada horizonte T:
#
# X(t) -> Q_Afluente(t + T)
#
# Pearson mede associação linear.
# Informação Mútua consegue detectar dependências mais gerais,
# inclusive relações não lineares.

mi_horizontes = pd.DataFrame(index=preditores)

for T in HORIZONTES:

    dados = df[preditores].copy()
    dados["target"] = df["Q_Afluente"].shift(-T)
    dados = dados.dropna()

    X = dados[preditores]
    y = dados["target"]

    mi_scores = mutual_info_regression(
        X,
        y,
        discrete_features=False,
        random_state=42
    )

    mi_horizontes[f"T+{T}"] = mi_scores

print("\nInformação Mútua com a Q_Afluente futura:")
print(mi_horizontes)


# %% 11. Visualizar Informação Mútua nos três horizontes
plt.figure(figsize=(8, 9))

sns.heatmap(
    mi_horizontes,
    annot=True,
    fmt=".2f",
    cmap="viridis"
)

plt.xlabel("Horizonte de previsão")
plt.ylabel("Variável em t")
plt.title("Informação Mútua com Q_Afluente futura")
plt.tight_layout()
plt.show()


# %% 12. Informação Mútua por horizonte - ranking das variáveis
for T in HORIZONTES:

    serie = mi_horizontes[f"T+{T}"].sort_values(ascending=False)

    plt.figure(figsize=(9, 6))

    sns.barplot(
        x=serie.values,
        y=serie.index
    )

    plt.xlabel("Informação Mútua")
    plt.ylabel("Variável em t")
    plt.title(f"Informação Mútua com Q_Afluente(t+{T})")
    plt.tight_layout()
    plt.show()


# %% 13. Comparação da perda de informação com o horizonte
# Média entre todas as estações de cada tipo.
# Serve como síntese exploratória da redução de associação/informação
# à medida que o horizonte de previsão aumenta.

resumo_horizontes = pd.DataFrame({
    "Pearson médio - Vazão": [
        pearson_horizontes.loc[colunas_vazao, f"T+{T}"].mean()
        for T in HORIZONTES
    ],
    "Pearson médio - Precipitação": [
        pearson_horizontes.loc[colunas_precipitacao, f"T+{T}"].mean()
        for T in HORIZONTES
    ],
    "MI médio - Vazão": [
        mi_horizontes.loc[colunas_vazao, f"T+{T}"].mean()
        for T in HORIZONTES
    ],
    "MI médio - Precipitação": [
        mi_horizontes.loc[colunas_precipitacao, f"T+{T}"].mean()
        for T in HORIZONTES
    ]
}, index=HORIZONTES)

resumo_horizontes.index.name = "Horizonte_dias"

print("\nResumo por horizonte:")
print(resumo_horizontes)


# %% 14. PCA - Padronização
# PCA não utiliza um alvo futuro: seu objetivo é identificar a estrutura
# de redundância entre as variáveis preditoras.
#
# Como PCA é sensível à escala, os dados são padronizados.

X_pca_raw = df[preditores].copy()

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_pca_raw)


# %% 15. PCA - Variância explicada
pca = PCA()
X_pca = pca.fit_transform(X_scaled)

variancia_acumulada = np.cumsum(
    pca.explained_variance_ratio_
) * 100

n_components = np.argmax(
    variancia_acumulada >= 95
) + 1

print(
    f"\nPCA: {n_components} componentes explicam "
    "pelo menos 95% da variância."
)

plt.figure(figsize=(8, 5))

plt.plot(
    range(1, len(variancia_acumulada) + 1),
    variancia_acumulada,
    marker="o"
)

plt.axhline(
    95,
    linestyle="--",
    label="95% da variância"
)

plt.xlabel("Número de componentes principais")
plt.ylabel("Variância explicada acumulada (%)")
plt.title("PCA - Variância explicada acumulada")
plt.grid(alpha=0.3)
plt.legend()
plt.tight_layout()
plt.show()


# %% 16. PCA - Loadings
loadings = pd.DataFrame(
    pca.components_[:n_components, :],
    columns=X_pca_raw.columns,
    index=[
        f"PC{i+1}"
        for i in range(n_components)
    ]
)

print("\nVariáveis com maior contribuição em cada componente:")

for componente in loadings.index:

    principais = (
        loadings.loc[componente]
        .abs()
        .sort_values(ascending=False)
        .head(5)
    )

    print(
        f"{componente}: "
        + ", ".join(principais.index)
    )

plt.figure(
    figsize=(10, max(6, len(X_pca_raw.columns) * 0.35))
)

sns.heatmap(
    loadings.T,
    cmap="coolwarm",
    center=0,
    annot=True,
    fmt=".2f"
)

plt.title("PCA - Loadings das variáveis")
plt.xlabel("Componente principal")
plt.ylabel("Variável")
plt.tight_layout()
plt.show()


# %% 17. PCA - Espaço das duas primeiras componentes
plt.figure(figsize=(9, 7))

plt.scatter(
    X_pca[:, 0],
    X_pca[:, 1],
    alpha=0.25
)

components = pca.components_[:2, :]

for i, variavel in enumerate(X_pca_raw.columns):

    plt.arrow(
        0,
        0,
        components[0, i] * 5,
        components[1, i] * 5,
        head_width=0.15,
        alpha=0.7
    )

    plt.text(
        components[0, i] * 5.3,
        components[1, i] * 5.3,
        variavel,
        fontsize=8,
        ha="center"
    )

plt.xlabel(
    f"PC1 ({pca.explained_variance_ratio_[0] * 100:.1f}%)"
)
plt.ylabel(
    f"PC2 ({pca.explained_variance_ratio_[1] * 100:.1f}%)"
)

plt.title("PCA - Espaço das duas primeiras componentes")
plt.axhline(0, linewidth=0.5)
plt.axvline(0, linewidth=0.5)
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()

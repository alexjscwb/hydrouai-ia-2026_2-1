# -*- coding: utf-8 -*-
"""
Created on Thu Jul 10 14:00:03 2025

@author: Perdigão
"""

import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset, random_split
from scipy.interpolate import interp1d
import win32com.client
import os
import geopandas as gpd
import random
import webbrowser

#%% Manter Reprodutibilidade
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True  # para redes determinísticas
    torch.backends.cudnn.benchmark = False     # desativa otimizações não determinísticas

set_seed(42)  # Define a seed fixa

#%% === 1. Carregamento e Preparação dos Dados ===
data = pd.read_csv(os.path.abspath(r"05. Arquivos de Texto\dataset_filled.csv"))
data['data'] = pd.to_datetime(data['data'])
data.set_index('data', inplace=True)

# Criar colunas de saída separadas para T+1 a T+7
for t in range(1, 8):
    data[f'Q_t+{t}'] = data['Q_Afluente'].shift(-t)

# Remover linhas com NaN
data = data.dropna().copy()

# Definir features
X = data.drop(columns=['Q_Afluente'] + [f'Q_t+{t}' for t in range(1, 8)]).values
X_tensor = torch.tensor(X, dtype=torch.float32)

#%%


# === 2. Definição da classe MLP ===
class MLP(nn.Module):
    def __init__(self, input_size, hidden_layers, neurons_per_layer):
        super(MLP, self).__init__()
        layers = [nn.Linear(input_size, neurons_per_layer), nn.ReLU()]
        for _ in range(hidden_layers - 1):
            layers.extend([nn.Linear(neurons_per_layer, neurons_per_layer), nn.ReLU()])
        layers.append(nn.Linear(neurons_per_layer, 1))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

# === 3. Treinamento de 7 modelos, um para cada T+1 a T+7 ===
modelos = []
resultados_pred = []
historicos_loss = []

epochs = 100  # número de épocas

for t in range(1, 8):
    y = data[f'Q_t+{t}'].values
    y_tensor = torch.tensor(y, dtype=torch.float32).view(-1, 1)
    dataset = TensorDataset(X_tensor, y_tensor)

    train_size = int(0.8 * len(dataset))
    generator = torch.Generator().manual_seed(42 + t)  # seed diferente por horizonte t

    train_dataset, val_dataset = random_split(dataset, [train_size, len(dataset) - train_size], generator=generator)
    train_loader = DataLoader(train_dataset, batch_size=1000, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=1000)

    model = MLP(input_size=X.shape[1], hidden_layers=3, neurons_per_layer=128)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    train_losses = []
    val_losses = []

    for epoch in range(epochs):
        model.train()
        batch_losses = []
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            loss = criterion(model(X_batch), y_batch)
            loss.backward()
            optimizer.step()
            batch_losses.append(loss.item())
        train_losses.append(np.mean(batch_losses))

        # Validação
        model.eval()
        with torch.no_grad():
            val_loss = np.mean([criterion(model(X_val), y_val).item() for X_val, y_val in val_loader])
        val_losses.append(val_loss)

    # Salvar histórico e modelo
    historicos_loss.append((train_losses, val_losses))
    modelos.append(model)

    # Previsões para todo o conjunto
    model.eval()
    with torch.no_grad():
        y_pred = model(X_tensor).numpy().flatten()
        resultados_pred.append(y_pred)

#%% 4. Análise dos Treinamentos

fig, axs = plt.subplots(2, 4, figsize=(18, 8))
axs = axs.flatten()

for t in range(7):
    train_loss, val_loss = historicos_loss[t]
    axs[t].plot(range(1, epochs + 1), train_loss, label='Train Loss', color='blue')
    axs[t].plot(range(1, epochs + 1), val_loss, label='Validation Loss', color='orange')
    axs[t].set_title(f'Modelo T+{t+1}')
    axs[t].set_xlabel('Epoch')
    axs[t].set_ylabel('Loss')
    axs[t].grid(True)
    axs[t].legend()

# Remover subplot vazio (posição 8)
fig.delaxes(axs[7])

fig.suptitle("Histórico de Treinamento - Modelos MLP (T+1 a T+7)", fontsize=16)
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.show()


#%% === Plotar série observada completa com destaque no índice escolhido ===
def plot_serie_com_indice(idx, data, col="Q_t+1"):
    """
    Plota a série temporal observada destacando o índice de interesse.
    
    Parâmetros:
    - idx: índice t₀ onde começa a previsão
    - data: DataFrame com as colunas Q_t+1 até Q_t+7
    - col: coluna a ser usada para mostrar a série observada
    """
    plt.figure(figsize=(12, 4))
    plt.plot(data[col].values, label=f"Série observada ({col})", color="gray")
    plt.axvline(x=idx, color="red", linestyle="--", label=f"t₀ = {idx}")
    plt.title("Série Observada com Destaque no t₀")
    plt.xlabel("Índice temporal")
    plt.ylabel("Q [m³/s]")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# Exemplo de uso
plot_serie_com_indice(3368, data, col="Q_t+1")

#%% === 5. Função para plotar o hidrograma previsto a partir do índice ===


def plot_hidrograma_7modelos(idx, resultados_pred, data):
    """
    Plota o hidrograma previsto por 7 modelos independentes.
    """
    if idx < 0 or idx >= len(data):
        print("Índice fora do intervalo válido.")
        return

    observados = [data[f'Q_t+{t}'].iloc[idx] for t in range(1, 8)]
    previstos = [resultados_pred[t - 1][idx] for t in range(1, 8)]

    plt.figure(figsize=(10, 4))
    plt.plot(range(1, 8), observados, label="Observado", marker="o", linestyle='--')
    plt.plot(range(1, 8), previstos, label="Previsto", marker="x", linestyle='-')
    plt.title(f"Hidrograma previsto a partir de t₀ = {idx}")
    plt.xlabel("Horizonte de Previsão (dias)")
    plt.ylabel("Q [m³/s]")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

# === Exemplo de uso ===
plot_hidrograma_7modelos(3368, resultados_pred, data)

#%% === 6. Amortecimento de Cheias com Método de Puls

#Curva de Descarga
cd=pd.read_csv(os.path.abspath(r"05. Arquivos de Texto\cd.txt"),sep="\t")

#Curva Cota Volume
cv=pd.read_csv(os.path.abspath(r"05. Arquivos de Texto\cv.txt"),sep="\t")


# Função adaptada com gráfico com extremos ligados a zero
def aplicar_puls_em_previsao(idx, resultados_pred, data, cv, cd, cota_inicial=549, dt_horas=24):
    """
    Obtém o hidrograma previsto pelos 7 modelos MLP e aplica o método de Puls,
    com gráfico que liga extremos ao zero.
    """
    # === Extrair curvas características ===
    cotas_cv = cv['Cota'].values
    volumes = cv['Volume (1000m3)'].values / 1000  # para hm³
    cotas_cd = cd['Cota'].values
    descargas = cd['Descarga(m3/s)'].values

    # Interpoladores
    vol_por_cota = interp1d(cotas_cv, volumes, fill_value="extrapolate")
    cota_por_vol = interp1d(volumes, cotas_cv, fill_value="extrapolate")
    descarga_por_cota = interp1d(cotas_cd, descargas, fill_value="extrapolate")

    def metodo_de_puls(hidro_afluente, dt_horas, cota_inicial):
        dt = dt_horas * 3600
        V_atual = vol_por_cota(cota_inicial) * 1e6  # m³
        Q_defluente = []

        for t in range(len(hidro_afluente)):
            Q_in = hidro_afluente[t]
            Q_out_ant = Q_defluente[-1] if t > 0 else 0
            Q_in_ant = hidro_afluente[t - 1] if t > 0 else Q_in

            V_novo = V_atual + dt/2 * (Q_in + Q_in_ant - Q_out_ant - Q_out_ant)
            cota_nova = float(cota_por_vol(V_novo / 1e6))
            Q_out_novo = float(descarga_por_cota(cota_nova))

            Q_defluente.append(Q_out_novo)
            V_atual = V_novo

        return Q_defluente

    # === Obter previsão ===
    if idx < 0 or idx >= len(data):
        print("Índice fora do intervalo válido.")
        return

    hidro_previsto = [resultados_pred[t - 1][idx] for t in range(1, 8)]
    hidro_amortecido = metodo_de_puls(hidro_previsto, dt_horas, cota_inicial)

    # === Plotagem com extremos em zero ===
    dias = np.arange(1, 8)
    dias_ext = np.concatenate(([0], dias, [8]))
    Q_in_ext = np.concatenate(([0], hidro_previsto, [0]))
    Q_out_ext = np.concatenate(([0], hidro_amortecido, [0]))

    plt.figure(figsize=(10, 4))
    plt.plot(dias_ext, Q_in_ext, label='Q Afluente (Previsto)', marker='o')
    plt.plot(dias_ext, Q_out_ext, label='Q Defluente (Puls)', marker='x')
    plt.xlabel("Dia")
    plt.ylabel("Vazão [m³/s]")
    plt.title(f"Amortecimento via Método de Puls (t₀ = {idx})")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

    return hidro_previsto, hidro_amortecido

    
hid_previsto,hid_amortecido=aplicar_puls_em_previsao(idx=3368, resultados_pred=resultados_pred, data=data,cv=cv,cd=cd, cota_inicial=549)

#%% === 7. Inserção de Hidrograma Previsto no HEC-RAS

def executar_modelo_ras_com_hidrograma(
    caminho_u01_original: str,
    caminho_u01_modificado: str,
    caminho_prj: str,
    vazoes_previstas: list[float],
    manter_primeiras_vazoes: list[float] = [0, 500, 1000]
):
    """
    Substitui as 7 últimas vazões em um arquivo .u01/.p02 de HEC-RAS 2D e executa o modelo.
    """
    assert len(vazoes_previstas) == 7, "A lista de vazões previstas deve conter exatamente 7 valores."
    assert len(manter_primeiras_vazoes) == 3, "A lista de vazões fixas deve conter 3 valores."

    # === Leitura do arquivo original ===
    with open(caminho_u01_original, 'r') as f:
        linhas = f.readlines()

    # === Linha de cabeçalho padrão ===
    linha_cabecalho = "Flow Hydrograph= 10 \n"

    # === Linha de valores formatada ===
    valores = manter_primeiras_vazoes + vazoes_previstas
    linha_valores = "".join(f"{v:8.3f}" for v in valores) + "\n"

    # === Substituição das linhas corretas ===
    for i, linha in enumerate(linhas):
        if linha.strip().startswith("Flow Hydrograph="):
            linhas[i] = linha_cabecalho
            linhas[i + 1] = linha_valores
            break

    # === Escrita do novo arquivo ===
    with open(caminho_u01_modificado, 'w') as f:
        f.writelines(linhas)

    # === Execução do HEC-RAS ===
    RC = win32com.client.Dispatch("RAS641.HECRASCONTROLLER")
    RC.ShowRAS()
    RC.Project_Open(caminho_prj)
    RC.Compute_CurrentPlan(None, None, True)
    RC.Project_Save()
    RC.QuitRAS()


executar_modelo_ras_com_hidrograma(
    caminho_u01_original=os.path.abspath(r"02. HECRAS\Model_3M.u01"),
    caminho_u01_modificado=os.path.abspath(r"02. HECRAS\Model_3M.u01"),
    caminho_prj=os.path.abspath(r"02. HECRAS\Model_3M.prj"),
    vazoes_previstas=hid_previsto
)

#%% === 8. Analisar edificações impactadas


def contar_edificacoes_inundadas(
    caminho_edificios: str,
    caminho_inundacao: str
) -> int:
    """
    Analisa a interseção entre edificações e a mancha de inundação,
    informando o número total de construções potencialmente impactadas.

    Parâmetros:
    - caminho_edificios: str – Caminho para o shapefile de edificações (polígonos).
    - caminho_inundacao: str – Caminho para o shapefile de mancha de inundação (polígonos).

    Retorna:
    - Número de edificações com interseção com a mancha de inundação.
    """

    print("===============================================")
    print("     SISTEMA DE ANÁLISE DE IMPACTO POR INUNDAÇÃO")
    print("===============================================\n")

    print("🔄 Carregando dados geoespaciais...")

    # Leitura dos shapefiles
    gdf_edificios = gpd.read_file(caminho_edificios)
    gdf_inundacao = gpd.read_file(caminho_inundacao)

    # Padronização do CRS
    if gdf_edificios.crs != gdf_inundacao.crs:
        print("🔁 Reprojetando sistemas de referência para garantir compatibilidade...")
        gdf_inundacao = gdf_inundacao.to_crs(gdf_edificios.crs)

    print("📍 Identificando interseções espaciais entre edificações e área inundada...")

    # Cálculo das interseções
    edificios_inundados = gdf_edificios[gdf_edificios.geometry.intersects(gdf_inundacao.unary_union)]
    total_impactadas = len(edificios_inundados)

    print("\n✅ Análise Concluída!")
    print(f"🏚️  Total de edificações impactadas pela inundação: **{total_impactadas}**\n")
    print("⚠️  Recomenda-se avaliação detalhada de vulnerabilidades estruturais e sociais.")

    return total_impactadas,edificios_inundados
total_impactadas,edificios_inundados=contar_edificacoes_inundadas(
    caminho_edificios=r"01. Shapefiles\Edificios.shp",
    caminho_inundacao=r"02. HECRAS\Plan_3M\Inundation Boundary (Max Value_0).shp"
)
#%% === 9. Geração de Mapa de Inundação

import folium

def gerar_mapa_inundacao_html(gdf_inundacao, gdf_edificios, nome_arquivo_html="mapa_inundacao.html"):
    """
    Gera um mapa interativo em HTML com base nos dados da mancha de inundação e edificações impactadas.

    Parâmetros:
    - gdf_inundacao: GeoDataFrame com a mancha de inundação (já reprojetado para EPSG:4326)
    - gdf_edificios: GeoDataFrame com as edificações impactadas (já reprojetado para EPSG:4326)
    - nome_arquivo_html: Nome do arquivo HTML de saída
    """
    # Centro do mapa
    centro = gdf_edificios.unary_union.centroid.coords[:][0][::-1]  # (lat, lon)

    # Criar o mapa com fundo Google Hybrid (via tiles do Google, não oficialmente suportado)
    m = folium.Map(location=centro, zoom_start=16, control_scale=True, tiles=None)
    folium.TileLayer(
        tiles='https://mt1.google.com/vt/lyrs=y&x={x}&y={y}&z={z}',
        attr='Google Hybrid',
        name='Google Hybrid',
        overlay=False,
        control=True
    ).add_to(m)

    # Adiciona camada da mancha de inundação
    folium.GeoJson(
        gdf_inundacao,
        name="Mancha de Inundação",
        style_function=lambda x: {'color': 'blue', 'fillColor': 'blue', 'fillOpacity': 0.3, 'weight': 1}
    ).add_to(m)

    # Adiciona camada das edificações impactadas
    folium.GeoJson(
        gdf_edificios,
        name="Edificações Impactadas",
        style_function=lambda x: {'color': 'red', 'fillColor': 'red', 'fillOpacity': 0.5, 'weight': 0.5}
    ).add_to(m)

    folium.LayerControl().add_to(m)

    # Salva o mapa como HTML
    m.save(nome_arquivo_html)
    print(f"🗺️ Mapa gerado com sucesso: {nome_arquivo_html}")


# Após rodar a função principal
total_impactadas, edificios_inundados = contar_edificacoes_inundadas(
    caminho_edificios=r"01. Shapefiles\Edificios.shp",
    caminho_inundacao=r"02. HECRAS\Plan_3M\Inundation Boundary (Max Value_0).shp"
)

# Reprojetar os dados para EPSG:4326 para visualização web
inundacao_4326 = gpd.read_file(r"02. HECRAS\Plan_3M\Inundation Boundary (Max Value_0).shp").to_crs(epsg=4326)
edificios_inundados_4326 = edificios_inundados.to_crs(epsg=4326)

# Gerar mapa
gerar_mapa_inundacao_html(inundacao_4326, edificios_inundados_4326, "mapa_inundacao.html")

# Caminho absoluto para o HTML
caminho_html = os.path.abspath("mapa_inundacao.html")
webbrowser.open(f"file://{caminho_html}")

#%% === 10. Danos econômicos por edificação a partir do raster de profundidade ===
import rasterio
from rasterio.mask import mask
from shapely.geometry import mapping
from pyproj import CRS

# --- Parâmetros ---
# Raster de profundidade (m)
caminho_depth_tif = os.path.join(os.getcwd(), "02. HECRAS", "Plan_3M", "Depth (Max).MDT_3M.MDT_3M.tif")


# Coeficientes da função de dano (y em R$/m², x em m)
A_CONST = 90.832
B_CONST = 39.334
MIN_DEPTH = 1e-3  # profundidade mínima (m) para evitar ln(0); abaixo disso dano=0

# --- Funções auxiliares ---
def _utm_from_lonlat(lon, lat):
    """
    Retorna um CRS UTM adequado para o par (lon, lat) no hemisfério Sul/Norte.
    """
    zone = int((lon + 180) // 6) + 1
    south = lat < 0
    epsg = 32700 + zone if south else 32600 + zone
    return CRS.from_epsg(epsg)

def _ensure_projected_area_crs(gdf, prefer_crs=None):
    """
    Garante que o GeoDataFrame esteja em CRS projetado (metros) para cálculo de área.
    Usa prefer_crs se ele for projetado; caso contrário, escolhe UTM pelo centróide.
    """
    if prefer_crs is not None:
        try:
            crs_pref = CRS.from_user_input(prefer_crs)
            if crs_pref.is_projected:
                return gdf.to_crs(crs_pref)
        except Exception:
            pass
    # Se o CRS atual já é projetado, mantém
    if gdf.crs and CRS.from_user_input(gdf.crs).is_projected:
        return gdf
    # Caso contrário, escolhe UTM pela média dos centróides
    cent = gdf.to_crs(epsg=4326).unary_union.centroid
    target_crs = _utm_from_lonlat(cent.x, cent.y)
    return gdf.to_crs(target_crs)

def _mean_depth_for_polygon(geom, src, use_centroid_fallback=True, buffer_cells=0.25):
    """
    Retorna a profundidade média (m) do raster dentro do polígono `geom`.
    - all_touched=True para capturar células tocadas por footprints pequenos.
    - Corrige geometrias inválidas (buffer(0)).
    - Opcionalmente amplia levemente o polígono (buffer de fração do pixel).
    - Fallback: amostra o valor no centróide se o recorte vier todo como NoData.
    """
    try:
        g = geom if geom.is_valid else geom.buffer(0)
        if g.is_empty:
            return np.nan

        # pequeno buffer para footprints muito pequenos (ex.: casas < célula)
        try:
            px_size = max(abs(src.transform.a), abs(src.transform.e))  # resolução (m ou grau)
        except Exception:
            px_size = 0.0
        if buffer_cells and px_size and px_size < 1e6:  # evita exagero caso CRS em graus
            g = g.buffer(px_size * buffer_cells)

        out_img, _ = mask(
            src, [mapping(g)], crop=True, filled=True, all_touched=True
        )
        arr = out_img[0].astype("float32")

        nodata = src.nodata
        if nodata is not None and np.isfinite(nodata):
            arr = np.where(np.isclose(arr, nodata), np.nan, arr)

        # profundidades negativas não são físicas
        arr = np.where(arr < 0, 0.0, arr)

        if np.isnan(arr).all():
            if not use_centroid_fallback:
                return np.nan
            # Fallback: amostra o centróide
            cx, cy = g.centroid.x, g.centroid.y
            val = float(list(src.sample([(cx, cy)], indexes=1))[0][0])
            if nodata is not None and (np.isnan(val) or np.isclose(val, nodata)):
                return np.nan
            return max(val, 0.0)

        return float(max(np.nanmean(arr), 0.0))
    except ValueError:
        # geometria completamente fora da extensão
        return np.nan


def _damage_per_m2(depth_m):
    """
    Aplica y = 90.832 + 39.334*ln(x), com salvaguardas para x<=0.
    depth_m <= MIN_DEPTH => y = 0 (sem dano por m²).
    """
    if depth_m is None or np.isnan(depth_m) or depth_m <= MIN_DEPTH:
        return 0.0
    return float(A_CONST + B_CONST * np.log(depth_m))

# --- Preparação dos dados de entrada ---
# Recarrega (ou reaproveita) as edificações impactadas já obtidas anteriormente
# Edificações impactadas estão em `edificios_inundados` (mesmo CRS de gdf_edificios)
gdf_impacto = edificios_inundados.copy()

# === AJUSTE o bloco onde calcula 'depths' e os campos de dano ===
with rasterio.open(caminho_depth_tif) as src:
    raster_crs = src.crs

    # 1) reprojetar edifícios para o CRS do raster
    gdf_for_depth = gdf_impacto if gdf_impacto.crs == raster_crs else gdf_impacto.to_crs(raster_crs)

    # 2) profundidade média por edificação (com all_touched + fallback)
    depths = []
    for geom in gdf_for_depth.geometry:
        d = _mean_depth_for_polygon(geom, src, use_centroid_fallback=True, buffer_cells=0.25)
        depths.append(d)

# 3) área em m² (garante CRS projetado)
gdf_area = _ensure_projected_area_crs(gdf_for_depth, prefer_crs=raster_crs)
areas_m2 = gdf_area.geometry.area.values

# 4) dano por m² (ln) e total por edificação
#    => converte NaN de profundidade para 0 ANTES de aplicar a função de dano (evita "null" no HTML)
depths_arr = np.array(depths, dtype="float64")
depths_arr = np.nan_to_num(depths_arr, nan=0.0, posinf=0.0, neginf=0.0)

damage_rperm2 = [ _damage_per_m2(d) for d in depths_arr ]
damage_total_R = [ dmg_m2 * area for dmg_m2, area in zip(damage_rperm2, areas_m2) ]

gdf_for_depth["depth_m"] = depths_arr
gdf_for_depth["area_m2"] = areas_m2
gdf_for_depth["damage_Rperm2"] = damage_rperm2
gdf_for_depth["damage_R"] = damage_total_R


# 6) Dano total
dano_total_R = float(np.nansum(gdf_for_depth["damage_R"].values))

# 7) Converter para WGS84 para visualização web
gdf_danos_4326 = gdf_for_depth.to_crs(epsg=4326)

# 8) Atualizar o mapa HTML: adicionar camada com danos + marcador com total
def _format_currency_br(value):
    if value is None or np.isnan(value):
        return "R$ 0,00"
    # Formatação simples R$ 1.234.567,89
    s = f"{value:,.2f}"
    s = s.replace(",", "X").replace(".", ",").replace("X", ".")
    return f"R$ {s}"

def gerar_mapa_inundacao_html_com_danos(gdf_inundacao, gdf_edificios_danos, nome_arquivo_html="mapa_inundacao.html"):
    # Centro do mapa: usar centroide das edificações impactadas (caso não haja, cair para inundação)
    if not gdf_edificios_danos.empty:
        centro = gdf_edificios_danos.unary_union.centroid.coords[:][0][::-1]
    else:
        centro = gdf_inundacao.unary_union.centroid.coords[:][0][::-1]

    m = folium.Map(location=centro, zoom_start=16, control_scale=True, tiles=None)
    # Base
    folium.TileLayer(
        tiles='https://mt1.google.com/vt/lyrs=y&x={x}&y={y}&z={z}',
        attr='Google Hybrid',
        name='Google Hybrid',
        overlay=False,
        control=True
    ).add_to(m)

    # Mancha de inundação
    folium.GeoJson(
        gdf_inundacao,
        name="Mancha de Inundação",
        style_function=lambda x: {'color': 'blue', 'fillColor': 'blue', 'fillOpacity': 0.3, 'weight': 1}
    ).add_to(m)

    # Edificações impactadas com danos (tooltip/popup)
    def _popup(feat):
        d = feat["properties"].get("depth_m", None)
        a = feat["properties"].get("area_m2", None)
        dm2 = feat["properties"].get("damage_Rperm2", None)
        dtot = feat["properties"].get("damage_R", None)
        return folium.Popup(
            html=(
                f"<b>Profundidade média:</b> {0 if d is None or np.isnan(d) else round(d,3)} m<br>"
                f"<b>Área:</b> {0 if a is None or np.isnan(a) else int(round(a))} m²<br>"
                f"<b>Dano por m²:</b> {_format_currency_br(dm2)}<br>"
                f"<b>Dano (edificação):</b> {_format_currency_br(dtot)}"
            ),
            max_width=350
        )

    layer_edif = folium.GeoJson(
        gdf_edificios_danos,
        name="Edificações Impactadas (com danos)",
        style_function=lambda x: {'color': 'red', 'fillColor': 'red', 'fillOpacity': 0.5, 'weight': 0.5},
        tooltip=folium.GeoJsonTooltip(
            fields=["depth_m", "area_m2", "damage_Rperm2", "damage_R"],
            aliases=["Profundidade (m)", "Área (m²)", "Dano (R$/m²)", "Dano (R$)"],
            localize=True,
            sticky=False
        )
    )
    layer_edif.add_to(m)

    # Vincular popups
    for feat, obj in zip(gdf_edificios_danos.iterfeatures(), layer_edif.data["features"]):
        # Iterfeatures retorna o dict feature; layer_edif.data contém as mesmas features
        pass
    # A maneira mais simples: adicionar popups por ponto central de cada polígono
    for _, row in gdf_edificios_danos.iterrows():
        centroid = row.geometry.centroid
        folium.CircleMarker(
            location=[centroid.y, centroid.x],
            radius=2,
            fill=True,
            fill_opacity=0.9,
            opacity=0.9,
            popup=folium.Popup(
                html=(
                    f"<b>Profundidade média:</b> {0 if pd.isna(row['depth_m']) else round(row['depth_m'],3)} m<br>"
                    f"<b>Área:</b> {0 if pd.isna(row['area_m2']) else int(round(row['area_m2']))} m²<br>"
                    f"<b>Dano por m²:</b> {_format_currency_br(row['damage_Rperm2'])}<br>"
                    f"<b>Dano (edificação):</b> {_format_currency_br(row['damage_R'])}"
                ),
                max_width=350
            )
        ).add_to(m)

    # Dano total (marcador/controle)
    from branca.element import Element
    
    html_total = f"""
    <div style="
        position: fixed; 
        top: 10px; left: 10px; 
        z-index: 9999; 
        background-color: rgba(255,255,255,0.9);
        padding: 8px 12px; 
        border: 1px solid #333; 
        border-radius: 6px; 
        font-size: 14px;
        ">
        <b>Dano Econômico de Edificações (R$):</b><br>
        {_format_currency_br(dano_total_R)}
    </div>
    """
    m.get_root().html.add_child(Element(html_total))

    folium.LayerControl().add_to(m)
    m.save(nome_arquivo_html)
    print(f"🗺️ Mapa com danos gerado: {nome_arquivo_html}")
    print(f"💰 Dano total: {_format_currency_br(dano_total_R)}")


gerar_mapa_inundacao_html_com_danos(inundacao_4326, gdf_danos_4326, "mapa_inundacao.html")

# Abrir no navegador
caminho_html = os.path.abspath("mapa_inundacao.html")
webbrowser.open(f"file://{caminho_html}")

# Logs finais
print("\n====== RESUMO DE DANOS ======")
print(f"Edificações analisadas: {len(gdf_danos_4326)}")
print(f"Dano total estimado: {_format_currency_br(dano_total_R)}")
print("Campos adicionados ao GeoDataFrame: 'depth_m', 'area_m2', 'damage_Rperm2', 'damage_R'")

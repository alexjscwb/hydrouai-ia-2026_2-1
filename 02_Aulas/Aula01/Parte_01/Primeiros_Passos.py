"""Curso: mapa das estações e comparação de chuva de 1 dia e 24 horas.

Abra no Spyder e execute as células na ordem (Ctrl+Enter).
Os CSVs e o GeoPackage devem ficar na mesma pasta deste script.
"""

# %% 1. Importar bibliotecas
from pathlib import Path
import pandas as pd
import geopandas as gpd
import folium
import matplotlib.pyplot as plt



# %% 2. Definir arquivos de entrada e pasta de resultados
PASTA = Path(__file__).resolve().parent
RESULTADOS = PASTA / 'resultados'
RESULTADOS.mkdir(exist_ok=True)

ARQUIVO_CHUVA = PASTA / 'chuva_CEMADEN_RMBH.csv'
ARQUIVO_COORDENADAS = PASTA / 'coordenadas_CEMADEN_RMBH.csv'
ARQUIVO_MALHA = PASTA / 'municipios_RMBH.gpkg'

# %% 3. Ler os dois CSVs e explorar os DataFrames
chuva = pd.read_csv(ARQUIVO_CHUVA, parse_dates=['data'], index_col='data')
coordenadas = pd.read_csv(ARQUIVO_COORDENADAS, dtype={'estacao': str})
chuva.columns = chuva.columns.astype(str)
chuva = chuva.sort_index()
print(chuva.head())
print(coordenadas)
print('\nDimensões da chuva:', chuva.shape)
print('\nHoras sem dados por estação:')
print(chuva.isna().sum())

# Garantir uma grade horária contínua. Horas ausentes permanecem NaN.
chuva = chuva.asfreq('h')
chuva.index.name = 'data'

# %% 4. Criar GeoDataFrames dos municípios e estações
municipios = gpd.read_file(ARQUIVO_MALHA, layer='municipios_rmbh').to_crs('EPSG:4326')
estacoes = gpd.GeoDataFrame(
    coordenadas,
    geometry=gpd.points_from_xy(coordenadas['longitude'], coordenadas['latitude']),
    crs='EPSG:4326',
)
print('\nMunicípios da RMBH:', len(municipios))
print(estacoes[['estacao', 'municipio', 'geometry']])

# %% 5. Montar e salvar o mapa interativo em HTML
mapa = folium.Map(location=[-19.9, -44.0], zoom_start=9, tiles=None)
folium.TileLayer(
    tiles='https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
    attr='Tiles © Esri — Source: Esri, Maxar, Earthstar Geographics, and the GIS User Community',
    name='Satélite — Esri', max_zoom=19,
).add_to(mapa)
folium.TileLayer('CartoDB positron', name='Mapa claro — CARTO', show=False).add_to(mapa)

folium.GeoJson(
    municipios.to_json(),
    name='Municípios da RMBH',
    style_function=lambda feicao: {
        'color': '#42c8ff', 'weight': 2,
        'fillColor': '#cfe4ef', 'fillOpacity': 0.08,
    },
    tooltip=folium.GeoJsonTooltip(fields=['municipio'], aliases=['Município:']),
).add_to(mapa)

camada_estacoes = folium.FeatureGroup(name='Estações CEMADEN — top 10')
for _, estacao in coordenadas.iterrows():
    popup = (
        f"<b>Estação {estacao['estacao']}</b><br>"
        f"Município: {estacao['municipio']}<br>"
        f"Latitude: {estacao['latitude']:.5f}<br>"
        f"Longitude: {estacao['longitude']:.5f}<br>"
        f"Horas sem dados: {int(estacao['horas_sem_dados'])}<br>"
        f"Falhas no período: {estacao['percentual_falhas']:.2f}%"
    )
    folium.CircleMarker(
        location=[estacao['latitude'], estacao['longitude']],
        radius=7, color='#992c22', fill=True,
        fill_color='#e95f46', fill_opacity=0.95,
        tooltip=f"{estacao['estacao']} — {estacao['municipio']}",
        popup=folium.Popup(popup, max_width=300),
    ).add_to(camada_estacoes)
camada_estacoes.add_to(mapa)
folium.LayerControl(collapsed=False).add_to(mapa)
oeste, sul, leste, norte = municipios.total_bounds
mapa.fit_bounds([[sul, oeste], [norte, leste]])
mapa.save(str(RESULTADOS / 'mapa_RMBH_estacoes.html'))
print('\nMapa salvo:', RESULTADOS / 'mapa_RMBH_estacoes.html')
# Abra o arquivo HTML no navegador. As bibliotecas JavaScript requerem internet.

# %% 6. Série diária: somar as horas de cada dia fixo
# Dia fixo: de 00:00 (inclusive) até 00:00 do dia seguinte (exclusive).
# Um dia com uma ou mais horas vazias terá NaN, sem completar com zero.
chuva_diaria = chuva.resample('D').sum(min_count=24)
chuva_diaria.to_csv(RESULTADOS / 'chuva_diaria.csv', encoding='utf-8-sig')
print('\nChuva diária:')
print(chuva_diaria.head())

# %% 7. Agrupamentos de 24 horas e janelas móveis
# Blocos FIXOS de 24h, iniciando à meia-noite:
chuva_24h_fixa = chuva.resample('24h').sum(min_count=24)
# Compará-los daria coeficiente 1: por isso usamos também 24h MÓVEIS.
chuva_24h_fixa.to_csv(RESULTADOS / 'chuva_24h_blocos_fixos.csv', encoding='utf-8-sig')

# Cada janela móvel soma 24 valores da grade horária, avançando de hora em hora.
# A data da linha identifica a última hora incluída, não o final físico da hora.
chuva_24h_movel = chuva.rolling(window=24, min_periods=24).sum()
chuva_24h_movel.to_csv(RESULTADOS / 'chuva_24h_movel.csv', encoding='utf-8-sig')
print('\nChuva de 24 horas móveis:')
print(chuva_24h_movel.head(26))

# %% 8. Extrair as máximas anuais e calcular os coeficientes
# groupby(index.year) evita diferenças de aliases anuais entre versões do pandas.
maximas_1dia = chuva_diaria.groupby(chuva_diaria.index.year).max()
maximas_24h = chuva_24h_movel.groupby(chuva_24h_movel.index.year).max()
maximas_1dia.index.name = 'ano'
maximas_24h.index.name = 'ano'

# Coeficiente por estação e ano = máxima de 24h móveis / máxima de 1 dia.
# Denominador zero fica NaN, evitando divisão por zero.
# Uma janela que atravessa 31/12 é atribuída ao ano da última hora incluída.
coeficientes = maximas_24h.div(maximas_1dia.where(maximas_1dia > 0))
coeficientes.index.name = 'ano'
maximas_1dia.to_csv(RESULTADOS / 'maximas_anuais_1dia.csv', encoding='utf-8-sig')
maximas_24h.to_csv(RESULTADOS / 'maximas_anuais_24h.csv', encoding='utf-8-sig')
coeficientes.to_csv(RESULTADOS / 'coeficientes_estacao_ano.csv', encoding='utf-8-sig')
print('\nCoeficientes por estação e ano:')
print(coeficientes.round(3))

# %% 9. Organizar uma tabela com qualidade e resultados por estação/ano
# Contamos horas válidas, mas isso não prova integridade dentro de cada hora.
horas_validas = chuva.notna().groupby(chuva.index.year).sum()
dias_validos = chuva_diaria.notna().groupby(chuva_diaria.index.year).sum()
linhas = []
for ano in coeficientes.index:
    horas_ano = 8784 if pd.Timestamp(year=ano, month=12, day=31).is_leap_year else 8760
    for codigo in coeficientes.columns:
        linhas.append({
            'ano': ano, 'estacao': codigo,
            'maxima_1dia_mm': maximas_1dia.loc[ano, codigo],
            'maxima_24h_mm': maximas_24h.loc[ano, codigo],
            'coeficiente_24h_1dia': coeficientes.loc[ano, codigo],
            'horas_com_dados': int(horas_validas.loc[ano, codigo]),
            'percentual_horas_ano_com_dados': 100 * horas_validas.loc[ano, codigo] / horas_ano,
            'dias_com_24_horas_validas': int(dias_validos.loc[ano, codigo]),
        })
resumo = pd.DataFrame(linhas).merge(
    coordenadas[['estacao', 'municipio', 'latitude', 'longitude']],
    on='estacao', validate='many_to_one',
)
resumo.to_csv(RESULTADOS / 'resumo_estacao_ano.csv', index=False, encoding='utf-8-sig')
print('\nResumo:')
print(resumo.head())
# 2014 é parcial. Mesmo nos demais anos, falhas podem ocultar eventos máximos.
# Os coeficientes são exploratórios, sem ajuste por tempo de retorno.

# %% 10. Boxplot dos coeficientes por estação
ax = coeficientes.boxplot(
    figsize=(12, 6),
    rot=45,
    grid=False
)

ax.set_title('CEMADEN — RMBH: distribuição dos coeficientes anuais')
ax.set_xlabel('Estação')
ax.set_ylabel('Máxima de 24h móveis / máxima de 1 dia')
ax.axhline(1, color='gray', linestyle='--', linewidth=1)
ax.grid(axis='y', alpha=0.25)

plt.tight_layout()
plt.savefig(RESULTADOS / 'boxplot_coeficientes.png', dpi=160)
plt.show()

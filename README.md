# Estudos de caso: inteligência artificial aplicada à hidrologia

Material didático com dados, scripts Python e arquivos de apoio para três aplicações: previsão de afluência e trânsito de cheias, previsão de níveis e estimativa simplificada de potencial energético.

## Estudos disponíveis

| Caso | Aplicação | Métodos principais |
|---|---|---|
| 1 — Três Marias | Previsão de afluência, trânsito de cheias e análise de inundação | MLP, curvas do reservatório, integração com HEC-RAS e análise geoespacial |
| 2 — Piracicaba | Previsão de nível com dados de chuva e nível | LSTM e grid search, com diferentes configurações de janelas temporais |
| 3 — Três Marias | Previsão de afluência e conversão em potencial energético equivalente | LSTM, grid search e taxa de aprendizado adaptativa |

O caso 3 utiliza a **UHE Três Marias**. O nome original da pasta, `03. Geração de Energia PCH`, foi mantido para preservar a organização dos arquivos.

## Organização

O material está organizado na pasta `EstudosDeCaso`:

```text
EstudosDeCaso/
├── 01. Transito de Cheias 3M/
│   ├── 01. Shapefiles/
│   ├── 02. HECRAS/
│   ├── 05. Arquivos de Texto/
│   └── 06. Script/EstudodeCaso01.py
├── 02. Sistema de Previsão Piracicaba/
│   ├── 01. Arquivo de Texto/dados_interpolados.txt
│   └── 02. Scripts/
│       ├── Grid_Search_CenI_Paper.py
│       └── Grid_Search_CenII_Paper.py
└── 03. Geração de Energia PCH/
    ├── 01. Arquivo de Texto/dataset_filled.csv
    ├── 02. Script/EstudodeCaso03_Grid_LSTM.py
    └── 03. Documentação/PAE_UHE_Tres_Marias_revF-Tachado.pdf
```

A árvore resume os principais arquivos; o pacote contém outros arquivos de apoio. Se o repositório distribuir apenas `EstudosDeCaso.zip`, extraia-o antes de executar os exemplos. Preserve os arquivos auxiliares dos shapefiles e a estrutura do projeto HEC-RAS.

## Ambiente Python

Os scripts utilizam PyTorch, NumPy, pandas, Matplotlib e scikit-learn. O caso 1 também utiliza SciPy, pywin32, GeoPandas, Folium, Rasterio, Shapely, pyproj e branca.

Utilize um ambiente Python separado. Instale o PyTorch conforme o sistema e a GPU, seguindo as [instruções oficiais](https://pytorch.org/get-started/locally/). Para as demais bibliotecas dos casos 2 e 3:

```bash
python -m pip install numpy pandas matplotlib scikit-learn
```

Para os recursos adicionais do caso 1, no Windows:

```bash
python -m pip install scipy pywin32 geopandas folium rasterio shapely pyproj branca
```

A automação hidráulica do caso 1 usa o controlador `RAS641.HECRASCONTROLLER` e requer HEC-RAS 6.4.1 instalado e configurado. O pacote Python não instala o HEC-RAS.

No Spyder, selecione o interpretador do ambiente que contém as dependências. Confira no console:

```python
import sys
import torch
print(sys.executable)
print("PyTorch:", torch.__version__)
print("CUDA disponível:", torch.cuda.is_available())
```

Os scripts LSTM selecionam GPU CUDA quando disponível e CPU caso contrário. O tempo de execução depende do equipamento e do número de combinações treinadas.

## Caso 1 — Previsão e trânsito de cheias em Três Marias

O script `EstudodeCaso01.py` treina sete MLPs para prever afluência nos horizontes de 1 a 7 dias. Na configuração incluída, cada rede tem três camadas ocultas com 128 neurônios, treinamento de 100 épocas, lote de 1.000 exemplos e otimizador Adam com taxa de aprendizado de 0,001.

O fluxo utiliza também as curvas `cd.txt` e `cv.txt`, arquivos do HEC-RAS e dados de edificações para demonstrar aplicações hidráulicas e geoespaciais.

Defina a pasta `01. Transito de Cheias 3M` como diretório de trabalho. A partir dela:

```bash
python "06. Script/EstudodeCaso01.py"
```

No Spyder, configure esse mesmo diretório de trabalho e execute por seções para acompanhar as etapas. A parte hidráulica modifica `02. HECRAS/Model_3M.u01` e depende dos resultados espaciais do HEC-RAS; trabalhe sobre uma cópia do projeto.

**Avaliação:** a versão incluída utiliza divisão aleatória de 80%/20%, sem um conjunto de teste temporal independente. Seus resultados não são diretamente comparáveis aos do caso 3, que adota separação cronológica.

## Caso 2 — Sistema de previsão para Piracicaba

Os scripts utilizam dados em passos de 10 minutos e têm como alvo o nível `FLU(m)46`.

- **Cenário I:** janela de 36 passos, equivalente a 6 horas; valores configurados de horizonte de 6, 18, 36, 54 e 72 passos.
- **Cenário II:** janelas específicas por variável, com deslocamentos temporais; o arquivo contém dois blocos de experimentos e está configurado com `T_list = [54]`.

Os parâmetros representam passos de amostragem. Para interpretar o tempo exato entre a última entrada e o alvo, confira também os índices usados na construção das sequências.

Antes de executar, substitua **todas as ocorrências** do caminho absoluto de `dados_interpolados.txt` nos scripts pelo caminho do arquivo fornecido. Esse caminho ainda referencia a máquina de origem. No primeiro bloco do Cenário II, `H = 48` corresponde a **8 horas** em passos de 10 minutos, apesar do comentário mencionar 6 horas.

Execute o script do cenário desejado após ajustar os caminhos. Os experimentos salvam tabelas de resultados, modelos, scalers e figuras no diretório de trabalho. A execução integral do Cenário II percorre os dois blocos presentes no arquivo.

## Caso 3 — Previsão de afluência e potencial energético

O script `EstudodeCaso03_Grid_LSTM.py` utiliza uma janela diária de sete dias, de **T−6 até T**, para prever separadamente **Q(T+1), Q(T+3) e Q(T+7)**.

As entradas são as colunas de precipitação e vazão da base. Por padrão, incluem também a afluência histórica de Três Marias, supondo que ela esteja disponível ao final do dia T. Essa opção é controlada por `INCLUIR_AFLUENCIA_HISTORICA`.

| Parâmetro | Configuração incluída |
|---|---|
| Janela | 7 dias |
| Horizontes | 1, 3 e 7 dias |
| Camadas LSTM | 1, 2 e 3 |
| Unidades ocultas | 64, 128 e 256 |
| Grid search | 9 combinações por horizonte; 27 treinamentos |
| Batch size | 128 |
| Épocas máximas | 200 |
| Otimizador | Adam |
| Learning rate inicial | 0,001 |
| Redução do learning rate | `ReduceLROnPlateau`, fator 0,5 e paciência 5 |
| Learning rate mínimo | 0,000001 |
| Early stopping | Paciência de 25 épocas |
| Divisão temporal | 70% treino, 15% validação e 15% teste |

Os scalers são ajustados somente no treino. A validação orienta a seleção da configuração, a redução da taxa de aprendizado e a parada antecipada. O conjunto de teste é reservado para avaliar o modelo escolhido de cada horizonte.

### Execução

Na pasta `03. Geração de Energia PCH`, execute:

```bash
python "02. Script/EstudodeCaso03_Grid_LSTM.py" --dados "01. Arquivo de Texto/dataset_filled.csv" --out "resultados_energia"
```

O argumento `--dados` é necessário nessa organização: o caminho padrão do script procura o CSV na mesma pasta do arquivo Python. Para executar com F5 no Spyder sem argumentos, copie o CSV para `02. Script` ou ajuste o caminho padrão no código.

Os resultados incluem métricas, previsões, rankings de validação, históricos de treinamento e taxa de aprendizado, figuras, modelos e scalers. O arquivo `experimento.json` registra as configurações e informações da execução.

### Conversão para energia

A conversão didática utiliza:

```text
P(MW) = min[0,00981 × η × H × max(Q, 0), 396]
E(MWh) = 24 × P(MW)
```

Adotam-se rendimento constante `η = 0,90` e altura de referência `H = 56,8 m`. A potência instalada de 396 MW é informada no PAE incluído. A altura de referência corresponde à diferença entre níveis escolhidos no documento e não deve ser interpretada como queda líquida nominal verificada.

O resultado representa **potencial energético equivalente**, supondo conversão da afluência em vazão utilizável até o limite de potência. Não reproduz a operação real do reservatório: não simula armazenamento, despacho, vertimento, indisponibilidade das unidades ou variação da queda.

Para cada horizonte, a energia representa **24 horas do dia-alvo**. A previsão de 7 dias não é energia acumulada durante os próximos sete dias. A referência energética é calculada a partir da vazão observada pela mesma equação; não é geração observada.

## Uso e interpretação

Os exemplos têm finalidade didática. As bases são previamente preenchidas e os arquivos fornecidos não permitem reconstruir integralmente a procedência de cada preenchimento. A separação temporal do treinamento não garante, por si só, que o processamento anterior das bases esteve livre de informação futura.

Os casos possuem protocolos de avaliação diferentes. Compare modelos usando os mesmos períodos, variáveis, horizontes e critérios de avaliação. Grid search não garante desempenho superior, e métricas de treino não substituem avaliação fora da amostra.

Este README descreve os arquivos distribuídos; não representa uma certificação de execução integral de todos os fluxos em outros computadores.

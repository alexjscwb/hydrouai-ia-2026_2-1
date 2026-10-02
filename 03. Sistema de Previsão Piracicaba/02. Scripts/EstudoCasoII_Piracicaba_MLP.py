# -*- coding: utf-8 -*-
"""Estudo II: Piracicaba, cenarios I e II com MLP.
Coloque em 02. Scripts; o TXT fica em ../01. Arquivo de Texto.
Execute com F5 no Spyder ou python EstudoCasoII_Piracicaba_MLP.py.
Entradas disponiveis ate t; alvo em t+h (h passos de 10 minutos).
"""
# %% 1. Configuracoes para a aula
import os
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
from pathlib import Path
from itertools import product
import json
import pickle
import random
import numpy as np
import pandas as pd
import torch
from torch import nn
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

PASTA = Path(__file__).resolve().parent
DADOS = PASTA.parent / "01. Arquivo de Texto" / "dados_interpolados.txt"
SAIDA = PASTA / "resultados_mlp"
CENARIOS = ["I", "II"]
HORIZONTES = [18, 72]  # 3 e 12 horas, em passos de 10 minutos.
CAMADAS = [1, 2]
NEURONIOS = [64, 128]
EPOCAS = 60
BATCH_SIZE = 1024
LR = 1e-3
PACIENCIA = 12
PACIENCIA_LR = 3
SEMENTE = 42
PASSO_TREINO = 1  # 6 usa uma origem por hora no treino; validacao/teste completos.
ALVO = "FLU(m)46"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Cenario I original exclui historico do alvo; II inclui.
# True inclui no I tambem, para comparar janelas com esse historico em ambos.
ALVO_NO_CENARIO_I = True
H_I = 144  # 6 horas de historico, incluindo t.
# Cenario II: segunda variante (varH) do arquivo original.
# H=60 significa 10 horas, e nao 8 horas como dizia o comentario original.
H_DEFAULT_II = 36
H_POR_COLUNA = {ALVO: 36, "FLU(m)713": 36, "PLU(mm)713": 36}
CENTROS = {
    18: {"713": 36},
    72: {"713": 36, "48": 54, "57": 48, "59": 60, "50": 54},
}

# %% 2. Leitura: nao interpolamos novamente a base

def ler_dados(caminho):
    df = pd.read_csv(caminho, sep="\t")
    df["DATA"] = pd.to_datetime(df["DATA"], format="%m/%d/%Y %H:%M")
    df = df.set_index("DATA").sort_index()
    if df.index.has_duplicates or df.index.hasnans:
        raise ValueError("Datas duplicadas/invalidas: revisar a base.")
    if len(df) < 200 or not (df.index.to_series().diff().iloc[1:] == pd.Timedelta(minutes=10)).all():
        raise ValueError("A base deve ter calendario continuo de 10 minutos e >=200 registros.")
    df = df.apply(pd.to_numeric, errors="raise").replace([np.inf, -np.inf], np.nan)
    if ALVO not in df:
        raise ValueError(f"Coluna alvo ausente: {ALVO}")
    # Ausencias invalidam as janelas correspondentes, sem comprimir o calendario.
    return df


def especificacao(colunas, cenario, horizonte):
    """Lista (coluna, quantidade de passos, atraso do ultimo valor ate t)."""
    if cenario == "I":
        return [(c, H_I, 0) for c in colunas if c != ALVO or ALVO_NO_CENARIO_I]
    if cenario != "II" or horizonte not in CENTROS:
        raise ValueError("Cenario/horizonte sem configuracao de janelas.")
    spec = []
    for estacao, centro in CENTROS[horizonte].items():
        for tipo in ["FLU(m)", "PLU(mm)"]:
            c = tipo + estacao
            h = H_POR_COLUNA.get(c, H_DEFAULT_II)
            atraso = centro - h // 2
            if atraso < 0:
                raise ValueError("A janela usaria informacao futura.")
            spec.append((c, h, atraso))
    spec.append((ALVO, H_POR_COLUNA[ALVO], 0))
    return spec


class BaseJanelas:
    """Monta somente o lote solicitado: evita copiar todas as janelas na RAM.
    Cada variavel ocupa um bloco de atrasos, do mais antigo ao mais recente.
    A MLP recebe esses blocos concatenados em um vetor, sem padding de zeros.
    """
    def __init__(self, df, spec, horizonte):
        self.spec, self.horizonte = spec, horizonte
        self.colunas = [c for c, h, a in spec]
        raw = df[self.colunas].to_numpy(dtype=np.float32)
        q = df[ALVO].to_numpy(dtype=np.float32)
        n = len(df)
        c1, c2 = int(.70*n), int(.85*n)
        self.blocos = np.zeros(n, dtype=int)
        self.blocos[c1:c2], self.blocos[c2:] = 1, 2
        primeiro = max(h-1+a for c, h, a in spec)
        origens = np.arange(primeiro, n-horizonte)
        valido = np.isfinite(q[origens+horizonte])
        for j, (c, h, atraso) in enumerate(spec):
            # Soma cumulativa conta ausencias em cada janela sem grandes matrizes.
            faltas = np.r_[0, np.cumsum(~np.isfinite(raw[:, j]))]
            fim = origens-atraso+1
            valido &= (faltas[fim]-faltas[fim-h]) == 0
        valido &= self.blocos[origens] == self.blocos[origens+horizonte]
        origens = origens[valido]
        self.indices = {nome: origens[self.blocos[origens] == k]
                        for k, nome in enumerate(["treino", "validacao", "teste"])}
        self.indices["treino"] = self.indices["treino"][::PASSO_TREINO]
        if any(len(v) < 10 for v in self.indices.values()):
            raise ValueError("Menos de 10 exemplos validos em algum bloco.")
        if not np.isfinite(raw[:c1]).any(axis=0).all():
            raise ValueError("Preditor sem dados no treino.")
        # Scaler X apenas em datas do treino; scaler y apenas em alvos de treino.
        self.sx = StandardScaler().fit(raw[:c1])
        self.sy = StandardScaler().fit(q[self.indices["treino"]+horizonte, None])
        self.x = self.sx.transform(raw).astype(np.float32)
        self.y = self.sy.transform(q[:, None]).astype(np.float32)
        self.dim = sum(h for c, h, a in spec)

    def lote(self, origens):
        partes = []
        for j, (c, h, atraso) in enumerate(self.spec):
            pos = origens[:, None] - atraso - np.arange(h-1, -1, -1)
            partes.append(self.x[pos, j])
        x = torch.from_numpy(np.concatenate(partes, axis=1)).to(DEVICE)
        y = torch.from_numpy(self.y[origens+self.horizonte]).to(DEVICE)
        return x, y

# %% 3. MLP e treinamento

def criar_modelo(entradas, camadas, unidades):
    modulos = []
    for _ in range(camadas):
        modulos.extend([nn.Linear(entradas, unidades), nn.ReLU()])
        entradas = unidades
    return nn.Sequential(*modulos, nn.Linear(entradas, 1)).to(DEVICE)


@torch.no_grad()
def prever(modelo, base, indices):
    modelo.eval()
    saida = []
    for i in range(0, len(indices), BATCH_SIZE):
        x, _ = base.lote(indices[i:i+BATCH_SIZE])
        saida.append(modelo(x).cpu().numpy())
    return np.concatenate(saida)


def treinar(base, camadas, unidades):
    random.seed(SEMENTE)
    np.random.seed(SEMENTE)
    torch.manual_seed(SEMENTE)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEMENTE)
    modelo = criar_modelo(base.dim, camadas, unidades)
    opt = torch.optim.Adam(modelo.parameters(), lr=LR, weight_decay=1e-4)
    # Se a validacao estacionar, reduz LR pela metade; nao olha o teste.
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, mode="min", factor=.5, patience=PACIENCIA_LR, min_lr=1e-6)
    melhor, espera, estado, melhor_epoca = float("inf"), 0, None, 0
    historico = []
    for epoca in range(1, EPOCAS+1):
        modelo.train()
        ordem = np.random.permutation(base.indices["treino"])
        soma = 0.
        lr_usado = opt.param_groups[0]["lr"]
        for i in range(0, len(ordem), BATCH_SIZE):
            ids = ordem[i:i+BATCH_SIZE]
            x, y = base.lote(ids)
            opt.zero_grad(set_to_none=True)
            loss = nn.functional.mse_loss(modelo(x), y)
            if not torch.isfinite(loss):
                raise ValueError("Perda nao finita: revisar entradas/treinamento.")
            loss.backward()
            nn.utils.clip_grad_norm_(modelo.parameters(), 5.)
            opt.step()
            soma += loss.item()*len(ids)
        ids = base.indices["validacao"]
        val = float(np.mean((prever(modelo, base, ids)-base.y[ids+base.horizonte])**2))
        if not np.isfinite(val):
            raise ValueError("Perda de validacao nao finita.")
        historico.append(dict(epoca=epoca, mse_treino=soma/len(ordem), mse_validacao=val, lr=lr_usado))
        scheduler.step(val)
        if val < melhor:
            melhor, espera, melhor_epoca = val, 0, epoca
            estado = {k: v.detach().cpu().clone() for k, v in modelo.state_dict().items()}
        else:
            espera += 1
        if epoca == 1 or epoca % 5 == 0:
            print(f"  epoca {epoca:02d}: MSE val={val:.5f}, LR={lr_usado:.1e}", flush=True)
        if espera >= PACIENCIA:
            break
    # Restaura sempre, mesmo quando chega ao limite de epocas.
    modelo.load_state_dict(estado)
    return modelo, melhor, melhor_epoca, pd.DataFrame(historico)

# %% 4. Metricas em metros e exportacao

def metricas(o, p):
    o, p = np.asarray(o).ravel(), np.asarray(p).ravel()
    erro = p-o
    den = np.sum((o-o.mean())**2)
    nse = 1-np.sum(erro**2)/den if den > 0 else np.nan
    kge = np.nan
    if o.std() > 0 and p.std() > 0 and abs(o.mean()) > 1e-12:
        r = np.corrcoef(o, p)[0, 1]
        kge = 1-np.sqrt((r-1)**2+(p.std()/o.std()-1)**2+(p.mean()/o.mean()-1)**2)
    return dict(RMSE=float(np.sqrt(np.mean(erro**2))), MAE=float(np.mean(abs(erro))),
                NSE=float(nse), R2=float(nse), KGE=float(kge), vies=float(erro.mean()))


def main():
    torch.set_num_threads(2)
    df = ler_dados(DADOS)
    SAIDA.mkdir(parents=True, exist_ok=True)
    print(f"Dispositivo: {DEVICE}; {len(CENARIOS)*len(HORIZONTES)*len(CAMADAS)*len(NEURONIOS)} treinamentos", flush=True)
    print("Base previamente interpolada: sem garantia sobre causalidade do preenchimento original.")
    resultados, auditoria = [], []
    for horizonte in HORIZONTES:
        bases = {c: BaseJanelas(df, especificacao(df.columns, c, horizonte), horizonte) for c in CENARIOS}
        # Mesmas datas para comparar cenarios, mesmo se tiverem ausencias distintas.
        for bloco in ["treino", "validacao", "teste"]:
            comum = bases[CENARIOS[0]].indices[bloco]
            for base in bases.values():
                comum = np.intersect1d(comum, base.indices[bloco])
            if len(comum) < 10:
                raise ValueError("Poucas datas comuns entre os cenarios.")
            for base in bases.values():
                base.indices[bloco] = comum
        for cenario, base in bases.items():
            destino = SAIDA / f"cenario_{cenario}_h{horizonte*10}min"
            destino.mkdir(parents=True, exist_ok=True)
            ranking, vencedor, melhor_val = [], None, float("inf")
            for camadas, unidades in product(CAMADAS, NEURONIOS):
                print(f"Cenario {cenario}, {horizonte*10} min, L={camadas}, U={unidades}", flush=True)
                modelo, val, epoca, hist = treinar(base, camadas, unidades)
                hist.to_csv(destino/f"historico_L{camadas}_U{unidades}.csv", index=False)
                ranking.append(dict(camadas=camadas, unidades=unidades, mse_validacao_normalizado=val, melhor_epoca=epoca))
                if val < melhor_val:
                    melhor_val = val
                    vencedor = (camadas, unidades, {k: v.detach().cpu().clone() for k, v in modelo.state_dict().items()}, hist)
                del modelo
            pd.DataFrame(ranking).sort_values("mse_validacao_normalizado").to_csv(destino/"ranking.csv", index=False)
            camadas, unidades, estado, hist = vencedor
            modelo = criar_modelo(base.dim, camadas, unidades)
            modelo.load_state_dict(estado)
            torch.save(dict(state_dict=estado, entradas=base.dim, camadas=camadas, unidades=unidades,
                            horizonte_passos=horizonte, janelas=base.spec), destino/"modelo.pt")
            with open(destino/"scalers.pkl", "wb") as f:
                pickle.dump(dict(x=base.sx, y=base.sy, colunas=base.colunas), f)
            fig, axs = plt.subplots(3, 1, figsize=(13, 9), constrained_layout=True)
            contagens = {}
            for ax, (bloco, ids) in zip(axs, base.indices.items()):
                p = base.sy.inverse_transform(prever(modelo, base, ids)).ravel()
                o = df[ALVO].iloc[ids+horizonte].to_numpy()
                datas = df.index[ids+horizonte]
                pd.DataFrame(dict(data_origem=df.index[ids], data_alvo=datas, observado=o, previsto=p)).to_csv(destino/f"previsoes_{bloco}.csv", index=False)
                resultados.append(dict(cenario=cenario, horizonte_min=horizonte*10, bloco=bloco,
                                       camadas=camadas, unidades=unidades, n=len(ids), **metricas(o, p)))
                contagens[bloco] = dict(n=len(ids), primeiro_alvo=str(datas[0]), ultimo_alvo=str(datas[-1]))
                ax.plot(datas, o, label="Observado", lw=.8)
                ax.plot(datas, p, label="MLP", lw=.8)
                ax.set(title=bloco, ylabel="Nivel (m)")
                ax.legend()
            fig.savefig(destino/"observado_previsto.png", dpi=150)
            plt.close(fig)
            fig, axs = plt.subplots(2, 1, figsize=(8, 6), constrained_layout=True)
            axs[0].plot(hist.epoca, hist.mse_treino, label="Treino")
            axs[0].plot(hist.epoca, hist.mse_validacao, label="Validacao")
            axs[0].set_ylabel("MSE normalizado")
            axs[0].legend()
            axs[1].plot(hist.epoca, hist.lr)
            axs[1].set(xlabel="Epoca", ylabel="Learning rate", yscale="log")
            fig.savefig(destino/"aprendizado.png", dpi=150)
            plt.close(fig)
            auditoria.append(dict(cenario=cenario, horizonte_passos=horizonte, janelas=base.spec, blocos=contagens))
            pd.DataFrame(resultados).to_csv(SAIDA/"metricas.csv", index=False)
    config = dict(dados=str(DADOS.resolve()), dispositivo=str(DEVICE), seed=SEMENTE,
                  camadas=CAMADAS, neuronios=NEURONIOS, epocas=EPOCAS, batch=BATCH_SIZE,
                  lr=LR, paciencia=PACIENCIA, paciencia_lr=PACIENCIA_LR, passo_treino=PASSO_TREINO,
                  split=[.70,.15,.15], base_previamente_interpolada=True, experimentos=auditoria)
    (SAIDA/"experimento.json").write_text(json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8")
    print(pd.DataFrame(resultados).query("bloco == 'teste'").to_string(index=False))
    print("Resultados:", SAIDA)


if __name__ == "__main__":
    main()

# -*- coding: utf-8 -*-
"""Case study 3: previsão de afluência e potencial energético equivalente.

Janela: T-6 ... T. Três LSTMs independentes: Q(T+1), Q(T+3), Q(T+7).
Execute no Spyder com F5 ou: python EstudodeCaso03_Grid_LSTM_LR.py
Coloque dataset_filled.csv na mesma pasta deste script.
"""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
from pathlib import Path
import argparse
import hashlib
import json
import pickle
import random
from itertools import product
import time
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset
from torch import nn
from sklearn.preprocessing import StandardScaler
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# %% 1. Configuração do estudo
JANELA = 7
HORIZONTES = [1, 3, 7]
CAMADAS_GRID = [1, 2, 3]
HIDDEN_GRID = [64, 128, 256]
BATCH_SIZE = 128  # Cerca de 44 atualizações/época para ~5.600 exemplos.
LR_INICIAL = 1e-3
LR_MINIMO = 1e-6
FATOR_REDUCAO_LR = 0.5
PACIENCIA_LR = 5  # PyTorch reduz após mais de 5 épocas sem melhora (6).
MELHORA_MINIMA = 1e-5  # MSE normalizado; mesmo critério no scheduler e early stopping.
EMBARALHAR_TREINO = True  # Embaralha exemplos completos, nunca dias dentro da janela.
EPOCAS = 200
PACIENCIA = 25  # Dá tempo para o LR ser reduzido antes de encerrar.
SEMENTE = 42
ALVO = 'Q_Afluente'
# False preserva os 20 preditores do case study 1 (11 P e 9 Q).
# True acrescenta a afluência histórica, supondo disponibilidade ao fim do dia T.
INCLUIR_AFLUENCIA_HISTORICA = True
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Opção 1: cenário didático, não reconstrução da geração real da usina.
H_REF_M = 56.8   # Hipótese: diferença 572,50 - 515,70; NÃO é queda líquida nominal.
RENDIMENTO = 0.90  # Hipótese de rendimento global constante.
P_INST_MW = 396.0  # Potência instalada informada no PAE, pág. 7.
FATOR_MW_POR_M3S = 0.00981 * RENDIMENTO * H_REF_M


def energia_equivalente(q):
    """Retorna potência MW e energia MWh do DIA-ALVO, não acumulado do horizonte."""
    potencia = np.minimum(FATOR_MW_POR_M3S * np.maximum(np.asarray(q), 0), P_INST_MW)
    return potencia, 24 * potencia


def fixar_semente():
    random.seed(SEMENTE)
    np.random.seed(SEMENTE)
    torch.manual_seed(SEMENTE)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEMENTE)
    torch.set_num_threads(2)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)


# %% 2. Leitura e auditoria: nenhum novo preenchimento é feito
def ler_dados(caminho):
    dados = pd.read_csv(caminho, parse_dates=['data']).set_index('data')
    if not dados.index.is_unique:
        raise ValueError('Datas duplicadas: resolver na preparação antes do treinamento.')
    dados = dados.sort_index()
    if not dados.index.equals(pd.date_range(dados.index.min(), dados.index.max(), name='data')):
        raise ValueError('Calendário não diário/contínuo. Não se pode comprimir lacunas.')
    if ALVO not in dados:
        raise ValueError('Coluna Q_Afluente ausente.')
    features = [c for c in dados if c.startswith(('P_', 'Q_')) and c != ALVO]
    if INCLUIR_AFLUENCIA_HISTORICA:
        features += [ALVO]
    dados = dados[features + ([] if ALVO in features else [ALVO])].apply(pd.to_numeric, errors='raise')
    invalidos = (~np.isfinite(dados)) | (dados < 0)
    registro = []
    for c in dados:
        for data in dados.index[invalidos[c]]:
            registro.append(dict(data=str(data.date()), coluna=c, valor=str(dados.loc[data,c])))
    # Valor negativo de afluência pode decorrer de balanço/medição; não o tratamos
    # como vazão física nem o substituímos arbitrariamente por zero como alvo.
    dados = dados.mask(invalidos)
    n = len(dados)
    c1, c2 = int(.70*n), int(.85*n)
    blocos = np.array(['treino']*c1 + ['validacao']*(c2-c1) + ['teste']*(n-c2))
    return dados, features, blocos, registro


# %% 3. Exemplos: sete dias de cada variável e um alvo futuro
def criar_sequencias(dados, features, blocos, horizonte):
    X, Y, origens, alvos = [], [], [], []
    valores = dados[features].to_numpy(float)
    q = dados[ALVO].to_numpy(float)
    for t in range(JANELA-1, len(dados)-horizonte):
        j = t + horizonte
        # Purga das fronteiras: origem e alvo pertencem ao mesmo bloco.
        # Histórico anterior ao começo do bloco é permitido por já ser conhecido.
        if blocos[t] != blocos[j]:
            continue
        janela = valores[t-JANELA+1:t+1]
        # Entradas e alvo precisam ser válidos.
        if not np.isfinite(janela).all() or not np.isfinite(q[j]):
            continue
        X.append(janela)
        Y.append(q[j])
        origens.append(t)
        alvos.append(j)
    X, Y = np.asarray(X), np.asarray(Y).reshape(-1,1)
    origens, alvos = np.asarray(origens), np.asarray(alvos)
    if len(Y)==0:
        raise ValueError('Sem exemplos válidos.')
    treino = blocos[alvos]=='treino'
    if not treino.any():
        raise ValueError('Sem exemplos de treino.')
    sx = StandardScaler().fit(X[treino].reshape(-1,len(features)))
    sy = StandardScaler().fit(Y[treino])
    X = sx.transform(X.reshape(-1,len(features))).reshape(-1,JANELA,len(features))
    Y = sy.transform(Y)
    conjuntos = {}
    for nome in ['treino','validacao','teste']:
        m = blocos[alvos]==nome
        if m.sum()<10:
            raise ValueError(f'Menos de 10 exemplos em {nome}.')
        conjuntos[nome] = dict(x=torch.tensor(X[m],dtype=torch.float32),
                               y=torch.tensor(Y[m],dtype=torch.float32),
                               origem=origens[m], alvo=alvos[m])
    return conjuntos, sx, sy


# %% 4. Rede LSTM: saída direta para um horizonte
class LSTM(nn.Module):
    def __init__(self, n_features, camadas, hidden_size):
        super().__init__()
        self.lstm = nn.LSTM(n_features,hidden_size,num_layers=camadas,batch_first=True)
        self.saida = nn.Sequential(nn.Linear(hidden_size,16),nn.ReLU(),nn.Linear(16,1))

    def forward(self,x):
        _, (h, _) = self.lstm(x)
        return self.saida(h[-1])


@torch.no_grad()
def prever(modelo,x):
    modelo.eval()
    return np.concatenate([modelo(x[i:i+BATCH_SIZE].to(DEVICE)).cpu().numpy()
                           for i in range(0,len(x),BATCH_SIZE)])


# %% 5. Treino e early stopping: o teste não participa
def treinar(conjuntos,n_features,epocas,camadas,hidden_size):
    fixar_semente()
    modelo = LSTM(n_features,camadas,hidden_size).to(DEVICE)
    otimizador = torch.optim.Adam(modelo.parameters(),lr=LR_INICIAL,weight_decay=1e-5)
    # Adam adapta as atualizações por parâmetro. Este scheduler ajusta o LR global:
    # se a validação estacionar, reduz pela metade para tentar passos menores.
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        otimizador, mode='min', factor=FATOR_REDUCAO_LR,
        patience=PACIENCIA_LR, threshold=MELHORA_MINIMA,
        threshold_mode='abs', min_lr=LR_MINIMO)
    loss_fn = nn.MSELoss()
    x,y = conjuntos['treino']['x'],conjuntos['treino']['y']
    # A separação treino/validação/teste continua cronológica. As janelas são
    # independentes (estado LSTM reiniciado em cada forward), então podemos
    # misturá-las SOMENTE no treino, evitando lotes dominados por uma estação do ano.
    gerador = torch.Generator().manual_seed(SEMENTE)
    loader = DataLoader(TensorDataset(x,y),batch_size=BATCH_SIZE,
                        shuffle=EMBARALHAR_TREINO,generator=gerador,
                        num_workers=0,pin_memory=DEVICE.type=='cuda',drop_last=False)
    melhor,espera = float('inf'),0
    historico = []
    for epoca in range(1,epocas+1):
        modelo.train()
        lr_usado = otimizador.param_groups[0]['lr']
        soma = 0.
        for xb,yb in loader:
            xb,yb = xb.to(DEVICE,non_blocking=True),yb.to(DEVICE,non_blocking=True)
            otimizador.zero_grad(set_to_none=True)
            loss = loss_fn(modelo(xb),yb)
            if not torch.isfinite(loss):
                raise RuntimeError('Perda de treino não finita.')
            loss.backward()
            nn.utils.clip_grad_norm_(modelo.parameters(),1.)
            otimizador.step()
            soma += loss.item()*len(xb)
        val = float(np.mean((prever(modelo,conjuntos['validacao']['x'])-
                            conjuntos['validacao']['y'].numpy())**2))
        if not np.isfinite(val):
            raise RuntimeError('Perda de validação não finita.')
        # Step só depois de calcular a validação; teste nunca entra aqui.
        scheduler.step(val)
        lr_proximo = otimizador.param_groups[0]['lr']
        historico.append(dict(epoca=epoca,treino=soma/len(x),validacao=val,
                              lr_usado=lr_usado,lr_proxima_epoca=lr_proximo))
        if lr_proximo < lr_usado:
            print(f'Validação estagnou: LR {lr_usado:.2e} -> {lr_proximo:.2e}',flush=True)
        if val < melhor-MELHORA_MINIMA:
            melhor,espera,melhor_epoca = val,0,epoca
            pesos = {k:v.detach().cpu().clone() for k,v in modelo.state_dict().items()}
        else:
            espera += 1
        if epoca==1 or epoca%10==0:
            print(f'Época {epoca}: treino={soma/len(x):.4f}, validação={val:.4f}, LR={lr_usado:.2e}',flush=True)
        if espera>=PACIENCIA:
            break
    modelo.load_state_dict(pesos)
    return modelo,pd.DataFrame(historico),melhor_epoca


# %% 6. Métricas: referência de energia é transformada, não geração medida
def metricas(o,p):
    o,p = np.asarray(o,float),np.asarray(p,float)
    erro = p-o
    den = np.sum((o-o.mean())**2)
    nse = 1-np.sum(erro**2)/den if den>0 else np.nan
    r = np.corrcoef(o,p)[0,1] if o.std()>0 and p.std()>0 else np.nan
    alpha = p.std()/o.std() if o.std()>0 else np.nan
    beta = p.mean()/o.mean() if o.mean()!=0 else np.nan
    return dict(RMSE=float(np.sqrt(np.mean(erro**2))),MAE=float(np.mean(abs(erro))),
                NSE=float(nse),R2=float(nse),KGE=float(1-np.sqrt((r-1)**2+(alpha-1)**2+(beta-1)**2)),
                vies=float(erro.mean()),PBIAS=float(100*erro.sum()/o.sum()) if o.sum()!=0 else np.nan)


def exportar_resultados(modelo,conjuntos,sy,dados,blocos,horizonte,pasta):
    linhas, tabelas = [], []
    fig,eixos = plt.subplots(3,2,figsize=(15,10),constrained_layout=True)
    for e,(nome,c) in enumerate(conjuntos.items()):
        q_bruto = sy.inverse_transform(prever(modelo,c['x'])).ravel()
        q_pred = np.maximum(q_bruto,0)
        q_ref = dados[ALVO].iloc[c['alvo']].to_numpy()
        p_ref,e_ref = energia_equivalente(q_ref)
        p_pred,e_pred = energia_equivalente(q_pred)
        tabela = pd.DataFrame(dict(data_origem=dados.index[c['origem']],data_alvo=dados.index[c['alvo']],
            horizonte_dias=horizonte,bloco=nome,Q_referencia=q_ref,Q_prevista=q_pred,
            Q_prevista_bruta=q_bruto,P_referencia_MW=p_ref,
            P_prevista_MW=p_pred,E_referencia_MWh=e_ref,
            E_prevista_MWh=e_pred))
        tabela.to_csv(pasta/f'previsoes_{nome}.csv',index=False)
        tabelas.append(tabela)
        for variavel,ref,pred in [('Q',q_ref,q_pred),('E',e_ref,e_pred)]:
            linhas.append(dict(horizonte=horizonte,bloco=nome,variavel=variavel,modelo='LSTM',
                               n=len(ref),**metricas(ref,pred),delta_soma=float(np.sum(pred-ref))))
        grade = tabela.set_index('data_alvo').reindex(dados.index[blocos==nome])
        for ax,cols,unidade in [(eixos[e,0],['Q_referencia','Q_prevista'],'Vazão (m³/s)'),
                               (eixos[e,1],['E_referencia_MWh','E_prevista_MWh'],'Energia equivalente (MWh/dia)')]:
            for col,label,cor in zip(cols,['Referência','LSTM'],['#174d75','#e97732']):
                ax.plot(grade.index,grade[col],label=label,color=cor,lw=.8,alpha=.85)
            ax.set(title=f'{nome} — T+{horizonte}',ylabel=unidade)
            ax.grid(alpha=.2)
            ax.legend(fontsize=8)
    fig.savefig(pasta/'vazao_energia.png',dpi=150)
    plt.close(fig)
    return linhas,pd.concat(tabelas,ignore_index=True)


def plotar_aprendizado(hist,melhor_epoca,caminho,titulo):
    fig,(ax,lr_ax) = plt.subplots(2,1,figsize=(9,6),sharex=True,constrained_layout=True)
    ax.plot(hist.epoca,hist.treino,label='Treino (média dos lotes)')
    ax.plot(hist.epoca,hist.validacao,label='Validação (fim da época)')
    ax.axvline(melhor_epoca,color='gray',ls='--',label='Melhor época')
    ax.set(ylabel='MSE normalizado',title=titulo)
    ax.legend()
    lr_ax.step(hist.epoca,hist.lr_usado,where='post')
    lr_ax.set(xlabel='Época',ylabel='Learning rate',yscale='log')
    fig.savefig(caminho,dpi=150)
    plt.close(fig)


# %% 7. Executar três modelos e salvar tudo para uso em aula
def main():
    pasta = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dados',type=Path,default=pasta/'dataset_filled.csv')
    parser.add_argument('--out',type=Path,default=pasta/'resultados_grid_lr_energia')
    parser.add_argument('--epocas',type=int,default=EPOCAS)
    args = parser.parse_args()
    if args.epocas<1:
        raise ValueError('Épocas deve ser positivo.')
    args.out.mkdir(parents=True,exist_ok=True)
    dados,features,blocos,invalidos = ler_dados(args.dados)
    pd.DataFrame(invalidos,columns=['data','coluna','valor']).to_csv(args.out/'valores_invalidos.csv',index=False)
    print('Dispositivo:',torch.cuda.get_device_name(0) if DEVICE.type=='cuda' else 'CPU',flush=True)
    print('Preditores:',features,flush=True)
    print('ATENÇÃO: base previamente preenchida; a origem das imputações não está documentada.',flush=True)
    cortes = []
    for nome in ['treino','validacao','teste']:
        datas = dados.index[blocos==nome]
        cortes.append(dict(bloco=nome,inicio=str(datas.min().date()),fim=str(datas.max().date()),dias=len(datas)))
    print(pd.DataFrame(cortes).to_string(index=False),flush=True)
    auditoria = dict(inicio=str(dados.index.min().date()),fim=str(dados.index.max().date()),dias=len(dados),
        preditores=features,janela=JANELA,horizontes=HORIZONTES,cortes=cortes,
        base_preenchida_sem_mascara=True,sha256=hashlib.sha256(args.dados.read_bytes()).hexdigest(),
        energia=dict(H_ref_m=H_REF_M,rendimento=RENDIMENTO,P_inst_MW=P_INST_MW,
                     coeficiente=FATOR_MW_POR_M3S,Q_saturacao=P_INST_MW/FATOR_MW_POR_M3S,
                     interpretacao='Energia equivalente de 24 horas do dia-alvo, não acumulada até o horizonte'),
        treinamento=dict(camadas_grid=CAMADAS_GRID,hidden_grid=HIDDEN_GRID,criterio="MSE de vazão na validação, antes de clipping",lote=BATCH_SIZE,epocas_max=args.epocas,
                         paciencia=PACIENCIA,lr_inicial=LR_INICIAL,lr_minimo=LR_MINIMO,
                         fator_lr=FATOR_REDUCAO_LR,paciencia_lr=PACIENCIA_LR,
                         melhora_minima=MELHORA_MINIMA,embaralhar_treino=EMBARALHAR_TREINO,
                         semente=SEMENTE,torch=torch.__version__,dispositivo=str(DEVICE)))
    todas_metricas,todas_previsoes,resumo = [],[],[]
    for h in HORIZONTES:
        print(f'\n=== Horizonte T+{h} ===',flush=True)
        destino = args.out/f'Tmais{h}'
        destino.mkdir(exist_ok=True)
        conjuntos,sx,sy = criar_sequencias(dados,features,blocos,h)
        for nome,c in conjuntos.items():
            print(nome,tuple(c['x'].shape),flush=True)
        # 9 arquiteturas por horizonte; o teste não entra nesta busca.
        ranking = []
        melhor_val = float('inf')
        q_val = dados[ALVO].iloc[conjuntos['validacao']['alvo']].to_numpy()
        for camadas,hidden_size in product(CAMADAS_GRID,HIDDEN_GRID):
            print(f'Grid T+{h}: camadas={camadas}, hidden_size={hidden_size}',flush=True)
            inicio = time.perf_counter()
            candidato,hist,ep = treinar(conjuntos,len(features),args.epocas,camadas,hidden_size)
            estimativa = sy.inverse_transform(prever(candidato,conjuntos['validacao']['x'])).ravel()
            mse = float(np.mean((estimativa-q_val)**2))
            registro = dict(camadas=camadas,hidden_size=hidden_size,melhor_epoca=ep,
                            epocas_executadas=len(hist),mse_validacao=mse,
                            segundos=time.perf_counter()-inicio,**metricas(q_val,estimativa))
            ranking.append(registro)
            hist.to_csv(destino/f'historico_L{camadas}_H{hidden_size}.csv',index=False)
            plotar_aprendizado(hist,ep,destino/f'aprendizado_L{camadas}_H{hidden_size}.png',f'T+{h}: L{camadas} H{hidden_size}')
            pd.DataFrame(ranking).sort_values('mse_validacao').to_csv(destino/'ranking_validacao.csv',index=False)
            if mse < melhor_val:
                melhor_val = mse
                vencedor = registro
                historico,epoca = hist.copy(),ep
                pesos = {k:v.detach().cpu().clone() for k,v in candidato.state_dict().items()}
            del candidato
            if DEVICE.type=='cuda':
                torch.cuda.empty_cache()
        modelo = LSTM(len(features),vencedor['camadas'],vencedor['hidden_size']).to(DEVICE)
        modelo.load_state_dict(pesos)
        print('Vencedor pela validação:',vencedor,flush=True)
        (destino/'vencedor_validacao.json').write_text(json.dumps(vencedor,indent=2),encoding='utf-8')
        historico.to_csv(destino/'historico.csv',index=False)
        plotar_aprendizado(historico,epoca,destino/'aprendizado.png',f'Vencedor T+{h}')
        torch.save(dict(pesos={k:v.detach().cpu() for k,v in modelo.state_dict().items()},
                        features=features,janela=JANELA,horizonte=h,camadas=vencedor['camadas'],hidden_size=vencedor['hidden_size']),
                   destino/'modelo.pt')
        with open(destino/'scalers.pkl','wb') as f:
            pickle.dump(dict(x=sx,y=sy,features=features),f)
        met,prev = exportar_resultados(modelo,conjuntos,sy,dados,blocos,h,destino)
        todas_metricas.extend(met)
        todas_previsoes.append(prev)
        resumo.append(dict(horizonte=h,camadas=vencedor['camadas'],hidden_size=vencedor['hidden_size'],melhor_epoca=epoca,epocas_executadas=len(historico),
                           **{nome:len(c['x']) for nome,c in conjuntos.items()}))
    pd.DataFrame(todas_metricas).to_csv(args.out/'metricas.csv',index=False)
    pd.DataFrame(resumo).to_csv(args.out/'resumo_treinamento.csv',index=False)
    previsoes = pd.concat(todas_previsoes,ignore_index=True)
    previsoes.to_csv(args.out/'previsoes_todos_horizontes.csv',index=False)
    # Demonstração: os três horizontes emitidos na mesma data de teste.
    teste = previsoes[previsoes.bloco=='teste']
    datas_completas = teste.groupby('data_origem').horizonte_dias.nunique()
    origem = datas_completas[datas_completas==len(HORIZONTES)].index.min()
    exemplo = teste[teste.data_origem==origem].sort_values('horizonte_dias')
    exemplo.to_csv(args.out/'exemplo_mesma_origem.csv',index=False)
    auditoria['modelos'] = resumo
    (args.out/'experimento.json').write_text(json.dumps(auditoria,ensure_ascii=False,indent=2),encoding='utf-8')
    print(pd.DataFrame(todas_metricas).query("bloco=='teste'").round(3).to_string(index=False),flush=True)
    print('Resultados em:',args.out,flush=True)


if __name__=='__main__':
    main()

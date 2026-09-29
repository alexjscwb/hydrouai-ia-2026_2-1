# -*- coding: utf-8 -*-
import pickle
import torch
from torch import nn
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import warnings
import matplotlib.pyplot as plt
from math import ceil

warnings.filterwarnings('ignore')

# ============================
# Dispositivo
# ============================
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# ============================
# Dados
# ============================
df = pd.read_csv(
    r"C:\Users\Aquasmart EHR\Desktop\Rodrigo\Tese\dados_interpolados.txt",
    sep='\t'
)
df = df.set_index("DATA")
df.index = pd.to_datetime(df.index)

target_col = 'FLU(m)46'            # estação exutório (alvo)
y_full     = df[target_col].values # série alvo (para t+T)

# ============================
# Configurações gerais
# ============================
hidden_layers = [1, 2, 3]
hidden_sizes  = [64, 128, 256]
T_list        = [54]   # 10-min steps

H = 48                   # 6 horas (amostragem de 10 min) -> janela de entrada
halfH = H // 2           # 18 passos
train_frac = 0.8

# Batch global (mantido)
train_size_global = int(len(df) * train_frac)
num_batches = 25
batch_size  = train_size_global // num_batches  # ajuste se necessário

# ============================
# Config. FÍSICA por T
# - chaves: T (em passos de 10 min)
# - valor: { nome_da_coluna: centro_em_passos }
#   centro=36 com H=36 => janela [t-54, t-18) (centralizada 6h atrás do t)
#   A '46' será adicionada AUTOMATICAMENTE como trailing [t-36, t) com centro=H/2
# ============================
PHYS_CONFIG = {
    6:  { 'FLU(m)713': 36 ,'PLU(mm)713':36,'PLU(mm)46':18},   # só 713, centrada em 6h
    18:  { 'FLU(m)713': 36 ,'PLU(mm)713':36,'PLU(mm)46':18},   # só 713, centrada em 6h
    36:  { 'FLU(m)713': 36 ,'PLU(mm)713':36,'PLU(mm)46':18},   # só 713, centrada em 6h
    53:  { 'FLU(m)713': 36 ,'PLU(mm)713':36,'FLU(m)48': 54 ,'PLU(mm)48':54,'FLU(m)57': 48 ,'PLU(mm)57':48,'FLU(m)59': 60 ,'PLU(mm)59':60,'FLU(m)50': 54 ,'PLU(mm)50':54,'PLU(mm)46':18} ,   # só 713, centrada em 6h
    54:  { 'FLU(m)48': 54 ,'PLU(mm)48':54,'FLU(m)57': 48 ,'PLU(mm)57':48,'FLU(m)59': 60 ,'PLU(mm)59':60,'FLU(m)50': 54 ,'PLU(mm)50':54,'PLU(mm)46':18} ,   # só 713, centrada em 6h

    72:  { 'FLU(m)713': 36 ,'PLU(mm)713':36,'FLU(m)48': 54 ,'PLU(mm)48':54,'FLU(m)57': 48 ,'PLU(mm)57':48,'FLU(m)59': 60 ,'PLU(mm)59':60,'FLU(m)50': 54 ,'PLU(mm)50':54,'PLU(mm)46':18} ,   # só 713, centrada em 6h

    
   
    # 36: { 'FLU(m)713': 36, 'FLU(m)57': 90 },
    # 54: { 'FLU(m)713': 36, 'FLU(m)57': 90, 'FLU(m)48': 120 },
    # 72: { 'FLU(m)713': 36, 'FLU(m)57': 90, 'FLU(m)48': 120, 'FLU(m)XXX': ccc },
}

# ============================
# Métricas
# ============================
@torch.no_grad()
def nse_torch(y_true: torch.Tensor, y_pred: torch.Tensor, eps: float = 1e-8) -> float:
    y_true = y_true.view(-1)
    y_pred = y_pred.view(-1)
    denom = torch.sum((y_true - torch.mean(y_true)) ** 2) + eps
    num   = torch.sum((y_true - y_pred) ** 2)
    return float((1.0 - num / denom).detach().cpu())

@torch.no_grad()
def kge_torch(y_true: torch.Tensor, y_pred: torch.Tensor, eps: float = 1e-8) -> float:
    y_true = y_true.view(-1); y_pred = y_pred.view(-1)
    mu_o = torch.mean(y_true); mu_s = torch.mean(y_pred)
    sd_o = torch.std(y_true) + eps; sd_s = torch.std(y_pred) + eps
    r_num = torch.sum((y_true - mu_o) * (y_pred - mu_s))
    r_den = (torch.sqrt(torch.sum((y_true - mu_o)**2)) + eps) * (torch.sqrt(torch.sum((y_pred - mu_s)**2)) + eps)
    r = r_num / r_den
    alpha = sd_s / sd_o
    beta  = mu_s / (mu_o + eps)
    return float((1.0 - torch.sqrt((r-1.0)**2 + (alpha-1.0)**2 + (beta-1.0)**2)).detach().cpu())

# ============================
# Early Stopping flexível (NSE)
# ============================
class EarlyStoppingFlexible:
    def __init__(self, mode='max', rel_min_delta=0.005, patience=20, smooth=3, restore_best=True):
        assert mode in ('min', 'max')
        self.mode = mode
        self.rel_min_delta = rel_min_delta
        self.patience = patience
        self.smooth = max(1, int(smooth))
        self.restore_best = restore_best
        self.best = None
        self.counter = 0
        self.best_state = None
        self.hist = []

    def _is_better(self, current):
        if self.best is None:
            return True
        if self.mode == 'min':
            return (self.best - current) / (abs(self.best) + 1e-12) >= self.rel_min_delta
        else:
            return (current - self.best) / (abs(self.best) + 1e-12) >= self.rel_min_delta

    def step(self, current_value, model=None):
        self.hist.append(float(current_value))
        current = sum(self.hist[-self.smooth:]) / self.smooth if len(self.hist) >= self.smooth else self.hist[-1]
        if self._is_better(current):
            self.best = current
            self.counter = 0
            if self.restore_best and model is not None:
                self.best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            return False
        else:
            self.counter += 1
            return self.counter >= self.patience

    def restore(self, model):
        if self.restore_best and self.best_state is not None:
            model.load_state_dict(self.best_state)

# ============================
# Modelo
# ============================
class LSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, layer_dim, output_dim):
        super(LSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.layer_dim = layer_dim
        self.rnn = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=layer_dim,
            batch_first=True
        )
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, X):
        h0 = torch.zeros(self.layer_dim, X.size(0), self.hidden_dim).to(device)
        c0 = torch.zeros(self.layer_dim, X.size(0), self.hidden_dim).to(device)
        out, _ = self.rnn(X, (h0.detach(), c0.detach()))
        out = self.fc(out[:, -1, :])
        return out

# ============================
# Helper: construção de janelas físicas
# ============================
def build_phys_windowed_arrays(X_df_scaled: pd.DataFrame,
                               y_np: np.ndarray,
                               T: int,
                               centers_by_col: dict,
                               H: int,
                               train_frac: float):
    """
    Constrói X_train, y_train, X_val, y_val usando janelas por-coluna
    centralizadas em 'centers_by_col[col]' (passos de 10 min).
    Para trailing [t-H, t) use center = H//2.
    """
    cols = list(centers_by_col.keys())
    for c in cols:
        if c not in X_df_scaled.columns:
            raise KeyError(f"Coluna '{c}' não encontrada em X_df_scaled.")

    # índice mínimo de t para janelas válidas
    min_t = max(centers_by_col[c] + (H // 2) for c in cols)
    N = len(X_df_scaled) - T  # último índice válido para y[t+T]

    # split cronológico
    split = int(N * train_frac)

    # tamanhos efetivos
    n_train = max(0, split - min_t)  # t em [min_t .. split-1]
    n_val   = max(0, N - split)      # t em [split .. N-1]

    D_eff = len(cols)
    X_train = np.zeros((n_train, H, D_eff), dtype=np.float32)
    y_train = np.zeros((n_train, 1), dtype=np.float32)
    X_val   = np.zeros((n_val, H, D_eff), dtype=np.float32)
    y_val   = np.zeros((n_val, 1), dtype=np.float32)

    # TRAIN
    for k, t in enumerate(range(min_t, split)):
        for j, col in enumerate(cols):
            c = centers_by_col[col]
            start = t - (c + (H // 2))
            end   = start + H
            X_train[k, :, j] = X_df_scaled[col].values[start:end]
        y_train[k, 0] = y_np[t + T]

    # VAL
    for k, t in enumerate(range(split, N)):
        for j, col in enumerate(cols):
            c = centers_by_col[col]
            start = t - (c + (H // 2))
            end   = start + H
            if start < 0:
                raise RuntimeError("Janela negativa em validação; revise centers/H.")
            X_val[k, :, j] = X_df_scaled[col].values[start:end]
        y_val[k, 0] = y_np[t + T]

    return X_train, y_train, X_val, y_val, cols, min_t, N, split

# ============================
# Grid Search (Cenário II - físico)
# ============================
results = []

for hidden_layer in hidden_layers:
    for hidden_size in hidden_sizes:
        for T in T_list:
            if T not in PHYS_CONFIG:
                print(f"⚠️ PULANDO T={T}: sem configuração física em PHYS_CONFIG.")
                continue

            # ------ monta centros: montantes + 46 trailing ------
            centers_up  = PHYS_CONFIG[T]        # definido por você (estações de montante)
            centers_all = dict(centers_up)      # copia
            centers_all[target_col] = H // 2    # adiciona 46 como trailing [t-H, t)

            cols_all = list(centers_all.keys())

            try:
                print(f"\nConfig (Phys-II): L={hidden_layer}, U={hidden_size}, T={T} | cols={cols_all}")

                # N efetivo para o split (depende de T)
                N_eff = len(df) - T
                split = int(N_eff * train_frac)

                # ------ Scaler SÓ no treino: nas colunas selecionadas (montantes + 46) ------
                X_sel_train = df[cols_all].iloc[:split].values
                scaler = StandardScaler().fit(X_sel_train)

                # Transformar TODO o intervalo [0:N_eff] nessas colunas
                X_sel_all = df[cols_all].iloc[:N_eff].copy()
                X_sel_all.loc[:, :] = scaler.transform(X_sel_all.values).astype(np.float32)

                # ------ Construção das janelas físicas ------
                X_train, y_train, X_val, y_val, used_cols, min_t, N, split2 = build_phys_windowed_arrays(
                    X_df_scaled=X_sel_all,
                    y_np=y_full,
                    T=T,
                    centers_by_col=centers_all,
                    H=H,
                    train_frac=train_frac
                )

                if X_train.shape[0] == 0 or X_val.shape[0] == 0:
                    print(f"⚠️ T={T}: poucas amostras após alinhamento físico (min_t={min_t}). Pulando.")
                    continue

                # ===== Modelo / otimizador / loss =====
                D_eff = X_train.shape[2]
                model = LSTM(input_dim=D_eff, hidden_dim=hidden_size, layer_dim=hidden_layer, output_dim=1).to(device)
                criterion = nn.MSELoss()
                optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)

                early = EarlyStoppingFlexible(mode='max', rel_min_delta=0.005, patience=20, smooth=3, restore_best=True)

                # ===== Treino =====
                epochs = 100
                train_losses, val_losses = [], []

                for epoch in range(1, epochs + 1):
                    model.train()
                    n_tr = X_train.shape[0]
                    perm = np.random.permutation(n_tr)
                    epoch_loss = 0.0

                    for i in range(0, n_tr, batch_size):
                        idx = perm[i:i+batch_size]
                        xb = torch.from_numpy(X_train[idx]).to(device, non_blocking=True)
                        yb = torch.from_numpy(y_train[idx]).to(device, non_blocking=True)

                        optimizer.zero_grad(set_to_none=True)
                        pred = model(xb)
                        loss = criterion(pred, yb)
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                        optimizer.step()

                        epoch_loss += float(loss.item()) * len(idx)
                        del xb, yb, pred, loss

                    epoch_loss /= n_tr
                    train_losses.append(epoch_loss)

                    # ===== Validação em batches =====
                    model.eval()
                    with torch.no_grad():
                        n_va = X_val.shape[0]
                        val_loss_acc = 0.0
                        preds_all, true_all = [], []

                        for c in range(ceil(n_va / batch_size)):
                            s = c * batch_size
                            e = min(n_va, s + batch_size)
                            xb = torch.from_numpy(X_val[s:e]).to(device, non_blocking=True)
                            yb = torch.from_numpy(y_val[s:e]).to(device, non_blocking=True)
                            vb_pred = model(xb)
                            val_loss_acc += float(criterion(vb_pred, yb).item()) * (e - s)
                            preds_all.append(vb_pred.detach().cpu())
                            true_all.append(yb.detach().cpu())
                            del xb, yb, vb_pred

                        val_loss = val_loss_acc / n_va
                        val_losses.append(val_loss)

                        preds_all = torch.vstack(preds_all)
                        true_all  = torch.vstack(true_all)
                        val_nse   = nse_torch(true_all, preds_all)

                    print(f"Época {epoch:03d} | Train MSE: {train_losses[-1]:.4f} | Val MSE: {val_loss:.4f} | Val NSE: {val_nse:.4f}")

                    # (opcional) LR manual por época
                    if epoch in [10, 20, 50, 75, 100]:
                        new_lr = {10: 1e-3, 20: 1e-3, 50: 5e-4, 75: 1e-5, 100: 1e-5}[epoch]
                        for g in optimizer.param_groups:
                            g['lr'] = new_lr
                        print(f"  -> LR ajustado para {new_lr:.0e} na época {epoch}")

                    if early.step(val_nse, model):
                        print(f"  -> Early stopping (NSE) na época {epoch} | best NSE ~ {early.best:.5f}")
                        early.restore(model)
                        break

                    torch.cuda.empty_cache()

                # ===== Avaliação final agregada (val) =====
                model.eval()
                with torch.no_grad():
                    n_va = X_val.shape[0]
                    preds_all, true_all = [], []
                    for c in range(ceil(n_va / batch_size)):
                        s = c * batch_size
                        e = min(n_va, s + batch_size)
                        xb = torch.from_numpy(X_val[s:e]).to(device, non_blocking=True)
                        yb = torch.from_numpy(y_val[s:e]).to(device, non_blocking=True)
                        vb_pred = model(xb)
                        preds_all.append(vb_pred.detach().cpu())
                        true_all.append(yb.detach().cpu())
                        del xb, yb, vb_pred
                    preds_all = torch.vstack(preds_all)
                    true_all  = torch.vstack(true_all)

                    val_mse = float(nn.MSELoss()(preds_all, true_all).item())
                    val_mae = float(torch.mean(torch.abs(preds_all - true_all)).item())
                    val_nse = nse_torch(true_all, preds_all)
                    val_kge = kge_torch(true_all, preds_all)

                results.append({
                    'scenario': 'Phys-II',
                    'layers': hidden_layer,
                    'units': hidden_size,
                    'T': T,
                    'H': H,
                    'cols': '|'.join(used_cols),  # inclui 46 + montantes
                    'val_MSE': val_mse,
                    'val_MAE': val_mae,
                    'NSE': val_nse,
                    'KGE': val_kge
                })

                # ===== Salvar MODELO + SCALER =====
                model_filename  = f"physII_L{hidden_layer}_U{hidden_size}_T{T}.pkl"
                scaler_filename = f"physII_scaler_L{hidden_layer}_U{hidden_size}_T{T}.pkl"
                with open(model_filename,  'wb') as f: pickle.dump(model,  f)
                with open(scaler_filename, 'wb') as f: pickle.dump(scaler, f)

                # ===== Figura de perdas =====
                plt.figure(figsize=(8,4))
                plt.plot(train_losses, label='Train MSE')
                plt.plot(val_losses,   label='Val MSE')
                plt.title(f'Phys-II | L={hidden_layer} U={hidden_size} T={T} | cols={len(used_cols)}')
                plt.xlabel('Epoch'); plt.ylabel('MSE')
                plt.ylim(0, 0.5)
                plt.legend(loc='best')
                fig_filename = model_filename.replace('.pkl', '.png')
                plt.savefig(fig_filename, dpi=300, bbox_inches='tight')
                plt.close()

                # CSV parcial
                pd.DataFrame(results).to_csv("grid_search_physII_partial.csv", index=False)

                del model
                torch.cuda.empty_cache()

            except Exception as e:
                print(f"⚠️ Erro na config Phys-II L={hidden_layer}, U={hidden_size}, T={T}: {e}")
                torch.cuda.empty_cache()
                continue

# Resultados finais
df_results = pd.DataFrame(results)
df_results.to_csv('grid_search_physII_results.csv', index=False)
print("Grid search Phys-II concluído. Resultados em 'grid_search_physII_results.csv'.")



# -*- coding: utf-8 -*-
import pickle
import torch
from torch import nn
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import warnings
import matplotlib.pyplot as plt
from math import ceil

warnings.filterwarnings('ignore')

# ============================
# Dispositivo
# ============================
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# ============================
# Dados
# ============================
df = pd.read_csv(
    r"C:\Users\Aquasmart EHR\Desktop\Rodrigo\Tese\dados_interpolados.txt",
    sep='\t'
)
df = df.set_index("DATA")
df.index = pd.to_datetime(df.index)

target_col = 'FLU(m)46'            # estação exutório (alvo)
y_full     = df[target_col].values # série alvo (para t+T)

# ============================
# Configurações gerais
# ============================
hidden_layers = [1, 2, 3]
hidden_sizes  = [128, 256]
T_list        = [ 54]   # 10-min steps

# Comprimento de janela padrão (em steps) para colunas que não estiverem em H_BY_COL
H_DEFAULT = 60   # 8h
# Comprimento de janela por coluna (ex.: 46 com 6h; montantes com 8h)
# Janela por coluna
H_BY_COL = {
    'FLU(m)46': 36,   # exutório, trailing 6h
    'FLU(m)713': 36,
    'PLU(mm)713': 36,
    
    
    
    
    # 713 com 36 passos
    # todas as outras colunas usam H_DEFAULT = 48
}


train_frac = 0.8

# Batch global (mantido)
train_size_global = int(len(df) * train_frac)
num_batches = 25
batch_size  = train_size_global // num_batches  # ajuste se necessário

# ============================
# Config. FÍSICA por T
# - chaves: T (em passos de 10 min)
# - valor: { nome_da_coluna: centro_em_passos }
#   - “centro” c e H_col definem a janela: [t - (c + H_col//2), t - (c - H_col//2))
#   - Para trailing de uma coluna, use c = H_col//2  => [t - H_col, t)
#   - A coluna target (46) será adicionada AUTOMATICAMENTE como trailing com H definido em H_BY_COL
# ============================
PHYS_CONFIG = {
    6:  {'FLU(m)713': 36, 'PLU(mm)713': 36},  
    18: {'FLU(m)713': 36, 'PLU(mm)713': 36},  
    36: {'FLU(m)713': 36, 'PLU(mm)713': 36},  
    54: {'FLU(m)713': 36, 'PLU(mm)713': 36,
         'FLU(m)48': 54, 'PLU(mm)48': 54,
         'FLU(m)57': 48, 'PLU(mm)57': 48}, 
    72: {'FLU(m)713': 36, 'PLU(mm)713': 36,
         'FLU(m)48': 54, 'PLU(mm)48': 54,
         'FLU(m)57': 48, 'PLU(mm)57': 48,
         'FLU(m)59': 60, 'PLU(mm)59': 60,
         'FLU(m)50': 54, 'PLU(mm)50': 54},
}


# ============================
# Métricas
# ============================
@torch.no_grad()
def nse_torch(y_true: torch.Tensor, y_pred: torch.Tensor, eps: float = 1e-8) -> float:
    y_true = y_true.view(-1)
    y_pred = y_pred.view(-1)
    denom = torch.sum((y_true - torch.mean(y_true)) ** 2) + eps
    num   = torch.sum((y_true - y_pred) ** 2)
    return float((1.0 - num / denom).detach().cpu())

@torch.no_grad()
def kge_torch(y_true: torch.Tensor, y_pred: torch.Tensor, eps: float = 1e-8) -> float:
    y_true = y_true.view(-1); y_pred = y_pred.view(-1)
    mu_o = torch.mean(y_true); mu_s = torch.mean(y_pred)
    sd_o = torch.std(y_true) + eps; sd_s = torch.std(y_pred) + eps
    r_num = torch.sum((y_true - mu_o) * (y_pred - mu_s))
    r_den = (torch.sqrt(torch.sum((y_true - mu_o)**2)) + eps) * (torch.sqrt(torch.sum((y_pred - mu_s)**2)) + eps)
    r = r_num / r_den
    alpha = sd_s / sd_o
    beta  = mu_s / (mu_o + eps)
    return float((1.0 - torch.sqrt((r-1.0)**2 + (alpha-1.0)**2 + (beta-1.0)**2)).detach().cpu())

# ============================
# Early Stopping flexível (NSE)
# ============================
class EarlyStoppingFlexible:
    def __init__(self, mode='max', rel_min_delta=0.005, patience=20, smooth=3, restore_best=True):
        assert mode in ('min', 'max')
        self.mode = mode
        self.rel_min_delta = rel_min_delta
        self.patience = patience
        self.smooth = max(1, int(smooth))
        self.restore_best = restore_best
        self.best = None
        self.counter = 0
        self.best_state = None
        self.hist = []

    def _is_better(self, current):
        if self.best is None:
            return True
        if self.mode == 'min':
            return (self.best - current) / (abs(self.best) + 1e-12) >= self.rel_min_delta
        else:
            return (current - self.best) / (abs(self.best) + 1e-12) >= self.rel_min_delta

    def step(self, current_value, model=None):
        self.hist.append(float(current_value))
        current = sum(self.hist[-self.smooth:]) / self.smooth if len(self.hist) >= self.smooth else self.hist[-1]
        if self._is_better(current):
            self.best = current
            self.counter = 0
            if self.restore_best and model is not None:
                self.best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            return False
        else:
            self.counter += 1
            return self.counter >= self.patience

    def restore(self, model):
        if self.restore_best and self.best_state is not None:
            model.load_state_dict(self.best_state)

# ============================
# Modelo
# ============================
class LSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, layer_dim, output_dim):
        super(LSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.layer_dim = layer_dim
        self.rnn = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=layer_dim,
            batch_first=True
        )
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, X):
        h0 = torch.zeros(self.layer_dim, X.size(0), self.hidden_dim).to(device)
        c0 = torch.zeros(self.layer_dim, X.size(0), self.hidden_dim).to(device)
        out, _ = self.rnn(X, (h0.detach(), c0.detach()))
        out = self.fc(out[:, -1, :])
        return out

# ============================
# Helpers de janela física (comprimento variável)
# ============================
def _get_H_for_col(col: str, H_by_col: dict, H_default: int) -> int:
    return int(H_by_col[col]) if col in H_by_col else int(H_default)

def build_phys_windowed_arrays_varH(X_df_scaled: pd.DataFrame,
                                    y_np: np.ndarray,
                                    T: int,
                                    centers_by_col: dict,
                                    H_by_col: dict,
                                    H_default: int,
                                    train_frac: float):
    """
    Constrói X_train, y_train, X_val, y_val usando janelas por-coluna,
    cada uma com seu próprio comprimento H_col.
    A sequência final tem comprimento fixo H_max (= max(H_col)), e
    cada coluna com H_col < H_max é 'right-alinhada' e preenchida com zeros à esquerda.
    """
    cols = list(centers_by_col.keys())
    for c in cols:
        if c not in X_df_scaled.columns:
            raise KeyError(f"Coluna '{c}' não encontrada em X_df_scaled.")

    # H por coluna e H_max
    H_cols = {c: _get_H_for_col(c, H_by_col, H_default) for c in cols}
    H_max  = max(H_cols.values())

    # índice mínimo de t para janelas válidas (depende de cada H_col)
    min_t = max(centers_by_col[c] + (H_cols[c] // 2) for c in cols)
    N = len(X_df_scaled) - T  # último índice válido para y[t+T]

    # split cronológico
    split = int(N * train_frac)

    # tamanhos efetivos
    n_train = max(0, split - min_t)  # t em [min_t .. split-1]
    n_val   = max(0, N - split)      # t em [split .. N-1]

    D_eff = len(cols)
    X_train = np.zeros((n_train, H_max, D_eff), dtype=np.float32)
    y_train = np.zeros((n_train, 1), dtype=np.float32)
    X_val   = np.zeros((n_val, H_max, D_eff), dtype=np.float32)
    y_val   = np.zeros((n_val, 1), dtype=np.float32)

    # TRAIN
    for k, t in enumerate(range(min_t, split)):
        for j, col in enumerate(cols):
            c     = centers_by_col[col]
            H_col = H_cols[col]
            start = t - (c + (H_col // 2))
            end   = start + H_col
            seq = X_df_scaled[col].values[start:end]  # len = H_col

            # Right-align dentro de H_max (padding à esquerda com zeros)
            X_train[k, H_max - H_col: H_max, j] = seq
        y_train[k, 0] = y_np[t + T]

    # VAL
    for k, t in enumerate(range(split, N)):
        for j, col in enumerate(cols):
            c     = centers_by_col[col]
            H_col = H_cols[col]
            start = t - (c + (H_col // 2))
            end   = start + H_col
            if start < 0:
                raise RuntimeError("Janela negativa em validação; revise centers/H_by_col/H_default.")

            seq = X_df_scaled[col].values[start:end]
            X_val[k, H_max - H_col: H_max, j] = seq
        y_val[k, 0] = y_np[t + T]

    return X_train, y_train, X_val, y_val, cols, H_cols, H_max, min_t, N, split

# ============================
# Grid Search (Cenário II - físico com H variável)
# ============================
results = []

for hidden_layer in hidden_layers:
    for hidden_size in hidden_sizes:
        for T in T_list:
            if T not in PHYS_CONFIG:
                print(f"⚠️ PULANDO T={T}: sem configuração física em PHYS_CONFIG.")
                continue

            # ------ centros definidos por T (montantes/chuva etc) ------
            centers_up  = dict(PHYS_CONFIG[T])  # cópia

            # ------ adicionar a coluna alvo como trailing ------
            H_46 = _get_H_for_col(target_col, H_BY_COL, H_DEFAULT)
            centers_up[target_col] = H_46 // 2   # trailing [t-H_46, t)

            cols_all = list(centers_up.keys())

            try:
                print(f"\nConfig (Phys-II varH): L={hidden_layer}, U={hidden_size}, T={T} | cols={cols_all}")

                # N efetivo para split (depende de T)
                N_eff = len(df) - T
                split = int(N_eff * train_frac)

                # ------ Scaler SÓ no treino: nas colunas selecionadas ------
                X_sel_train = df[cols_all].iloc[:split].values
                scaler = StandardScaler().fit(X_sel_train)

                # Transformar TODO o intervalo [0:N_eff] nessas colunas
                X_sel_all = df[cols_all].iloc[:N_eff].copy()
                X_sel_all.loc[:, :] = scaler.transform(X_sel_all.values).astype(np.float32)

                # ------ Construção das janelas físicas com H variável ------
                X_train, y_train, X_val, y_val, used_cols, H_cols, H_max, min_t, N, split2 = build_phys_windowed_arrays_varH(
                    X_df_scaled=X_sel_all,
                    y_np=y_full,
                    T=T,
                    centers_by_col=centers_up,
                    H_by_col=H_BY_COL,
                    H_default=H_DEFAULT,
                    train_frac=train_frac
                )

                if X_train.shape[0] == 0 or X_val.shape[0] == 0:
                    print(f"⚠️ T={T}: poucas amostras após alinhamento físico (min_t={min_t}). Pulando.")
                    continue

                # ===== Modelo / otimizador / loss =====
                D_eff = X_train.shape[2]
                model = LSTM(input_dim=D_eff, hidden_dim=hidden_size, layer_dim=hidden_layer, output_dim=1).to(device)
                criterion = nn.MSELoss()
                optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)

                early = EarlyStoppingFlexible(mode='max', rel_min_delta=0.005, patience=20, smooth=3, restore_best=True)

                # ===== Treino =====
                epochs = 100
                train_losses, val_losses = [], []

                for epoch in range(1, epochs + 1):
                    model.train()
                    n_tr = X_train.shape[0]
                    perm = np.random.permutation(n_tr)
                    epoch_loss = 0.0

                    for i in range(0, n_tr, batch_size):
                        idx = perm[i:i+batch_size]
                        xb = torch.from_numpy(X_train[idx]).to(device, non_blocking=True)
                        yb = torch.from_numpy(y_train[idx]).to(device, non_blocking=True)

                        optimizer.zero_grad(set_to_none=True)
                        pred = model(xb)
                        loss = criterion(pred, yb)
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                        optimizer.step()

                        epoch_loss += float(loss.item()) * len(idx)
                        del xb, yb, pred, loss

                    epoch_loss /= n_tr
                    train_losses.append(epoch_loss)

                    # ===== Validação em batches =====
                    model.eval()
                    with torch.no_grad():
                        n_va = X_val.shape[0]
                        val_loss_acc = 0.0
                        preds_all, true_all = [], []

                        for c in range(ceil(n_va / batch_size)):
                            s = c * batch_size
                            e = min(n_va, s + batch_size)
                            xb = torch.from_numpy(X_val[s:e]).to(device, non_blocking=True)
                            yb = torch.from_numpy(y_val[s:e]).to(device, non_blocking=True)
                            vb_pred = model(xb)
                            val_loss_acc += float(criterion(vb_pred, yb).item()) * (e - s)
                            preds_all.append(vb_pred.detach().cpu())
                            true_all.append(yb.detach().cpu())
                            del xb, yb, vb_pred

                        val_loss = val_loss_acc / n_va
                        val_losses.append(val_loss)

                        preds_all = torch.vstack(preds_all)
                        true_all  = torch.vstack(true_all)
                        val_nse   = nse_torch(true_all, preds_all)

                    print(f"Época {epoch:03d} | Train MSE: {train_losses[-1]:.4f} | Val MSE: {val_loss:.4f} | Val NSE: {val_nse:.4f}")

                    # (opcional) LR manual por época
                    if epoch in [10, 20, 50, 75, 100]:
                        new_lr = {10: 5e-4, 20: 1e-4, 50: 5e-5, 75: 1e-5, 100: 1e-5}[epoch]
                        for g in optimizer.param_groups:
                            g['lr'] = new_lr
                        print(f"  -> LR ajustado para {new_lr:.0e} na época {epoch}")

                    if early.step(val_nse, model):
                        print(f"  -> Early stopping (NSE) na época {epoch} | best NSE ~ {early.best:.5f}")
                        early.restore(model)
                        break

                    torch.cuda.empty_cache()

                # ===== Avaliação final agregada (val) =====
                model.eval()
                with torch.no_grad():
                    n_va = X_val.shape[0]
                    preds_all, true_all = [], []
                    for c in range(ceil(n_va / batch_size)):
                        s = c * batch_size
                        e = min(n_va, s + batch_size)
                        xb = torch.from_numpy(X_val[s:e]).to(device, non_blocking=True)
                        yb = torch.from_numpy(y_val[s:e]).to(device, non_blocking=True)
                        vb_pred = model(xb)
                        preds_all.append(vb_pred.detach().cpu())
                        true_all.append(yb.detach().cpu())
                        del xb, yb, vb_pred
                    preds_all = torch.vstack(preds_all)
                    true_all  = torch.vstack(true_all)

                    val_mse = float(nn.MSELoss()(preds_all, true_all).item())
                    val_mae = float(torch.mean(torch.abs(preds_all - true_all)).item())
                    val_nse = nse_torch(true_all, preds_all)
                    val_kge = kge_torch(true_all, preds_all)

                results.append({
                    'scenario': 'Phys-II_varH',
                    'layers': hidden_layer,
                    'units': hidden_size,
                    'T': T,
                    'H_max': H_max,
                    'cols': '|'.join(used_cols),
                    'H_cols': '|'.join([f'{c}:{H_cols[c]}' for c in used_cols]),
                    'val_MSE': val_mse,
                    'val_MAE': val_mae,
                    'NSE': val_nse,
                    'KGE': val_kge
                })

                # ===== Salvar MODELO + SCALER =====
                model_filename  = f"physIIvarH_L{hidden_layer}_U{hidden_size}_T{T}.pkl"
                scaler_filename = f"physIIvarH_scaler_L{hidden_layer}_U{hidden_size}_T{T}.pkl"
                with open(model_filename,  'wb') as f: pickle.dump(model,  f)
                with open(scaler_filename, 'wb') as f: pickle.dump(scaler, f)

                # ===== Figura de perdas =====
                plt.figure(figsize=(8,4))
                plt.plot(train_losses, label='Train MSE')
                plt.plot(val_losses,   label='Val MSE')
                plt.title(f'Phys-II varH | L={hidden_layer} U={hidden_size} T={T} | H_max={H_max} | cols={len(used_cols)}')
                plt.xlabel('Epoch'); plt.ylabel('MSE')
                plt.ylim(0, 0.5)
                plt.legend(loc='best')
                fig_filename = model_filename.replace('.pkl', '.png')
                plt.savefig(fig_filename, dpi=300, bbox_inches='tight')
                plt.close()

                # CSV parcial
                pd.DataFrame(results).to_csv("grid_search_physII_varH_partial.csv", index=False)

                del model
                torch.cuda.empty_cache()

            except Exception as e:
                print(f"⚠️ Erro na config Phys-II varH L={hidden_layer}, U={hidden_size}, T={T}: {e}")
                torch.cuda.empty_cache()
                continue

# Resultados finais
df_results = pd.DataFrame(results)
df_results.to_csv('grid_search_physII_varH_results.csv', index=False)
print("Grid search Phys-II (varH) concluído. Resultados em 'grid_search_physII_varH_results.csv'.")



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

# Definição de dispositivo (GPU/CPU)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# ============================
# Carregar dados
# ============================
df = pd.read_csv(
    r"C:\Users\Aquasmart EHR\Desktop\Rodrigo\Tese\dados_interpolados.txt",
    sep='\t'
)
df = df.set_index("DATA")
df.index = pd.to_datetime(df.index)

# Definir variável alvo e preditores
target_col = 'FLU(m)46'
X_df = df.drop(columns=[target_col])
y = df[target_col].values

# ============================
# Configurações
# ============================
hidden_layers = [1,2, 3]
hidden_sizes = [64,128, 256]
T_list        = [6, 18, 36, 54, 72]   # 10-min steps (1h, 3h, 6h, 9h, 12h)

H = 36                        # janela temporal
D = X_df.shape[1]             # número de variáveis explicativas

train_size_global = int(len(df) * 0.8)
num_batches = 25
#batch_size = 5012     
batch_size = train_size_global // num_batches
        # tamanho do batch (ajustar se necessário)
# batch_size = min(batch_size, 1024)

# ============================
# Métricas customizadas
# ============================
@torch.no_grad()
def nse_torch(y_true: torch.Tensor, y_pred: torch.Tensor, eps: float = 1e-8) -> float:
    """Nash-Sutcliffe Efficiency (NSE)."""
    y_true = y_true.view(-1)
    y_pred = y_pred.view(-1)
    denom = torch.sum((y_true - torch.mean(y_true)) ** 2) + eps
    num = torch.sum((y_true - y_pred) ** 2)
    nse = 1.0 - (num / denom)
    return float(nse.detach().cpu())


@torch.no_grad()
def kge_torch(y_true: torch.Tensor, y_pred: torch.Tensor, eps: float = 1e-8) -> float:
    """Kling-Gupta Efficiency (KGE)."""
    y_true = y_true.view(-1)
    y_pred = y_pred.view(-1)
    mu_o = torch.mean(y_true)
    mu_s = torch.mean(y_pred)
    sd_o = torch.std(y_true) + eps
    sd_s = torch.std(y_pred) + eps

    r_num = torch.sum((y_true - mu_o) * (y_pred - mu_s))
    r_den = (torch.sqrt(torch.sum((y_true - mu_o) ** 2)) + eps) * \
            (torch.sqrt(torch.sum((y_pred - mu_s) ** 2)) + eps)
    r = r_num / r_den

    alpha = sd_s / sd_o
    beta = mu_s / (mu_o + eps)

    kge = 1.0 - torch.sqrt((r - 1.0) ** 2 + (alpha - 1.0) ** 2 + (beta - 1.0) ** 2)
    return float(kge.detach().cpu())


# ============================
# Early Stopping Flexível
# ============================
class EarlyStoppingFlexible:
    """
    Early stopping baseado em melhora relativa.
    mode='max' para NSE/KGE ou 'min' para erros normalizados.
    """
    def __init__(self, mode='max', rel_min_delta=0.005, patience=20,
                 smooth=3, restore_best=True):
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
        if len(self.hist) >= self.smooth:
            current = sum(self.hist[-self.smooth:]) / self.smooth
        else:
            current = self.hist[-1]

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
# Definição do Modelo LSTM
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
# Grid Search
# ============================
results = []

for hidden_layer in hidden_layers:
    for hidden_size in hidden_sizes:
        for T in T_list:
            try:
                print(f"Config: layers={hidden_layer}, units={hidden_size}, T={T}")

                # Ajuste do tamanho efetivo por T
                N = len(X_df) - T
                train_size = int(N * 0.8)

                # Split cronológico
                X_train_df = X_df.iloc[:train_size]
                X_val_df = X_df.iloc[train_size:N]
                y_full = y

                # Fit do scaler apenas no treino
                scaler = StandardScaler()
                scaler.fit(X_train_df.values)

                X_all_scaled = np.empty_like(X_df.values, dtype=np.float32)
                X_all_scaled[:train_size, :] = scaler.transform(
                    X_df.iloc[:train_size].values
                ).astype(np.float32)
                X_all_scaled[train_size:N, :] = scaler.transform(
                    X_df.iloc[train_size:N].values
                ).astype(np.float32)

                # Construção das janelas (treino e validação)
                X_train = np.zeros((train_size, H, D), dtype=np.float32)
                y_train = np.zeros((train_size, 1), dtype=np.float32)
                for t in range(H, train_size):
                    X_train[t, :, :] = X_all_scaled[t-H:t, :]
                    y_train[t, 0] = y_full[t + T]

                X_val = np.zeros((N - train_size, H, D), dtype=np.float32)
                y_val = np.zeros((N - train_size, 1), dtype=np.float32)
                for i in range(N - train_size):
                    t = i + train_size
                    X_val[i, :, :] = X_all_scaled[t-H:t, :]
                    y_val[i, 0] = y_full[t + T]

                # ===== Modelo / otimizador / loss =====
                model = LSTM(D, hidden_size, hidden_layer, 1).to(device)
                criterion = nn.MSELoss()
                optimizer = torch.optim.Adam(
                    model.parameters(), lr=1e-3, weight_decay=1e-4
                )

                # Early Stopping
                early = EarlyStoppingFlexible(
                    mode='max', rel_min_delta=0.005, patience=20,
                    smooth=3, restore_best=True
                )


                # Treinamento
                epochs = 100
                train_losses, val_losses, val_nse_hist = [], [], []

                for epoch in range(1, epochs + 1):
                    model.train()
                    n_tr = X_train.shape[0]
                    perm = np.random.permutation(n_tr)
                    epoch_loss = 0.0

                    for i in range(0, n_tr, batch_size):
                        idx = perm[i:i+batch_size]
                        xb_cpu = X_train[idx]
                        yb_cpu = y_train[idx]

                        xb = torch.from_numpy(xb_cpu).to(device, non_blocking=True)
                        yb = torch.from_numpy(yb_cpu).to(device, non_blocking=True)

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

                    # Validação
                    model.eval()
                    with torch.no_grad():
                        n_va = X_val.shape[0]
                        val_loss_acc = 0.0
                        val_preds_all, val_true_all = [], []

                        for c in range(ceil(n_va / batch_size)):
                            s = c * batch_size
                            e = min(n_va, s + batch_size)

                            xb = torch.from_numpy(X_val[s:e]).to(device, non_blocking=True)
                            yb = torch.from_numpy(y_val[s:e]).to(device, non_blocking=True)

                            vb_pred = model(xb)
                            vb_loss = criterion(vb_pred, yb).item()

                            val_loss_acc += vb_loss * (e - s)
                            val_preds_all.append(vb_pred.detach().cpu())
                            val_true_all.append(yb.detach().cpu())

                            del xb, yb, vb_pred

                        val_loss = val_loss_acc / n_va
                        val_losses.append(val_loss)

                        # NSE
                        val_preds_all = torch.vstack(val_preds_all)
                        val_true_all = torch.vstack(val_true_all)
                        val_nse_epoch = nse_torch(val_true_all, val_preds_all)
                        val_nse_hist.append(val_nse_epoch)

                    print(
                        f"Época {epoch:03d} | Train MSE: {train_losses[-1]:.4f} "
                        f"| Val MSE: {val_loss:.4f} | Val NSE: {val_nse_epoch:.4f}"
                    )

     
                    # Ajuste LR manual
                    if epoch in [10, 20, 50, 75, 100]:
                        new_lr = {10: 5e-4, 20: 1e-4, 50: 5e-5, 75: 1e-5, 100: 1e-5}[epoch]
                        for g in optimizer.param_groups:
                            g['lr'] = new_lr
                        print(f" -> LR ajustado para {new_lr:.0e} na época {epoch}")

                    # Early stopping
                    if early.step(val_nse_epoch, model):
                        print(f" -> Early stopping (NSE) na época {epoch} | best NSE ~ {early.best:.5f}")
                        early.restore(model)
                        break

                    torch.cuda.empty_cache()

                # Avaliação final
                model.eval()
                with torch.no_grad():
                    n_va = X_val.shape[0]
                    val_preds_all, val_true_all = [], []

                    for c in range(ceil(n_va / batch_size)):
                        s = c * batch_size
                        e = min(n_va, s + batch_size)
                        xb = torch.from_numpy(X_val[s:e]).to(device, non_blocking=True)
                        yb = torch.from_numpy(y_val[s:e]).to(device, non_blocking=True)
                        vb_pred = model(xb)
                        val_preds_all.append(vb_pred.detach().cpu())
                        val_true_all.append(yb.detach().cpu())
                        del xb, yb, vb_pred

                    val_preds_all = torch.vstack(val_preds_all)
                    val_true_all = torch.vstack(val_true_all)

                    val_mse = float(nn.MSELoss()(val_preds_all, val_true_all).item())
                    val_mae = float(torch.mean(torch.abs(val_preds_all - val_true_all)).item())
                    val_nse = nse_torch(val_true_all, val_preds_all)
                    val_kge = kge_torch(val_true_all, val_preds_all)

                results.append({
                    'layers': hidden_layer,
                    'units': hidden_size,
                    'T': T,
                    'val_MSE': val_mse,
                    'val_MAE': val_mae,
                    'NSE': val_nse,
                    'KGE': val_kge
                })

                # Salvar MODELO
                model_filename = f"lstm_L{hidden_layer}_U{hidden_size}_T{T}.pkl"
                with open(model_filename, 'wb') as f:
                    pickle.dump(model, f)

                # Salvar SCALER
                scaler_filename = f"scaler_L{hidden_layer}_U{hidden_size}_T{T}.pkl"
                with open(scaler_filename, 'wb') as f:
                    pickle.dump(scaler, f)

                # Salvar FIGURA
                plt.figure(figsize=(8, 4))
                plt.plot(train_losses, label='Train MSE')
                plt.plot(val_losses, label='Val MSE')
                plt.title(f'L={hidden_layer} U={hidden_size} T={T}')
                plt.xlabel('Epoch')
                plt.ylabel('MSE')
                plt.ylim(0, 0.5)
                plt.legend(loc='best')
                fig_filename = model_filename.replace('.pkl', '.png')
                plt.savefig(fig_filename, dpi=300, bbox_inches='tight')
                plt.close()

                # Salvar CSV parcial
                df_results = pd.DataFrame(results)
                df_results.to_csv("grid_search_results_partial.csv", index=False)

                # Liberar memória
                del model, optimizer
                torch.cuda.empty_cache()

            except RuntimeError as e:
                print(f"⚠️ Erro na config L={hidden_layer}, U={hidden_size}, T={T}: {e}")
                torch.cuda.empty_cache()
                continue

# ============================
# Resultados finais
# ============================
df_results = pd.DataFrame(results)
df_results.to_csv(r'grid_search_results.csv', index=False)
print("Grid search concluído. Resultados em 'grid_search_results.csv'.")




















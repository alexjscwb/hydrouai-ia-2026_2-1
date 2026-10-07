# -*- coding: utf-8 -*-
"""Experimento de preenchimento de falhas artificiais em séries diárias.

Dependências: numpy, pandas, scikit-learn, matplotlib; plotly é opcional.
"""

# %% BIBLIOTECAS E CONFIGURAÇÕES EDITÁVEIS
from pathlib import Path
import argparse
import hashlib
import html
import json
import platform
import re
import warnings

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # Salva figuras sem abrir janelas no Spyder.
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from sklearn.impute import KNNImputer
import sklearn

try:
    import plotly.graph_objects as go
except ImportError:
    go = None  # Todos os PNGs e as métricas funcionam sem Plotly.

BASE_DIR = Path(__file__).resolve().parent
INPUT_CSV = BASE_DIR / "series_selecionadas.csv"
OUTPUT_FOLDER = BASE_DIR / "resultados_preenchimento"
FINAL_CSV_NAME = "series_preenchidas.csv"  # Datas e TODAS as séries, independentemente dos relatórios.

RANDOM_SEED = 42
DRY_MONTHS = (4, 5, 6, 7, 8, 9)
WET_MONTHS = (10, 11, 12, 1, 2, 3)
GAP_LENGTHS_DAYS = (7, 15, 30)
REPETITIONS_PER_SEASON = 1
# EDITE ESTA LINHA: somente estas séries terão tabelas e figuras.
# Todas as séries da base continuam sendo avaliadas e preenchidas.
ESTACOES_RESULTADOS = ["Q_Afluente","Q_40032000"]  # Ex.: ["Q_Afluente", "Q_40032000", "P_1944059"]
MIN_SEPARATION_DAYS = 1  # Um dia intacto entre falhas, inclusive de séries diferentes.
KNN_NEIGHBORS = 5
WET_FLOW_GAPS_AT_PEAKS = True
WET_PEAK_QUANTILE = 0.95
PEAK_HALF_WINDOW_DAYS = 3  # Máximo local em uma vizinhança de ±3 dias.
ZOOM_CONTEXT_DAYS = 30
FIGURE_DPI = 160
CREATE_INTERACTIVE_HTML = True  # Exige Plotly; o JS é incluído para uso offline.

METHOD_LABELS = {"Mean": "Mean", "Linear": "Linear interpolation",
                 "KNN": f"KNN (k={KNN_NEIGHBORS})", "Historical": "Historical interval mean"}
METHOD_COLORS = {"Mean": "#278C55", "Linear": "#E88C20", "KNN": "#8450B3", "Historical": "#CC4256"}
METHOD_MARKERS = {"Mean": "o", "Linear": "D", "KNN": "x", "Historical": "s"}
METHOD_NAMES_PT = {"Mean": "Média global", "Linear": "Interpolação linear",
                   "KNN": "KNN", "Historical": "Média histórica do intervalo"}
SEASON_LABELS = {"Dry": "Dry season", "Wet": "Wet season"}
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                     "axes.spines.top": False, "axes.spines.right": False,
                     "axes.titleweight": "semibold", "figure.facecolor": "white"})


# %% LEITURA DA BASE
def find_input_file(requested):
    requested = Path(requested).expanduser()
    if requested.is_file():
        return requested.resolve()
    # Reconhece o nome original sem o sufixo do download.
    if requested == INPUT_CSV:
        for name in ("series_selecionadas.csv", "series_selecionadas (1).csv"):
            candidate = BASE_DIR / name
            if candidate.is_file():
                return candidate.resolve()
    raise FileNotFoundError(f"Base não encontrada: {requested}. Use --input ou edite INPUT_CSV.")


def read_dataset(path):
    dataset = pd.read_csv(path)
    candidates = ("data", "Data", "DataHora", "datetime", "Datetime", "DATE", "Date")
    time_column = next((c for c in candidates if c in dataset.columns), None)
    if time_column is None:
        raise ValueError("É necessária uma coluna de datas para distinguir seco e úmido.")
    dataset[time_column] = pd.to_datetime(dataset[time_column], errors="coerce")
    if dataset[time_column].isna().any():
        raise ValueError("Há datas inválidas. Corrija-as antes de criar falhas.")
    if dataset[time_column].duplicated().any():
        raise ValueError("Há datas duplicadas; não é seguro gerar blocos temporais.")
    dataset = dataset.sort_values(time_column).reset_index(drop=True)
    dates = dataset[time_column]
    if len(dates) < 3 or not dates.diff().iloc[1:].eq(pd.Timedelta(days=1)).all():
        raise ValueError("O experimento espera datas diárias consecutivas, como na base enviada.")
    numeric = dataset.select_dtypes(include="number").astype(float).copy()
    if numeric.empty:
        raise ValueError("Nenhuma série numérica encontrada.")
    if np.isinf(numeric.to_numpy()).any():
        raise ValueError("Há valores infinitos nas séries; corrija-os antes do experimento.")
    return dataset, dates, numeric, time_column


def seasonal_climatology(numeric, dates):
    """Diagnóstico da escolha dos meses; não é enviado aos preenchedores."""
    rain_columns = [c for c in numeric if c.startswith("P_")]
    if not rain_columns:
        return pd.DataFrame()
    # Média de cada estação dentro do mês, depois média entre estações.
    monthly = numeric[rain_columns].groupby(dates.dt.month).mean()
    result = pd.DataFrame({"Month": np.arange(1, 13)})
    result["Mean_daily_precipitation_mm"] = monthly.mean(axis=1).reindex(range(1, 13)).to_numpy()
    result["Season"] = np.where(result.Month.isin(DRY_MONTHS), "Dry", "Wet")
    return result


# %% CRIAÇÃO DE FALHAS SEM SOBREPOSIÇÃO
def wet_flow_peaks(series, wet_mask):
    """Define os picos no desenho experimental, antes de ocultar a referência.

    Um pico é máximo local na janela configurada, tem pelo menos um vizinho
    estritamente menor e está acima/igual ao quantil do período úmido.
    O valor do pico NÃO será fornecido a nenhum método de preenchimento.
    """
    if not 0 < WET_PEAK_QUANTILE < 1 or PEAK_HALF_WINDOW_DAYS < 1:
        raise ValueError("Quantil de pico deve estar entre 0 e 1; vizinhança deve ser positiva.")
    values = series.to_numpy(dtype=float)
    valid_wet = wet_mask & np.isfinite(values)
    threshold = float(np.quantile(values[valid_wet], WET_PEAK_QUANTILE))
    window = 2 * PEAK_HALF_WINDOW_DAYS + 1
    rolling_max = series.rolling(window, center=True, min_periods=1).max().to_numpy()
    rolling_min = series.rolling(window, center=True, min_periods=1).min().to_numpy()
    peak_mask = valid_wet & (values >= threshold) & (values == rolling_max) & (rolling_max > rolling_min)
    indices = np.flatnonzero(peak_mask)
    # Tenta primeiro os maiores picos disponíveis; sorteio desempata a posição do bloco.
    indices = indices[np.argsort(-values[indices], kind="stable")]
    return indices, threshold


def select_artificial_gaps(original, dates, columns, lengths, repetitions, seed):
    """Seleciona blocos só com referência observada e contornos intactos.

    A restrição é GLOBAL: nunca duas falhas artificiais na mesma data,
    mesmo em séries diferentes. Falhas úmidas das vazões são alocadas primeiro
    sobre picos; os demais blocos são sorteados, sem escolher sua amplitude.
    Não relaxa o critério de pico nem a ausência de sobreposição se faltar espaço.
    """
    if set(DRY_MONTHS) & set(WET_MONTHS) or set(DRY_MONTHS) | set(WET_MONTHS) != set(range(1, 13)):
        raise ValueError("DRY_MONTHS e WET_MONTHS devem particionar os 12 meses.")
    lengths = tuple(sorted(set(int(n) for n in lengths), reverse=True))
    if not lengths or min(lengths) < 1 or repetitions < 1:
        raise ValueError("Durações e repetições devem ser positivas.")
    if MIN_SEPARATION_DAYS < 1:
        raise ValueError("MIN_SEPARATION_DAYS deve ser pelo menos 1.")
    invalid_columns = set(columns) - set(original.columns)
    if invalid_columns:
        raise ValueError(f"Séries não encontradas: {sorted(invalid_columns)}")
    all_empty = [c for c in columns if original[c].isna().all()]
    if all_empty:
        warnings.warn(f"Sem referência observada; não serão mascaradas: {all_empty}")
    columns = [c for c in columns if c not in all_empty]
    if not columns:
        raise ValueError("Nenhuma série alvo possui valores observados.")

    rng = np.random.default_rng(seed)
    n = len(original)
    occupied = np.zeros(n, dtype=bool)
    mask = pd.DataFrame(False, index=original.index, columns=original.columns)
    season_masks = {
        "Dry": dates.dt.month.isin(DRY_MONTHS).to_numpy(),
        "Wet": dates.dt.month.isin(WET_MONTHS).to_numpy(),
    }
    peak_cache = {c: wet_flow_peaks(original[c], season_masks["Wet"])
                  for c in columns if c.startswith("Q_") and WET_FLOW_GAPS_AT_PEAKS}
    peak_jobs, other_jobs = [], []
    for length in lengths:
        priority = [(c, "Wet", repeat, length) for c in peak_cache
                    for repeat in range(1, repetitions + 1)]
        remainder = [(c, season, repeat, length) for c in columns for season in ("Dry", "Wet")
                     for repeat in range(1, repetitions + 1)
                     if not (season == "Wet" and c in peak_cache)]
        rng.shuffle(priority)
        rng.shuffle(remainder)
        priority.sort(key=lambda job: job[0] != "Q_Afluente")
        peak_jobs.extend(priority)
        other_jobs.extend(remainder)
    records = []
    for column, season, repeat, length in peak_jobs + other_jobs:
        observed = original[column].notna().to_numpy()
        eligible = observed & season_masks[season]
        cumulative = np.r_[0, np.cumsum(eligible.astype(int))]
        candidates = np.flatnonzero(cumulative[length:] - cumulative[:-length] == length)
        candidates = candidates[(candidates > 0) & (candidates + length < n)]
        acceptable = []
        for candidate in candidates:
            end_candidate = int(candidate + length - 1)
            left = max(0, int(candidate) - MIN_SEPARATION_DAYS)
            right = min(n, end_candidate + MIN_SEPARATION_DAYS + 1)
            if observed[candidate - 1] and observed[end_candidate + 1] and not occupied[left:right].any():
                acceptable.append(int(candidate))
        if not acceptable:
            raise ValueError(f"Sem espaço para {column}, {season}, {length} dias, repetição {repeat}.")
        peak_index, peak_value, peak_threshold = np.nan, np.nan, np.nan
        strategy = "random"
        if season == "Wet" and column in peak_cache:
            acceptable_array = np.asarray(acceptable)
            peaks, peak_threshold = peak_cache[column]
            start = None
            for peak in peaks:
                contains_peak = acceptable_array[(acceptable_array <= peak) & (acceptable_array + length > peak)]
                if not len(contains_peak):
                    continue
                distance = np.abs(contains_peak + (length - 1) / 2 - peak)
                starts = contains_peak[distance == distance.min()]
                start = int(rng.choice(starts))
                peak_index = int(peak)
                peak_value = float(original.at[peak_index, column])
                strategy = "wet_flow_peak"
                break
            if start is None:
                raise ValueError(
                    f"Sem pico >= quantil {WET_PEAK_QUANTILE} disponível para {column}, "
                    f"{length} dias, sem sobreposição. Reduza durações/repetições/séries. "
                    "Os critérios não serão relaxados automaticamente."
                )
        else:
            start = int(rng.choice(acceptable))
        end = start + length - 1
        occupied[start:end + 1] = True
        mask.loc[start:end, column] = True
        records.append({"Variable": column, "Season": season, "Duration_days": length,
                        "Repeat": repeat, "Start_index": start, "End_index": end,
                        "Start_date": dates.iloc[start], "End_date": dates.iloc[end],
                        "Selection_strategy": strategy, "Peak_index": peak_index,
                        "Peak_date": dates.iloc[int(peak_index)] if np.isfinite(peak_index) else pd.NaT,
                        "Peak_observed": peak_value, "Peak_threshold": peak_threshold})

    gaps = pd.DataFrame(records).sort_values(["Start_index", "Variable"]).reset_index(drop=True)
    gaps.insert(0, "Gap_ID", [f"G{i:04d}" for i in range(1, len(gaps) + 1)])
    corrupted = original.mask(mask)
    checks = verify_gap_design(original, corrupted, dates, mask, gaps)
    return corrupted, mask, gaps, checks


def verify_gap_design(original, corrupted, dates, mask, gaps):
    assert not (mask & original.isna()).to_numpy().any(), "Falha artificial sobre falha real."
    assert int(mask.sum(axis=1).max()) <= 1, "Falhas artificiais simultâneas."
    assert original.where(~mask).equals(corrupted), "Valor externo à máscara foi alterado."
    reconstructed = pd.DataFrame(False, index=mask.index, columns=mask.columns)
    last_end = -MIN_SEPARATION_DAYS - 1
    for gap in gaps.itertuples(index=False):
        start, end = gap.Start_index, gap.End_index
        assert start - last_end > MIN_SEPARATION_DAYS, "Falta separação entre falhas."
        last_end = end
        assert end - start + 1 == gap.Duration_days
        assert original.loc[start:end, gap.Variable].notna().all()
        assert original.at[start - 1, gap.Variable] == corrupted.at[start - 1, gap.Variable]
        assert original.at[end + 1, gap.Variable] == corrupted.at[end + 1, gap.Variable]
        months = DRY_MONTHS if gap.Season == "Dry" else WET_MONTHS
        assert dates.iloc[start:end + 1].dt.month.isin(months).all()
        if gap.Selection_strategy == "wet_flow_peak":
            assert gap.Variable.startswith("Q_") and gap.Season == "Wet"
            assert start <= gap.Peak_index <= end
            assert gap.Peak_observed >= gap.Peak_threshold
            valid_peaks, threshold = wet_flow_peaks(original[gap.Variable], dates.dt.month.isin(WET_MONTHS).to_numpy())
            assert int(gap.Peak_index) in valid_peaks and gap.Peak_threshold == threshold
        reconstructed.loc[start:end, gap.Variable] = True
    assert reconstructed.equals(mask), "Registro de falhas não corresponde à máscara."
    assert int(mask.to_numpy().sum()) == int(gaps.Duration_days.sum())
    return {"global_nonoverlap": True, "separation_days": MIN_SEPARATION_DAYS,
            "no_original_missing_value_used_as_reference": True,
            "observed_boundaries_preserved": True, "all_blocks_inside_one_season": True,
            "original_values_preserved_outside_mask": True, "mask_registry_consistent": True,
            "n_gaps": len(gaps), "n_hidden_values": int(mask.to_numpy().sum()),
            "n_wet_flow_peak_gaps": int(gaps.Selection_strategy.eq("wet_flow_peak").sum()),
            "all_targeted_peaks_inside_gaps_and_above_threshold": True,
            "max_simultaneous_artificial_gaps": int(mask.sum(axis=1).max())}


# %% MÉTODOS DO CÓDIGO ORIGINAL: RECEBEM APENAS A BASE COM FALHAS
def fill_with_existing_methods(corrupted):
    """Mantém exatamente as três regras originais, inclusive KNN sem escala.

    Não há argumento com a referência completa: valores ocultados não entram
    na média, na interpolação nem nos doadores/distâncias do KNN.
    Preenchimento retrospectivo: interpolação e KNN podem usar dados futuros.
    """
    mean_filled = corrupted.fillna(corrupted.mean())
    linear_filled = corrupted.interpolate(method="linear", limit_direction="both", axis=0)
    knn_filled = corrupted.copy()
    knn_columns = [c for c in corrupted if corrupted[c].notna().any()]
    if knn_columns:
        imputer = KNNImputer(n_neighbors=KNN_NEIGHBORS)
        values = imputer.fit_transform(corrupted[knn_columns])
        knn_filled[knn_columns] = pd.DataFrame(values, columns=knn_columns, index=corrupted.index)
    filled = {"Mean": mean_filled, "Linear": linear_filled, "KNN": knn_filled}
    observed = corrupted.notna().to_numpy()
    for method, frame in filled.items():
        assert np.array_equal(frame.to_numpy()[observed], corrupted.to_numpy()[observed]), \
            f"{method} alterou valores observados."
    return filled


# %% NOVO MÉTODO: MÉDIA HISTÓRICA DO MESMO INTERVALO DE CALENDÁRIO
def missing_blocks(mask):
    edges = np.diff(np.r_[False, np.asarray(mask, dtype=bool), False].astype(int))
    return list(zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1) - 1))


def historical_interval_donors(dates, start, end):
    """Mesmo intervalo mês/dia em outros anos; aceita travessia de dezembro.

    Mês/dia evita o deslocamento de datas depois de fevereiro em anos bissextos.
    Em intervalos que incluem 29/2, esse dia só existe nos anos bissextos.
    """
    first, last = dates.iloc[start], dates.iloc[end]
    if last.year - first.year > 1 or (last.year != first.year and (last - first).days >= 366):
        raise ValueError("A média de intervalo histórico exige blocos de até um ciclo anual.")
    calendar_key = (dates.dt.month * 100 + dates.dt.day).to_numpy()
    start_key, end_key = first.month * 100 + first.day, last.month * 100 + last.day
    crosses_year = first.year != last.year
    if crosses_year:
        same_interval = (calendar_key >= start_key) | (calendar_key <= end_key)
        anchor_years = dates.dt.year.to_numpy() - (calendar_key <= end_key).astype(int)
    else:
        same_interval = (calendar_key >= start_key) & (calendar_key <= end_key)
        anchor_years = dates.dt.year.to_numpy()
    donor_dates = same_interval & (anchor_years != first.year)
    return donor_dates, anchor_years


def fill_with_historical_interval_mean(corrupted, dates):
    """Uma média escalar por falha, usando apenas observações disponíveis.

    Exemplo: falha 15/8–30/8 -> média de TODOS os valores disponíveis de 15/8
    a 30/8 em todos os outros anos; repete essa média em todos os dias da falha.
    Anos com intervalos parcialmente observados contribuem com seus dias válidos.
    Não usa a base original, valores preenchidos, médias dos outros métodos,
    nem metadados com os valores dos picos. Também trata falhas reais.
    """
    result = corrupted.copy()
    summaries, by_year = [], []
    for column in corrupted:
        values = corrupted[column].to_numpy(dtype=float)
        if not np.isfinite(values).any():
            continue  # Sem observação de referência; mantém a coluna inteiramente ausente.
        for start, end in missing_blocks(~np.isfinite(values)):
            donor_dates, anchor_years = historical_interval_donors(dates, start, end)
            donor_mask = donor_dates & np.isfinite(values)
            assert not donor_mask[start:end + 1].any(), "Valores da própria falha entraram na média."
            assert np.isfinite(values[donor_mask]).all()
            estimate = float(values[donor_mask].mean()) if donor_mask.any() else np.nan
            donor_years = np.unique(anchor_years[donor_mask])
            result.loc[start:end, column] = estimate
            summaries.append({"Variable": column, "Start_index": int(start), "End_index": int(end),
                              "Start_date": dates.iloc[start], "End_date": dates.iloc[end],
                              "Start_month_day": dates.iloc[start].strftime("%m-%d"),
                              "End_month_day": dates.iloc[end].strftime("%m-%d"),
                              "Excluded_interval_anchor_year": int(dates.iloc[start].year),
                              "Historical_mean": estimate, "N_observations": int(donor_mask.sum()),
                              "N_years": len(donor_years), "Source_years": ";".join(map(str, donor_years)),
                              "Status": "ok" if donor_mask.any() else "no_historical_observations"})
            for year in np.unique(anchor_years[donor_dates]):
                interval = donor_dates & (anchor_years == year)
                available = donor_mask & (anchor_years == year)
                by_year.append({"Variable": column, "Start_index": int(start), "End_index": int(end),
                                "Donor_interval_anchor_year": int(year),
                                "N_dates_in_dataset": int(interval.sum()),
                                "N_observed_days": int(available.sum()),
                                "Mean_of_available_days": float(values[available].mean()) if available.any() else np.nan})
    known = corrupted.notna().to_numpy()
    assert np.array_equal(result.to_numpy()[known], corrupted.to_numpy()[known])
    return result, pd.DataFrame(summaries), pd.DataFrame(by_year)


# %% VALIDAÇÃO EXCLUSIVAMENTE NOS VALORES ARTIFICIALMENTE REMOVIDOS
def calculate_metrics(observed, estimated):
    observed = np.asarray(observed, dtype=float)
    estimated = np.asarray(estimated, dtype=float)
    finite = np.isfinite(observed) & np.isfinite(estimated)
    n_expected, n_valid = len(observed), int(finite.sum())
    if n_valid == 0:
        return {"N_expected": n_expected, "N_valid": 0, "NSE": np.nan,
                "MAE": np.nan, "RMSE": np.nan, "Status": "no_valid_pairs"}
    truth, prediction = observed[finite], estimated[finite]
    errors = prediction - truth
    denominator = float(np.sum((truth - truth.mean()) ** 2))
    nse = 1 - float(np.sum(errors ** 2)) / denominator if denominator > 0 else np.nan
    status = "ok" if denominator > 0 else "constant_reference_NSE_undefined"
    if n_valid != n_expected:
        status += ";partial_predictions"
    return {"N_expected": n_expected, "N_valid": n_valid, "NSE": nse,
            "MAE": float(np.mean(np.abs(errors))),
            "RMSE": float(np.sqrt(np.mean(errors ** 2))), "Status": status}


def evaluate_gaps(original, dates, filled, gaps):
    gap_metrics, hidden_rows = [], []
    for gap in gaps.itertuples(index=False):
        indices = np.arange(gap.Start_index, gap.End_index + 1)
        truth = original.loc[indices, gap.Variable].to_numpy()
        block = pd.DataFrame({"Gap_ID": gap.Gap_ID, "Variable": gap.Variable,
                              "Season": gap.Season, "Duration_days": gap.Duration_days,
                              "Selection_strategy": gap.Selection_strategy,
                              "Repeat": gap.Repeat, "Date": dates.iloc[indices].to_numpy(),
                              "Observed": truth})
        for method, frame in filled.items():
            estimates = frame.loc[indices, gap.Variable].to_numpy()
            # Nesta base todos os métodos devem preencher TODOS os valores de validação.
            if not np.isfinite(estimates).all():
                raise ValueError(f"{method} não estimou toda a falha {gap.Gap_ID}.")
            block[method] = estimates
            row = {"Gap_ID": gap.Gap_ID, "Variable": gap.Variable, "Season": gap.Season,
                   "Duration_days": gap.Duration_days, "Repeat": gap.Repeat,
                   "Start_date": gap.Start_date, "End_date": gap.End_date, "Method": method}
            row.update({"Selection_strategy": gap.Selection_strategy, "Peak_date": gap.Peak_date,
                        "Peak_observed": gap.Peak_observed})
            row.update(calculate_metrics(truth, estimates))
            gap_metrics.append(row)
        hidden_rows.append(block)
    hidden = pd.concat(hidden_rows, ignore_index=True)

    def aggregate(keys):
        records = []
        for key, group in hidden.groupby(keys, sort=True):
            key = key if isinstance(key, tuple) else (key,)
            metadata = dict(zip(keys, key))
            for method in filled:
                row = {**metadata, "Method": method, "N_gaps": int(group.Gap_ID.nunique())}
                # Recalcula NSE sobre todos os pontos do grupo; NÃO tira média de NSEs.
                row.update(calculate_metrics(group.Observed, group[method]))
                records.append(row)
        return pd.DataFrame(records)

    tables = {"metricas_por_falha": pd.DataFrame(gap_metrics),
              "metricas_por_periodo": aggregate(["Variable", "Season"]),
              "metricas_por_duracao": aggregate(["Variable", "Season", "Duration_days"]),
              "metricas_por_variavel": aggregate(["Variable"])}
    tables["metricas_picos_vazao"] = tables["metricas_por_falha"].loc[
        tables["metricas_por_falha"].Selection_strategy.eq("wet_flow_peak")].reset_index(drop=True)
    return hidden, tables


# %% MELHOR MÉTODO POR SÉRIE E BASE FINAL COM VAZIOS REAIS PREENCHIDOS
def resolve_report_columns(requested, numeric_columns):
    """A escolha muda a exibição; nunca muda a validação ou o preenchimento."""
    columns = list(numeric_columns) if requested is None else list(dict.fromkeys(requested))
    unknown = [c for c in columns if c not in numeric_columns]
    if unknown:
        raise ValueError(f"Estações não encontradas: {unknown}. Disponíveis: {list(numeric_columns)}")
    if not columns:
        raise ValueError("Informe ao menos uma série em ESTACOES_RESULTADOS; None exibe todas.")
    return columns


def select_best_methods(variable_metrics, all_columns):
    """Maior NSE agregado nos mesmos valores ocultados de cada série.

    NSE negativo continua válido. Não faz média dos NSEs por período/falha.
    Desempate: menor RMSE, menor MAE e ordem fixa dos métodos.
    Se TODOS os NSEs forem indefinidos, usa RMSE e declara a exceção.
    """
    order = {method: number for number, method in enumerate(METHOD_LABELS)}
    records = []
    for column in all_columns:
        group = variable_metrics.loc[variable_metrics.Variable.eq(column)].copy()
        if not group.empty:
            complete = (group.N_valid == group.N_expected) & group.N_valid.gt(0)
            group = group.loc[complete & np.isfinite(group.MAE) & np.isfinite(group.RMSE)].copy()
        if group.empty:
            records.append({"Variable": column, "Method": None, "NSE": np.nan,
                            "MAE": np.nan, "RMSE": np.nan, "N_valid": 0,
                            "Selection_criterion": "no_validation_reference"})
            continue
        group["_order"] = group.Method.map(order)
        valid_nse = group[np.isfinite(group.NSE)]
        if not valid_nse.empty:
            ranked = valid_nse.sort_values(["NSE", "RMSE", "MAE", "_order"],
                                          ascending=[False, True, True, True])
            criterion = "NSE"
        else:
            ranked = group.sort_values(["RMSE", "MAE", "_order"])
            criterion = "RMSE_fallback_NSE_undefined"
        best = ranked.iloc[0]
        records.append({"Variable": column, "Method": best.Method, "NSE": best.NSE,
                        "MAE": best.MAE, "RMSE": best.RMSE, "N_valid": int(best.N_valid),
                        "Selection_criterion": criterion})
    return pd.DataFrame(records)


def build_consolidated_dataset(dataset, original, dates, best_methods):
    """Depois da seleção, reaplica os métodos à base ORIGINAL.

    Mantém todos os valores medidos, inclusive os que foram ocultados somente
    na validação. Preenche apenas ausências reais com o método escolhido.
    Não exporta quatro bases completas: exportará uma única base consolidada.
    """
    refitted = fill_with_existing_methods(original)
    refitted["Historical"], _, _ = fill_with_historical_interval_mean(original, dates)
    combined = original.copy()
    for row in best_methods.itertuples(index=False):
        if row.Method is None:
            continue  # Coluna sem referência não admite escolher um método pelo NSE.
        missing = original[row.Variable].isna()
        predictions = refitted[row.Method].loc[missing, row.Variable]
        if predictions.isna().any():
            raise ValueError(f"{row.Method}, escolhido para {row.Variable}, não preencheu todos os vazios reais.")
        combined.loc[missing, row.Variable] = predictions
    known = original.notna().to_numpy()
    assert np.array_equal(combined.to_numpy()[known], original.to_numpy()[known]), \
        "A base final modificou valores observados."
    result = dataset.copy()
    result[original.columns] = combined
    return result


def simple_result_table(variable_metrics, best_methods, columns):
    selected = variable_metrics.loc[variable_metrics.Variable.isin(columns)].copy()
    order = {column: number for number, column in enumerate(columns)}
    method_order = {method: number for number, method in enumerate(METHOD_LABELS)}
    selected["_series_order"] = selected.Variable.map(order)
    selected["_method_order"] = selected.Method.map(method_order)
    selected = selected.sort_values(["_series_order", "_method_order"]).reset_index(drop=True)
    best_map = best_methods.set_index("Variable").Method.to_dict()
    return pd.DataFrame({"Série": selected.Variable, "Método": selected.Method.map(METHOD_NAMES_PT),
                         "NSE": selected.NSE, "MAE": selected.MAE, "RMSE": selected.RMSE,
                         "Melhor": ["Sim" if best_map.get(c) == m else "" for c, m in
                                    zip(selected.Variable, selected.Method)]}).reset_index(drop=True)


def save_simple_result_table(table, output):
    """Mesma tabela enxuta no console, CSV, HTML offline e PNG."""
    save_csv(table, output / "tabela_resultados.csv")
    printable = table.copy()
    for column in ("NSE", "MAE", "RMSE"):
        printable[column] = printable[column].map(lambda value: f"{value:.3f}" if np.isfinite(value) else "Indefinido")
    print("\nCOMPARAÇÃO DOS MÉTODOS — SOMENTE AS SÉRIES ESCOLHIDAS", flush=True)
    print(printable.to_string(index=False), flush=True)
    headers = "".join(f"<th>{html.escape(str(c))}</th>" for c in printable.columns)
    rows = []
    for values in printable.itertuples(index=False, name=None):
        css = ' class="best"' if values[-1] == "Sim" else ""
        cells = "".join(f"<td>{html.escape(str(value))}</td>" for value in values)
        rows.append(f"<tr{css}>{cells}</tr>")
    page = """<!doctype html><html lang="pt-BR"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Resultados do preenchimento</title><style>
body{font:16px Arial,sans-serif;color:#173955;background:#f5f7fa;margin:0;padding:32px}
main{max-width:1100px;margin:auto;background:white;padding:28px;border-radius:12px}
h1{font-size:24px;margin:0 0 12px}p{line-height:1.5;color:#546574}
.scroll{overflow-x:auto}table{border-collapse:collapse;width:100%;margin:20px 0}
th{background:#173955;color:white;text-align:left}th,td{padding:12px;border-bottom:1px solid #dee6ed}
td:nth-child(3),td:nth-child(4),td:nth-child(5){text-align:right;font-variant-numeric:tabular-nums}
.best{background:#e5f3ed;font-weight:bold}small{color:#546574}</style>
<main><h1>Comparação dos métodos de preenchimento</h1>
<p>NSE, MAE e RMSE calculados somente nas falhas artificiais, reunindo seco e úmido.
A linha destacada é o método escolhido para a série.</p><div class="scroll"><table><thead><tr>"""
    page += headers + "</tr></thead><tbody>" + "".join(rows) + "</tbody></table></div>"
    page += "<small>NSE maior é melhor. MAE/RMSE menores são melhores e têm a unidade da série "
    page += "(mm para precipitação; m³/s para vazão). Quando todos os NSEs são indefinidos, "
    page += "a escolha usa o menor RMSE, conforme o critério registrado.</small></main></html>"
    (output / "tabela_resultados.html").write_text(page, encoding="utf-8")
    figures = output / "figuras"
    figures.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(11.8, max(2.6, 1.35 + 0.34 * len(table))))
    ax.axis("off")
    rendered = ax.table(cellText=printable.values, colLabels=printable.columns,
                        colWidths=[0.19, 0.31, 0.10, 0.14, 0.14, 0.10],
                        cellLoc="center", loc="center")
    rendered.auto_set_font_size(False)
    rendered.set_fontsize(9.5)
    rendered.scale(1, 1.7)
    for (row, column), cell in rendered.get_celld().items():
        cell.set_edgecolor("#DCE5EB")
        if row == 0:
            cell.set_facecolor("#173955")
            cell.set_text_props(color="white", weight="bold")
        elif printable.iloc[row - 1]["Melhor"] == "Sim":
            cell.set_facecolor("#E5F3ED")
            cell.set_text_props(weight="bold", color="#173955")
        else:
            cell.set_facecolor("white" if row % 2 else "#F5F7FA")
    fig.suptitle("Resultados nas falhas artificiais | seco + úmido", fontsize=13, y=0.97)
    fig.text(0.05, 0.03, "NSE: maior é melhor. MAE/RMSE: unidade da série. Melhor: método escolhido.",
             fontsize=8, color="#546574")
    fig.subplots_adjust(left=0.025, right=0.995, top=0.85, bottom=0.14)
    fig.savefig(figures / "tabela_resultados.png", dpi=FIGURE_DPI)
    plt.close(fig)


# %% FIGURAS
def safe_name(text):
    return re.sub(r'[\\/:*?"<>|]', "_", str(text))


def variable_unit(column):
    if column.startswith("P_"):
        return "Precipitation (mm)"
    if column.startswith("Q_"):
        return "Discharge (m³/s)"
    return "Value"


def format_date_axis(ax):
    locator = mdates.AutoDateLocator(minticks=3, maxticks=6)
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))
    ax.grid(True, linewidth=0.5, alpha=0.22)


def plot_one_gap(ax, gap, original, corrupted, dates, filled, context=ZOOM_CONTEXT_DAYS):
    start, end, column = gap.Start_index, gap.End_index, gap.Variable
    left, right = max(0, start - context), min(len(dates), end + context + 1)
    ax.axvspan(dates.iloc[start] - pd.Timedelta(hours=12),
               dates.iloc[end] + pd.Timedelta(hours=12), color="#CCD8E5", alpha=0.45)
    ax.plot(dates.iloc[left:right], original[column].iloc[left:right], color="#697582",
            linewidth=1.5, linestyle="--", label="Original reference", zorder=3)
    ax.plot(dates.iloc[left:right], corrupted[column].iloc[left:right], color="#173955",
            linewidth=1.25, label="Available observations", zorder=4)
    ax.scatter(dates.iloc[start:end + 1], original[column].iloc[start:end + 1],
               color="#697582", s=11, zorder=5)
    for method, frame in filled.items():
        ax.plot(dates.iloc[start - 1:end + 2], frame[column].iloc[start - 1:end + 2],
                color=METHOD_COLORS[method], linewidth=1.6, marker=METHOD_MARKERS[method],
                markersize=3.5, label=METHOD_LABELS[method], zorder=6, alpha=0.9)
    peak_index = getattr(gap, "Peak_index", np.nan)
    if np.isfinite(peak_index):
        peak_index = int(peak_index)
        ax.scatter([dates.iloc[peak_index]], [original.at[peak_index, column]],
                   marker="*", s=80, color="#173955", zorder=8)
        ax.annotate("Peak", (dates.iloc[peak_index], original.at[peak_index, column]),
                    xytext=(6, 7), textcoords="offset points", fontsize=8, color="#173955")
        ax.margins(y=0.12)  # Espaço para o marcador/texto do pico abaixo do título.
    ax.set_ylabel(variable_unit(column))
    format_date_axis(ax)


def save_zoom_figures(folder, gaps, original, corrupted, dates, filled, metrics):
    for number, gap in enumerate(gaps.itertuples(index=False), 1):
        variable_folder = folder / safe_name(gap.Variable)
        variable_folder.mkdir(parents=True, exist_ok=True)
        fig, ax = plt.subplots(figsize=(11.8, 5.7))
        plot_one_gap(ax, gap, original, corrupted, dates, filled)
        ax.set_title(f"{gap.Variable} | {SEASON_LABELS[gap.Season]} | {gap.Duration_days} days\n"
                     f"{gap.Start_date:%Y-%m-%d} to {gap.End_date:%Y-%m-%d} | {gap.Gap_ID}")
        rows = metrics.loc[metrics.Gap_ID == gap.Gap_ID]
        texts = []
        for row in rows.itertuples(index=False):
            nse = f"{row.NSE:.3f}" if np.isfinite(row.NSE) else "undefined (constant reference)"
            texts.append(f"{METHOD_LABELS[row.Method]}: NSE={nse}, MAE={row.MAE:.3f}, RMSE={row.RMSE:.3f}")
        ax.legend(loc="upper left", bbox_to_anchor=(0, -0.14), ncols=3, fontsize=8.5)
        fig.text(0.09, 0.025, "\n".join(texts), fontsize=8.5, color="#344454", va="bottom")
        fig.subplots_adjust(left=0.09, right=0.98, top=0.85, bottom=0.34)
        name = f"{gap.Gap_ID}_{gap.Season}_{gap.Duration_days:02d}d_{gap.Start_date:%Y%m%d}.png"
        fig.savefig(variable_folder / name, dpi=FIGURE_DPI)
        plt.close(fig)
        if number % 40 == 0 or number == len(gaps):
            print(f"  Figuras por falha: {number}/{len(gaps)}", flush=True)


def method_segment(corrupted, frame, column, variable_gaps):
    series = pd.Series(np.nan, index=corrupted.index, dtype=float)
    for gap in variable_gaps.itertuples(index=False):
        left, right = gap.Start_index - 1, gap.End_index + 1
        series.loc[left:right] = frame.loc[left:right, column]
    return series


def save_full_series_figures(folder, gaps, original, corrupted, dates, filled):
    folder.mkdir(parents=True, exist_ok=True)
    for column, group in gaps.groupby("Variable", sort=False):
        fig, ax = plt.subplots(figsize=(14, 5))
        ax.plot(dates, original[column], color="#8A98A5", linewidth=0.6, alpha=0.65,
                label="Original reference")
        ax.plot(dates, corrupted[column], color="#173955", linewidth=0.65,
                label="Available observations")
        for method, frame in filled.items():
            ax.plot(dates, method_segment(corrupted, frame, column, group),
                    color=METHOD_COLORS[method], linewidth=1.6, label=METHOD_LABELS[method])
        for gap in group.itertuples(index=False):
            color = "#C5D5E1" if gap.Season == "Dry" else "#B9D9E9"
            ax.axvspan(gap.Start_date, gap.End_date, color=color, alpha=0.25)
        fig.suptitle(f"{column} | {len(group)} artificial gaps | dry and wet seasons", fontsize=13, y=0.99)
        ax.set_ylabel(variable_unit(column))
        handles, labels = ax.get_legend_handles_labels()
        fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.92), ncols=3, fontsize=9)
        fig.subplots_adjust(left=0.07, right=0.99, top=0.72, bottom=0.13)
        format_date_axis(ax)
        fig.savefig(folder / f"Full_{safe_name(column)}.png", dpi=FIGURE_DPI)
        plt.close(fig)


def save_example_figure(path, gaps, original, corrupted, dates, filled):
    column = "Q_Afluente" if "Q_Afluente" in gaps.Variable.values else gaps.Variable.iloc[0]
    selected = gaps[(gaps.Variable == column) & (gaps.Repeat == 1)]
    lengths = sorted(selected.Duration_days.unique())
    fig, axes = plt.subplots(len(lengths), 2, figsize=(15.2, 3.6 * len(lengths)), squeeze=False)
    for i, length in enumerate(lengths):
        for j, season in enumerate(("Dry", "Wet")):
            gap = selected[(selected.Duration_days == length) & (selected.Season == season)].iloc[0]
            ax = axes[i, j]
            plot_one_gap(ax, gap, original, corrupted, dates, filled, context=15)
            ax.set_title(f"{SEASON_LABELS[season]} | {length} days | {gap.Start_date:%Y-%m-%d}", fontsize=11)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.959), ncols=3, fontsize=10)
    fig.suptitle(f"{column} | artificial-gap reconstruction | repetition 1", fontsize=16, y=0.994)
    fig.subplots_adjust(left=0.07, right=0.99, top=0.86, bottom=0.035, hspace=0.42, wspace=0.18)
    fig.savefig(path, dpi=FIGURE_DPI)
    plt.close(fig)


def save_long_gap_example(path, gaps, original, corrupted, dates, filled):
    column = "Q_Afluente" if "Q_Afluente" in gaps.Variable.values else gaps.Variable.iloc[0]
    selected = gaps[(gaps.Variable == column) & (gaps.Repeat == 1)]
    length = int(selected.Duration_days.max())
    fig, axes = plt.subplots(1, 2, figsize=(14.5, 5.2))
    for ax, season in zip(axes, ("Dry", "Wet")):
        gap = selected[(selected.Duration_days == length) & (selected.Season == season)].iloc[0]
        plot_one_gap(ax, gap, original, corrupted, dates, filled, context=15)
        ax.set_title(f"{SEASON_LABELS[season]} | {length} days | {gap.Start_date:%Y-%m-%d}", fontsize=11)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.91), ncols=3, fontsize=9)
    fig.suptitle(f"{column} | reconstruction of {length}-day gaps", fontsize=14, y=0.99)
    fig.subplots_adjust(left=0.065, right=0.99, top=0.68, bottom=0.14, wspace=0.22)
    fig.savefig(path, dpi=FIGURE_DPI)
    plt.close(fig)


def save_metric_figure(path, period_metrics):
    column = "Q_Afluente" if "Q_Afluente" in period_metrics.Variable.values else period_metrics.Variable.iloc[0]
    table = period_metrics[period_metrics.Variable == column]
    fig, axes = plt.subplots(1, 3, figsize=(12.3, 4.7))
    x = np.arange(2)
    width = 0.8 / len(METHOD_LABELS)
    for ax, metric in zip(axes, ("NSE", "MAE", "RMSE")):
        for j, method in enumerate(METHOD_LABELS):
            values = table[table.Method == method].set_index("Season").reindex(["Dry", "Wet"])[metric]
            bars = ax.bar(x + (j - (len(METHOD_LABELS) - 1) / 2) * width, values, width=width * 0.94,
                          color=METHOD_COLORS[method], label=METHOD_LABELS[method])
            ax.bar_label(bars, fmt="%.2f", fontsize=7.5, padding=3 + 11 * (j % 2))
        ax.set_xticks(x, ["Dry season", "Wet season"])
        ax.set_title(metric)
        ax.set_ylabel("Dimensionless" if metric == "NSE" else variable_unit(column))
        ax.grid(axis="y", linewidth=0.5, alpha=0.2)
        ax.set_axisbelow(True)
        ax.margins(y=0.15)
        if metric == "NSE":
            ax.axhline(0, color="#6D7885", linewidth=0.8)
    handles, labels = axes[-1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.93), ncols=4, frameon=False, fontsize=9)
    fig.suptitle(f"{column} | metrics on artificially withheld observations", fontsize=13, y=0.995)
    fig.subplots_adjust(left=0.075, right=0.99, top=0.79, bottom=0.14, wspace=0.34)
    fig.savefig(path, dpi=FIGURE_DPI)
    plt.close(fig)


def save_climatology_figure(path, table):
    if table.empty:
        return
    fig, ax = plt.subplots(figsize=(10, 3.5), constrained_layout=True)
    colors = ["#A1B4C1" if s == "Dry" else "#277CB1" for s in table.Season]
    ax.bar(table.Month, table.Mean_daily_precipitation_mm, color=colors)
    ax.set_xticks(range(1, 13), ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"])
    ax.set_ylabel("Mean daily precipitation (mm)")
    ax.set_title("Monthly climatology | dry: Apr–Sep | wet: Oct–Mar")
    ax.grid(axis="y", linewidth=0.5, alpha=0.2)
    ax.set_axisbelow(True)
    fig.savefig(path, dpi=FIGURE_DPI)
    plt.close(fig)


def save_interactive_figures(folder, gaps, original, corrupted, dates, filled):
    if go is None:
        print("Plotly não instalado: PNGs gerados normalmente; HTMLs opcionais não gerados.", flush=True)
        return 0
    folder.mkdir(parents=True, exist_ok=True)
    for column, group in gaps.groupby("Variable", sort=False):
        fig = go.Figure()
        fig.add_trace(go.Scattergl(x=dates, y=original[column], name="Original reference",
                                  line=dict(color="#8493A0", width=1, dash="dash")))
        fig.add_trace(go.Scattergl(x=dates, y=corrupted[column], name="Available observations",
                                  line=dict(color="#173955", width=1), connectgaps=False))
        for method, frame in filled.items():
            fig.add_trace(go.Scattergl(x=dates, y=method_segment(corrupted, frame, column, group),
                                      name=METHOD_LABELS[method], mode="lines+markers",
                                      line=dict(color=METHOD_COLORS[method], width=2),
                                      marker=dict(size=4), connectgaps=False))
        fig.update_layout(title=f"{column} | artificial-gap validation", template="plotly_white",
                          yaxis_title=variable_unit(column), xaxis_title="Date", hovermode="x unified",
                          legend=dict(orientation="h"), height=650)
        fig.update_xaxes(rangeslider_visible=True)
        fig.write_html(folder / f"Filling_{safe_name(column)}.html", include_plotlyjs=True,
                       config={"displaylogo": False, "scrollZoom": True, "responsive": True})
    return int(gaps.Variable.nunique())


# %% SALVAMENTO E EXECUÇÃO
def save_csv(table, path):
    table.to_csv(path, index=False, encoding="utf-8-sig", date_format="%Y-%m-%d")


def clear_previous_generated_reports(output):
    """Evita conservar relatórios de séries que saíram da lista em uma nova execução.

    Remove somente nomes/padrões produzidos por este script. Não apaga a pasta
    inteira, arquivos de entrada ou outros arquivos que o usuário tenha colocado.
    Mantém as subpastas vazias: no Windows elas podem estar abertas/bloqueadas.
    Falhas de limpeza geram aviso; não interrompem o preenchimento.
    """
    blocked = []

    def remove_generated_file(path):
        try:
            path.unlink(missing_ok=True)
        except OSError as error:
            blocked.append(f"{path}: {error}")

    def find_generated_files(parent, pattern):
        try:
            return list(parent.glob(pattern))
        except OSError as error:
            blocked.append(f"{parent}: {error}")
            return []

    obsolete_files = ["dataset_preenchido_media.csv", "dataset_preenchido_interpolacao.csv",
                      "dataset_preenchido_knn.csv", "dataset_preenchido_media_historica.csv",
                      "climatologia_mensal.csv", "dataset_preenchido_melhor_NSE.csv"]
    for name in obsolete_files:
        remove_generated_file(output / name)
    figures = output / "figuras"
    for name in ["exemplo_preenchimento.png", "exemplo_30_dias.png", "metricas_Q_Afluente.png",
                 "climatologia_mensal.png", "tabela_resultados.png"]:
        remove_generated_file(figures / name)
    for pattern in ["Comparacao_*.png", "Longa_*.png", "Metricas_*.png", "series_completas/Full_*.png"]:
        for path in find_generated_files(figures, pattern):
            remove_generated_file(path)
    for path in find_generated_files(figures / "falhas_individuais", "*/*.png"):
        if re.fullmatch(r"G\d+_(Dry|Wet)_\d+d_\d{8}\.png", path.name):
            remove_generated_file(path)
    for path in find_generated_files(output / "html_interativos", "Filling_*.html"):
        remove_generated_file(path)
    if blocked:
        warnings.warn(
            "Não foi possível limpar alguns relatórios anteriores. "
            "O preenchimento continuará; os arquivos bloqueados podem permanecer "
            "na pasta de resultados. Detalhes:\n" + "\n".join(blocked),
            RuntimeWarning,
        )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=INPUT_CSV)
    parser.add_argument("--output", type=Path, default=OUTPUT_FOLDER)
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    parser.add_argument("--repetitions", type=int, default=REPETITIONS_PER_SEASON)
    parser.add_argument("--durations", type=int, nargs="+", default=GAP_LENGTHS_DAYS)
    parser.add_argument("--stations", "--columns", dest="stations", nargs="+", default=ESTACOES_RESULTADOS,
                        help="Séries que terão tabelas/figuras. Todas continuam sendo preenchidas.")
    parser.add_argument("--skip-gap-plots", action="store_true")
    parser.add_argument("--skip-interactive", action="store_true")
    args = parser.parse_args(argv)
    input_path = find_input_file(args.input)
    output = args.output.expanduser().resolve()
    dataset, dates, original, time_column = read_dataset(input_path)
    report_columns = resolve_report_columns(args.stations, list(original.columns))
    all_columns = list(original.columns)
    print(f"Base: {len(dataset)} datas, {len(original.columns)} séries numéricas.", flush=True)
    print(f"Relatórios somente para: {', '.join(report_columns)}", flush=True)
    corrupted, mask, gaps, checks = select_artificial_gaps(
        original, dates, all_columns, args.durations, args.repetitions, args.seed)
    print(f"Validação interna: {len(gaps)} blocos, {int(mask.to_numpy().sum())} valores ocultados; sem sobreposição.", flush=True)
    print("Avaliando quatro métodos para escolher o melhor de cada série pelo NSE...", flush=True)
    filled = fill_with_existing_methods(corrupted)
    historical_filled, historical_details, historical_years = fill_with_historical_interval_mean(corrupted, dates)
    filled["Historical"] = historical_filled
    block_keys = ["Variable", "Start_index", "End_index"]
    gap_lookup = gaps[block_keys + ["Gap_ID"]]
    historical_details = historical_details.merge(gap_lookup, on=block_keys, how="left", validate="one_to_one")
    historical_years = historical_years.merge(gap_lookup, on=block_keys, how="left", validate="many_to_one")
    historical_details["Gap_type"] = np.where(historical_details.Gap_ID.notna(), "artificial", "original_missing")
    hidden, tables = evaluate_gaps(original, dates, filled, gaps)
    best_methods = select_best_methods(tables["metricas_por_variavel"], all_columns)
    print("Reaplicando os métodos escolhidos aos vazios reais da base original...", flush=True)
    consolidated = build_consolidated_dataset(dataset, original, dates, best_methods)
    simple_table = simple_result_table(tables["metricas_por_variavel"], best_methods, report_columns)

    report_gaps = gaps.loc[gaps.Variable.isin(report_columns)].copy()
    report_hidden = hidden.loc[hidden.Variable.isin(report_columns)].copy()
    report_tables = {name: table.loc[table.Variable.isin(report_columns)].copy()
                     for name, table in tables.items()}
    report_history = historical_details.loc[historical_details.Variable.isin(report_columns)].copy()
    report_years = historical_years.loc[historical_years.Variable.isin(report_columns)].copy()
    report_best = best_methods.loc[best_methods.Variable.isin(report_columns)].copy()

    output.mkdir(parents=True, exist_ok=True)
    clear_previous_generated_reports(output)
    # ÚNICO CSV de dados preenchidos com TODAS as séries. Os demais são relatórios selecionados.
    save_csv(consolidated, output / FINAL_CSV_NAME)
    report_corrupted = corrupted[report_columns].copy()
    report_corrupted.insert(0, time_column, dates)
    save_csv(report_corrupted, output / "dataset_com_falhas.csv")
    mask_output = mask[report_columns].astype(int).copy()
    mask_output.insert(0, time_column, dates)
    save_csv(mask_output, output / "mascara_falhas_artificiais.csv")
    save_csv(report_gaps, output / "registro_falhas_artificiais.csv")
    save_csv(report_gaps[report_gaps.Selection_strategy.eq("wet_flow_peak")], output / "picos_incluidos_nas_falhas.csv")
    save_csv(report_history, output / "detalhes_media_historica.csv")
    save_csv(report_years, output / "doadores_media_historica_por_ano.csv")
    save_csv(report_hidden, output / "valores_ocultados_e_estimados.csv")
    save_csv(report_best, output / "metodos_escolhidos.csv")
    for name, table in report_tables.items():
        save_csv(table, output / f"{name}.csv")
    missing_summary = pd.DataFrame({"Variable": report_columns,
        "Original_missing": original[report_columns].isna().sum().to_numpy(),
        "Artificial_missing": mask[report_columns].sum().to_numpy(),
        "Missing_in_final_dataset": consolidated[report_columns].isna().sum().to_numpy()})
    save_csv(missing_summary, output / "resumo_falhas.csv")
    save_simple_result_table(simple_table, output)
    figures = output / "figuras"
    figures.mkdir(parents=True, exist_ok=True)
    print("Gerando figuras somente das séries solicitadas...", flush=True)
    save_full_series_figures(figures / "series_completas", report_gaps, original, corrupted, dates, filled)
    for column in report_columns:
        variable_gaps = report_gaps.loc[report_gaps.Variable.eq(column)]
        if variable_gaps.empty:
            continue
        save_example_figure(figures / f"Comparacao_{safe_name(column)}.png", variable_gaps,
                            original, corrupted, dates, filled)
        save_long_gap_example(figures / f"Longa_{safe_name(column)}.png", variable_gaps,
                             original, corrupted, dates, filled)
        variable_period_metrics = report_tables["metricas_por_periodo"].loc[
            report_tables["metricas_por_periodo"].Variable.eq(column)]
        save_metric_figure(figures / f"Metricas_{safe_name(column)}.png", variable_period_metrics)
    if not args.skip_gap_plots:
        save_zoom_figures(figures / "falhas_individuais", report_gaps, original, corrupted, dates,
                          filled, report_tables["metricas_por_falha"])
    n_html = 0
    if CREATE_INTERACTIVE_HTML and not args.skip_interactive:
        n_html = save_interactive_figures(output / "html_interativos", report_gaps, original, corrupted, dates, filled)

    checks["all_methods_filled_all_withheld_values"] = True
    checks["methods_preserved_available_observations"] = True
    checks["metrics_computed_only_on_artificial_gaps"] = True
    checks["historical_donors_only_from_available_observations_and_other_calendar_intervals"] = True
    checks["historical_method_does_not_use_imputed_donors"] = True
    checks["n_metric_rows_by_gap_internal"] = len(tables["metricas_por_falha"])
    checks["n_metric_rows_by_gap_exported"] = len(report_tables["metricas_por_falha"])
    checks["reported_series"] = report_columns
    checks["final_dataset_preserves_all_original_observations"] = True
    checks["artificial_validation_gaps_restored_to_observed_values_in_final_dataset"] = True
    checks["final_missing_values"] = int(consolidated[all_columns].isna().to_numpy().sum())
    checks["best_method_selection"] = best_methods.Selection_criterion.value_counts().to_dict()
    (output / "verificacoes.json").write_text(json.dumps(checks, ensure_ascii=False, indent=2), encoding="utf-8")
    metadata = {"input_file": input_path.name, "input_sha256": hashlib.sha256(input_path.read_bytes()).hexdigest(),
                "first_date": str(dates.min().date()), "last_date": str(dates.max().date()),
                "n_days": len(dataset), "validation_columns": all_columns,
                "report_columns": report_columns, "seed": args.seed,
                "dry_months": DRY_MONTHS, "wet_months": WET_MONTHS,
                "gap_lengths_days": sorted(set(args.durations)), "repetitions_per_season": args.repetitions,
                "wet_flow_gaps_at_peaks": WET_FLOW_GAPS_AT_PEAKS,
                "wet_peak_quantile": WET_PEAK_QUANTILE, "peak_half_window_days": PEAK_HALF_WINDOW_DAYS,
                "historical_mean_rule": "pooled available observations in the same month/day interval of other years",
                "historical_partial_intervals": "use available observed days",
                "historical_imputed_donors": False, "n_methods": len(filled),
                "selection_metric": "maximum pooled NSE on dry and wet artificial gaps per series",
                "final_fill_input": "original data with only real missing values",
                "final_csv_file": FINAL_CSV_NAME, "n_filled_series": len(all_columns),
                "methods_used_in_final_dataset": dict(zip(best_methods.Variable, best_methods.Method)),
                "knn_neighbors": KNN_NEIGHBORS, "knn_scaling": False, "n_interactive_html": n_html,
                "python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__,
                "sklearn": sklearn.__version__, "matplotlib": matplotlib.__version__}
    (output / "configuracao_experimento.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nBase final: {output / FINAL_CSV_NAME}", flush=True)
    print(f"Todas as {len(all_columns)} séries preservadas; vazios restantes: {checks['final_missing_values']}.", flush=True)
    print(f"Tabela simples: {output / 'tabela_resultados.html'}", flush=True)
    return {"output": output, "final_csv": output / FINAL_CSV_NAME,
            "checks": checks, "metrics": report_tables,
            "best_methods": best_methods, "simple_table": simple_table}


if __name__ == "__main__":
    main()

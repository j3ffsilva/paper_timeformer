"""
Testa se o sinal APD=0.21 sobrevive ao controle de frequência.

Experimentos:
1. Partial Spearman(APD, gold) controlando log(frequência total)
2. Partial Spearman por grupo binário (changed vs stable)
3. APD balanceado: reamostrar min(n_D0, n_D1) ocorrências por período
   repetido em 20 seeds — requer os tensores de ocorrências em cache
"""

import argparse
import json
import math
import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats
from scipy.stats import spearmanr

SCORES_CSV  = Path("outputs/semeval2020_pmi_dynamic_mlm_12_8_d128/hidden_relational_profiles/scores.csv")
TRUTH_TSV   = Path("data/processed/semeval2020_task1/eng_lemma/truth.tsv")
CACHE_DIR   = Path("outputs/semeval2020_pmi_dynamic_mlm_12_8_d128/hidden_relational_profiles/cache")
OUT_DIR     = Path("outputs/semeval2020_pmi_dynamic_mlm_12_8_d128/frequency_confound_analysis")


# ── helpers ──────────────────────────────────────────────────────────────────

def partial_spearman(x, y, z):
    """Spearman(x, y) parcializando z via resíduos de regressão linear."""
    def residuals(a, b):
        slope, intercept = np.polyfit(b, a, 1)
        return a - (slope * b + intercept)

    rx = residuals(stats.rankdata(x), z)
    ry = residuals(stats.rankdata(y), z)
    return stats.pearsonr(rx, ry)


def load_data(method: str = "relational_apd", layer: str = "layer_2"):
    scores = pd.read_csv(SCORES_CSV)
    scores = scores[(scores["method"] == method) & (scores["layer"] == layer)].copy()

    truth = pd.read_csv(TRUTH_TSV, sep="\t")
    df = scores.merge(truth, on="target")
    df["log_freq"] = np.log(df["count_d0"] + df["count_d1"] + 1)
    df["log_freq_min"] = np.log(df[["count_d0", "count_d1"]].min(axis=1) + 1)
    return df


# ── análise 1: partial Spearman ───────────────────────────────────────────────

def run_partial_spearman(df):
    print("\n=== 1. Partial Spearman (APD vs gold, controlando log-frequência) ===\n")

    # bruto
    rho_raw, p_raw = spearmanr(df["score"], df["graded"])
    print(f"Spearman bruto:                  rho={rho_raw:.4f}  p={p_raw:.4f}")

    # controlando log(total)
    rho_part, p_part = partial_spearman(
        df["score"].values, df["graded"].values, df["log_freq"].values
    )
    print(f"Partial Spearman (log_freq_tot): rho={rho_part:.4f}  p={p_part:.4f}")

    # controlando log(min)
    rho_min, p_min = partial_spearman(
        df["score"].values, df["graded"].values, df["log_freq_min"].values
    )
    print(f"Partial Spearman (log_freq_min): rho={rho_min:.4f}  p={p_min:.4f}")

    # frequência sozinha vs gold
    rho_fg, p_fg = spearmanr(df["log_freq"], df["graded"])
    print(f"\nSpearman(log_freq, gold):        rho={rho_fg:.4f}  p={p_fg:.4f}")

    # frequência sozinha vs APD
    rho_fa, p_fa = spearmanr(df["log_freq"], df["score"])
    print(f"Spearman(log_freq, APD):         rho={rho_fa:.4f}  p={p_fa:.4f}")

    print("\n  Interpretação:")
    print(f"  - APD correlaciona {abs(rho_fa/rho_fg):.1f}x mais com frequência do que o gold.")
    if abs(rho_part) < 0.05:
        print("  - Partial Spearman próximo de 0 → sinal APD é artefato de frequência.")
    elif abs(rho_part) >= 0.15:
        print("  - Partial Spearman preservado → existe sinal além do confundidor de frequência.")
    else:
        print("  - Partial Spearman ambíguo — sinal reduzido mas não eliminado.")

    return {"rho_raw": rho_raw, "p_raw": p_raw,
            "rho_partial": rho_part, "p_partial": p_part,
            "rho_partial_min": rho_min, "p_partial_min": p_min}


# ── análise 2: sinal dentro de cada grupo binário ────────────────────────────

def run_group_analysis(df):
    print("\n=== 2. APD por grupo binário ===\n")

    changed  = df[df["binary"] == 1]
    stable   = df[df["binary"] == 0]

    print(f"Grupo changed (n={len(changed)}): APD médio={changed['score'].mean():.4f}  "
          f"freq_total média={changed['count_d0'].add(changed['count_d1']).mean():.0f}")
    print(f"Grupo stable  (n={len(stable)}):  APD médio={stable['score'].mean():.4f}  "
          f"freq_total média={stable['count_d0'].add(stable['count_d1']).mean():.0f}")

    print("\nTop-5 APD (mais 'mudadas' pelo método):")
    top = df.nlargest(5, "score")[["target","score","graded","binary","count_d0","count_d1"]]
    print(top.to_string(index=False))

    print("\nBottom-5 APD (menos 'mudadas'):")
    bot = df.nsmallest(5, "score")[["target","score","graded","binary","count_d0","count_d1"]]
    print(bot.to_string(index=False))

    # frequência dentro de cada grupo
    print("\nCorrelação(APD, log_freq) dentro de cada grupo:")
    for label, grp in [("changed", changed), ("stable", stable)]:
        r, p = spearmanr(grp["score"], grp["log_freq"])
        print(f"  {label}: rho={r:.4f}  p={p:.4f}  (n={len(grp)})")


# ── análise 3: APD por faixa de frequência ───────────────────────────────────

def run_frequency_bins(df):
    print("\n=== 3. APD e Spearman por faixa de frequência ===\n")

    df = df.copy()
    df["freq_total"] = df["count_d0"] + df["count_d1"]

    # tercis de frequência
    tercis = pd.qcut(df["freq_total"], q=3, labels=["baixa", "média", "alta"])
    df["freq_bin"] = tercis

    for label in ["baixa", "média", "alta"]:
        grp = df[df["freq_bin"] == label]
        rho, p = spearmanr(grp["score"], grp["graded"])
        print(f"Frequência {label} (n={len(grp)}, "
              f"freq {int(grp['freq_total'].min())}–{int(grp['freq_total'].max())}): "
              f"Spearman={rho:.4f}  p={p:.3f}")


# ── análise 4: balanceamento por amostragem ──────────────────────────────────

def run_balanced_sampling(df, n_seeds: int = 20):
    """
    Para cada palavra, calcula APD balanceado usando min(n_D0, n_D1) ocorrências
    por período, reamostradas com reposição. Requer tensores .pt no cache.

    Se o cache não tiver tensores de ocorrências individuais, reporta e pula.
    """
    print("\n=== 4. APD balanceado (amostragem por min(n_D0, n_D1)) ===\n")

    import torch

    cache_files = list(CACHE_DIR.glob("*.pt"))
    if not cache_files:
        print("  Cache vazio ou sem tensores .pt — pulando balanceamento.")
        print(f"  (esperado em {CACHE_DIR})")
        return

    # descobrir estrutura do cache
    sample_key = cache_files[0].stem
    print(f"  Arquivos cache encontrados: {len(cache_files)}")
    print(f"  Exemplo: {cache_files[0].name}")

    sample = torch.load(cache_files[0], map_location="cpu")
    print(f"  Tipo do objeto: {type(sample)}")
    if isinstance(sample, dict):
        print(f"  Chaves: {list(sample.keys())}")
    elif isinstance(sample, torch.Tensor):
        print(f"  Shape: {sample.shape}")
    print("\n  (implementação de balanceamento requer estrutura conhecida do cache)")


# ── análise 5: all methods comparison ────────────────────────────────────────

def run_all_methods():
    print("\n=== 5. Partial Spearman para todos os métodos ===\n")
    scores = pd.read_csv(SCORES_CSV)
    truth  = pd.read_csv(TRUTH_TSV, sep="\t")

    rows = []
    for (method, layer), grp in scores.groupby(["method", "layer"]):
        df = grp.merge(truth, on="target")
        df["log_freq"] = np.log(df["count_d0"] + df["count_d1"] + 1)
        rho_raw, p_raw = spearmanr(df["score"], df["graded"])
        rho_part, p_part = partial_spearman(
            df["score"].values, df["graded"].values, df["log_freq"].values
        )
        rows.append({
            "method": method, "layer": layer,
            "spearman_raw": round(rho_raw, 4),
            "p_raw": round(p_raw, 4),
            "spearman_partial": round(rho_part, 4),
            "p_partial": round(p_part, 4),
            "delta": round(rho_raw - rho_part, 4),
        })

    result = pd.DataFrame(rows).sort_values("spearman_raw", ascending=False)
    print(result.to_string(index=False))
    return result


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    df_main = load_data("relational_apd", "layer_2")
    print(f"Targets carregados: {len(df_main)}")

    results = {}
    results["partial_spearman"] = run_partial_spearman(df_main)
    run_group_analysis(df_main)
    run_frequency_bins(df_main)
    run_balanced_sampling(df_main)
    all_methods = run_all_methods()

    # salva resultado principal
    out = OUT_DIR / "partial_spearman_results.json"
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResultados salvos em {out}")

    all_methods.to_csv(OUT_DIR / "all_methods_partial_spearman.csv", index=False)
    print(f"Tabela completa em {OUT_DIR / 'all_methods_partial_spearman.csv'}")


if __name__ == "__main__":
    main()

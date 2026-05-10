"""
analise_resultados.py
─────────────────────
Análise estatística completa dos resultados do Mini Estudo 01.
Gera para cada métrica (f1score, tempo_ms, cpu_time_ms):
  - Resumo estatístico (média, mediana, mín, máx, desvio padrão)
  - Intervalo de confiança 95% da média
  - Histograma
  - Boxplot
  - Série temporal dos valores brutos por repetição
  - Interpretação e limitações impressas no terminal
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy import stats

# ══════════════════════════════════════════════════════════════════════════════
# 1. CARREGAR DADOS
# ══════════════════════════════════════════════════════════════════════════════
CSV_PATH = "resultados.csv"
df = pd.read_csv(CSV_PATH)

filtros      = df["filtro"].unique()
metricas     = ["f1score", "tempo_ms", "cpu_time_ms"]
nomes_metricas = {
    "f1score":     "F1-Score",
    "tempo_ms":    "Tempo de Execução (ms)",
    "cpu_time_ms": "Tempo de CPU (ms)",
}

CONFIANCA = 0.95   # nível do intervalo de confiança
N_MIN_IC  = 30     # mínimo de amostras para calcular IC com confiança

# ══════════════════════════════════════════════════════════════════════════════
# 2. FUNÇÕES AUXILIARES
# ══════════════════════════════════════════════════════════════════════════════

def resumo_estatistico(serie: pd.Series) -> dict:
    """Calcula média, mediana, mín, máx, desvio padrão e IC 95%."""
    n    = len(serie)
    med  = serie.mean()
    mdn  = serie.median()
    mn   = serie.min()
    mx   = serie.max()
    std  = serie.std(ddof=1)

    if n >= 2:
        # IC via t de Student (robusto mesmo para n < 30)
        t_critico  = stats.t.ppf((1 + CONFIANCA) / 2, df=n - 1)
        margem     = t_critico * (std / np.sqrt(n))
        ic_inf     = med - margem
        ic_sup     = med + margem
        ic_nota    = f"t de Student (n={n})"
    else:
        ic_inf = ic_sup = med
        ic_nota = "amostra insuficiente"

    return {
        "n":       n,
        "média":   med,
        "mediana": mdn,
        "mín":     mn,
        "máx":     mx,
        "std":     std,
        "ic_inf":  ic_inf,
        "ic_sup":  ic_sup,
        "ic_nota": ic_nota,
    }


def interpretar(metrica: str, filtro: str, r: dict) -> str:
    """Gera texto de interpretação automática baseado nos valores."""
    cv = (r["std"] / r["média"] * 100) if r["média"] != 0 else 0  # coef. de variação
    amplitude = r["máx"] - r["mín"]

    linhas = [f"\n{'─'*60}",
              f"  Filtro: {filtro} | Métrica: {nomes_metricas[metrica]}",
              f"{'─'*60}"]

    # Resumo numérico
    linhas.append(f"  Média   : {r['média']:.4f}")
    linhas.append(f"  Mediana : {r['mediana']:.4f}")
    linhas.append(f"  Mín/Máx : {r['mín']:.4f} / {r['máx']:.4f}  (amplitude {amplitude:.4f})")
    linhas.append(f"  Std     : {r['std']:.4f}  (CV {cv:.1f}%)")
    linhas.append(f"  IC 95%  : [{r['ic_inf']:.4f}, {r['ic_sup']:.4f}]  — {r['ic_nota']}")

    # Interpretação
    linhas.append("\n  Interpretação:")
    if metrica == "f1score":
        if r["média"] >= 0.7:
            linhas.append("  → F1 alto: o filtro detecta bordas com boa precisão e recall.")
        elif r["média"] >= 0.4:
            linhas.append("  → F1 moderado: desempenho mediano, com erros relevantes.")
        else:
            linhas.append("  → F1 baixo: muitos falsos positivos ou negativos.")
        if cv > 20:
            linhas.append(f"  → CV={cv:.1f}%: resultados inconsistentes entre imagens.")
        else:
            linhas.append(f"  → CV={cv:.1f}%: resultados estáveis entre imagens.")

    elif metrica in ("tempo_ms", "cpu_time_ms"):
        label = "tempo de execução" if metrica == "tempo_ms" else "tempo de CPU"
        linhas.append(f"  → {label.capitalize()} médio de {r['média']:.2f} ms.")
        if cv > 30:
            linhas.append(f"  → CV={cv:.1f}%: alta variação — possível influência de cache ou carga do SO.")
        else:
            linhas.append(f"  → CV={cv:.1f}%: execução estável.")

    # Limitações
    linhas.append("\n  Limitações:")
    if r["n"] < N_MIN_IC:
        linhas.append(f"  → Apenas {r['n']} amostras: IC calculado, mas interpretar com cautela.")
    else:
        linhas.append(f"  → {r['n']} amostras: IC confiável.")
    linhas.append("  → Medição de tempo inclui variações do SO (escalonamento, cache).")
    linhas.append("  → cpu_time_ms não captura alocações internas do OpenCV (C++).")
    if metrica == "f1score":
        linhas.append("  → F1 depende do limiar fixo (127); outros limiares dariam resultados diferentes.")

    return "\n".join(linhas)


# ══════════════════════════════════════════════════════════════════════════════
# 3. ANÁLISE E VISUALIZAÇÃO POR MÉTRICA
# ══════════════════════════════════════════════════════════════════════════════

for metrica in metricas:

    fig = plt.figure(figsize=(16, 5 * len(filtros)))
    fig.suptitle(f"Análise: {nomes_metricas[metrica]}", fontsize=15, fontweight="bold", y=1.01)

    gs = gridspec.GridSpec(len(filtros), 3, figure=fig, hspace=0.5, wspace=0.35)

    for linha, filtro in enumerate(filtros):
        dados = df[df["filtro"] == filtro][metrica].dropna()
        r     = resumo_estatistico(dados)

        # ── Impressão no terminal ──────────────────────────────────────────
        print(interpretar(metrica, filtro, r))

        # ── Histograma ────────────────────────────────────────────────────
        ax1 = fig.add_subplot(gs[linha, 0])
        ax1.hist(dados, bins="auto", color="#4C72B0", edgecolor="white", alpha=0.85)
        ax1.axvline(r["média"],   color="red",    linestyle="--", linewidth=1.5, label=f"Média {r['média']:.3f}")
        ax1.axvline(r["mediana"], color="orange", linestyle=":",  linewidth=1.5, label=f"Mediana {r['mediana']:.3f}")
        ax1.set_title(f"{filtro} — Histograma")
        ax1.set_xlabel(nomes_metricas[metrica])
        ax1.set_ylabel("Frequência")
        ax1.legend(fontsize=8)

        # ── Boxplot ───────────────────────────────────────────────────────
        ax2 = fig.add_subplot(gs[linha, 1])
        bp = ax2.boxplot(dados, vert=True, patch_artist=True,
                         boxprops=dict(facecolor="#4C72B0", alpha=0.6),
                         medianprops=dict(color="orange", linewidth=2),
                         whiskerprops=dict(linewidth=1.5),
                         capprops=dict(linewidth=1.5),
                         flierprops=dict(marker="o", markersize=4, alpha=0.5))
        # IC como banda sombreada
        ax2.axhspan(r["ic_inf"], r["ic_sup"], color="red", alpha=0.15,
                    label=f"IC 95% [{r['ic_inf']:.3f}, {r['ic_sup']:.3f}]")
        ax2.set_title(f"{filtro} — Boxplot + IC 95%")
        ax2.set_ylabel(nomes_metricas[metrica])
        ax2.set_xticks([])
        ax2.legend(fontsize=8)

        # ── Série temporal (valores brutos por execução) ───────────────────
        ax3 = fig.add_subplot(gs[linha, 2])
        # Índice global de execução (1 ponto por imagem×repetição)
        x = np.arange(len(dados))
        ax3.scatter(x, dados.values, s=8, alpha=0.5, color="#4C72B0")
        ax3.axhline(r["média"],   color="red",    linestyle="--", linewidth=1.2, label=f"Média")
        ax3.axhline(r["mediana"], color="orange", linestyle=":",  linewidth=1.2, label=f"Mediana")
        ax3.fill_between(x, r["ic_inf"], r["ic_sup"], color="red", alpha=0.1, label="IC 95%")
        ax3.set_title(f"{filtro} — Valores Brutos")
        ax3.set_xlabel("Execução (imagem × repetição)")
        ax3.set_ylabel(nomes_metricas[metrica])
        ax3.legend(fontsize=8)

    plt.tight_layout()
    nome_fig = f"analise_{metrica}.png"
    plt.savefig(nome_fig, dpi=150, bbox_inches="tight")
    print(f"\n✅ Gráfico salvo: {nome_fig}")
    plt.close()

# ══════════════════════════════════════════════════════════════════════════════
# 4. COMPARAÇÃO ENTRE FILTROS (boxplot lado a lado)
# ══════════════════════════════════════════════════════════════════════════════

fig, axes = plt.subplots(1, len(metricas), figsize=(6 * len(metricas), 5))
fig.suptitle("Comparação entre Filtros", fontsize=14, fontweight="bold")

for ax, metrica in zip(axes, metricas):
    grupos = [df[df["filtro"] == f][metrica].dropna().values for f in filtros]
    bp = ax.boxplot(grupos, labels=filtros, patch_artist=True,
                    medianprops=dict(color="black", linewidth=2))
    cores = ["#4C72B0", "#DD8452", "#55A868", "#C44E52"]
    for patch, cor in zip(bp["boxes"], cores):
        patch.set_facecolor(cor)
        patch.set_alpha(0.7)
    ax.set_title(nomes_metricas[metrica])
    ax.set_ylabel(nomes_metricas[metrica])
    ax.set_xlabel("Filtro")

plt.tight_layout()
plt.savefig("comparacao_filtros.png", dpi=150, bbox_inches="tight")
print("\n✅ Gráfico salvo: comparacao_filtros.png")
plt.close()

print("\n✅ Análise concluída.")
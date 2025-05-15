import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

sns.set_theme(palette='bright')

PALETTE = 'coolwarm'
SCATTER_ALPHA = 0.2

# Plota a matriz de correlação de Pearson
def plot_correl_matrix(df: pd.DataFrame, figsize: tuple = (12,8), fontsize: int = 8, focus: float = None) -> None:
    sns.set_theme(style='white')
    
    col_num = df.select_dtypes(include='number').columns
    
    corr = df[col_num].corr()
    
    if focus:
        corr = corr.where(abs(corr) >= focus, other=np.nan)
    
    mask = np.triu(corr)

    fig, ax = plt.subplots(figsize=(12, 8))

    sns.heatmap(corr, mask=mask, annot=True, annot_kws={"size": fontsize}, fmt=".2f", ax=ax, cmap=PALETTE, vmin=-1, vmax=1, center=0)

    plt.tight_layout()
    plt.show()

# Plota coeficientes dos modelos
def plot_coeficientes(df_coefs: pd.DataFrame, tituto: str ="Coeficientes") -> None:
    df_coefs.plot.barh()
    plt.title(tituto)
    plt.axvline(x=0, color=".5")
    plt.xlabel("Coeficientes")
    plt.gca().get_legend().remove()
    plt.show()

# Plota comparação entre métricas dos modelos
def plot_comparar_metricas_modelos(df_resultados: pd.DataFrame) -> None:
    fig, axs = plt.subplots(4, 2, figsize=(9, 9), sharex=True)

    comparar_metricas = [
        "time_seconds",
        "test_accuracy",
        "test_balanced_accuracy",
        "test_f1",
        "test_precision",
        "test_recall",
        "test_roc_auc",
        "test_average_precision",
    ]

    nomes_metricas = [
        "Tempo (s)",
        "Acurácia",
        "Acurácia balanceada",
        "F1",
        "Precisão",
        "Recall",
        "AUROC",
        "AUPRC",
    ]

    for ax, metrica, nome in zip(axs.flatten(), comparar_metricas, nomes_metricas):
        sns.boxplot(
            x="model",
            y=metrica,
            data=df_resultados,
            ax=ax,
            showmeans=True,
        )
        ax.set_title(nome)
        ax.set_ylabel(nome)
        ax.tick_params(axis="x", rotation=90)

    plt.tight_layout()

    plt.show()

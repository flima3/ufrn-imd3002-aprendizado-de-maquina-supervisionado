import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

sns.set_theme(palette='bright')

PALETTE = 'coolwarm'
SCATTER_ALPHA = 0.2

# Plota a matriz de correlação de Pearson
def plot_correl_matrix(df: pd.DataFrame, figsize: tuple = (12,8), fontsize: int = 8, focus: float = None) -> None:
    
    # Define tema para o gráfico
    sns.set_theme(style='white')
    
    # Realiza a seleção das colunas numéricas do dataframe
    col_num = df.select_dtypes(include='number').columns
    # Calcula a correlação entre as variáveis
    corr = df[col_num].corr()
    # Aplica camada de foco
    if focus:
        corr = corr.where(abs(corr) >= focus, other=np.nan)
    # Máscara para mostrar apenas diagonal inferior da matriz
    mask = np.triu(corr)

    fig, ax = plt.subplots(figsize=(12, 8))

    sns.heatmap(corr, mask=mask, annot=True, annot_kws={"size": fontsize}, fmt=".2f", ax=ax, cmap=PALETTE, vmin=-1, vmax=1, center=0)

    plt.tight_layout()
    plt.show()
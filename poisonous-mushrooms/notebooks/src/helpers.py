import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency
    
def cramers_v(x, y):
    """
    Calcula a associação entre duas variáveis categóricas usando Cramér's V corrigido.

    Parâmetros:
    x (pd.Series): Variável categórica 1
    y (pd.Series): Variável categórica 2

    Retorno:
    float: Valor entre 0 (sem associação) e 1 (associação perfeita)
    """
    if not isinstance(x, pd.Series) or not isinstance(y, pd.Series):
        raise ValueError("As entradas devem ser pandas Series.")
    if len(x) != len(y):
        raise ValueError("As séries devem ter o mesmo comprimento.")
    
    # Remove entradas com NaN em pelo menos uma das variáveis
    valid = x.notna() & y.notna()
    x = x[valid]
    y = y[valid]

    if len(x) == 0 or len(y) == 0:
        return np.nan

    confusion_matrix = pd.crosstab(x, y)
    chi2 = chi2_contingency(confusion_matrix)[0]
    n = confusion_matrix.sum().sum()
    phi2 = chi2 / n
    r, k = confusion_matrix.shape
    phi2corr = max(0, phi2 - ((k - 1)*(r - 1)) / (n - 1))
    rcorr = r - ((r - 1) ** 2) / (n - 1)
    kcorr = k - ((k - 1) ** 2) / (n - 1)
    return np.sqrt(phi2corr / min((kcorr - 1), (rcorr - 1)))


def analisar_cardinalidade(df):
    """
    Analisa a cardinalidade de colunas categóricas de um DataFrame.

    Parâmetros:
    df (pd.DataFrame): DataFrame com colunas categóricas.

    Retorno:
    pd.DataFrame: Tabela com colunas:
        - feature (str): nome da coluna categórica
        - n_categorias (int): número de categorias únicas
        - pct_valor_mais_frequente (float): % do valor mais comum
    """
    if not isinstance(df, pd.DataFrame):
        raise ValueError("A entrada deve ser um pandas DataFrame.")

    resultados = []
    for col in df.select_dtypes(include=['category', 'object']).columns:
        series = df[col].dropna()
        if series.empty:
            pct = np.nan
        else:
            pct = round(series.value_counts(normalize=True).iloc[0] * 100, 2)
        resultados.append({
            'feature': col,
            'n_categorias': df[col].nunique(),
            'pct_valor_mais_frequente': pct
        })

    return pd.DataFrame(resultados).sort_values(by='n_categorias', ascending=False).reset_index(drop=True)

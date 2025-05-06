import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency

from sklearn.metrics import (
    accuracy_score,
    f1_score
)

def display_metricas(y_true, y_pred):
    print("Accuracy:", format(accuracy_score(y_true, y_pred), ".4f"))
    print("F1-Score:", format(f1_score(y_true, y_pred), ".4f"))
    
def cramers_v(x, y):
    confusion_matrix = pd.crosstab(x, y)
    chi2 = chi2_contingency(confusion_matrix)[0]
    n = confusion_matrix.sum().sum()
    phi2 = chi2 / n
    r, k = confusion_matrix.shape
    phi2corr = max(0, phi2 - ((k-1)*(r-1))/(n-1))
    rcorr = r - ((r-1)**2)/(n-1)
    kcorr = k - ((k-1)**2)/(n-1)
    return np.sqrt(phi2corr / min((kcorr-1), (rcorr-1)))

def analisar_cardinalidade(df):
    resultados = []
    for col in df.select_dtypes(include='object').columns:
        valor_mais_comum = df[col].value_counts(normalize=True).iloc[0]
        resultados.append({
            'feature': col,
            'n_categorias': df[col].nunique(),
            'valor_mais_frequente_%': round(valor_mais_comum * 100, 2)
        })
    return pd.DataFrame(resultados).sort_values(by='n_categorias', ascending=False).reset_index(drop=True)
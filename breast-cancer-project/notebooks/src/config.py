from pathlib import Path

PASTA_PROJETO = Path(__file__).resolve().parents[2]

# dados
PASTA_DADOS_RAW = PASTA_PROJETO / 'data' / 'raw'
PASTA_DADOS_PROCESSED = PASTA_PROJETO / 'data' / 'processed'

DADOS_ORIGINAIS = PASTA_DADOS_RAW / 'breast-cancer.csv'
DADOS_TRATADOS = PASTA_DADOS_PROCESSED / 'data.csv'
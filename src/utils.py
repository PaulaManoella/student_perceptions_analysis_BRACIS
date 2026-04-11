import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Diretório do arquivo utils.py
ROOT = Path(__file__).resolve().parent.parent   
DATA = ROOT / "data"


def read_dataset():
    bases = ['18.2', '18.4', '19.2', '19.4', '22.2', '22.4', '23.2', '23.4', '24.2', '24.4']
    avalia_anos = []

    for ano in bases:
        # print('lendo: ' + f'avalia-20{ano}.csv')

        csv_path = DATA / f"avalia-20{ano}.csv"
        xlsx_path = DATA / f"avalia-20{ano}.xlsx"

        if ano == '24.2':
            avalia_ano = pd.read_excel(xlsx_path)
        else:
            avalia_ano = pd.read_csv(csv_path, delimiter=';', encoding='latin1', on_bad_lines='skip')

        avalia_ano.columns.values[0] = 'ANO-PERIODO'
        avalia_anos.append(avalia_ano)

    return avalia_anos

def concat_dataset(dataset):
  return pd.concat(dataset, ignore_index=True)

def drop_null_duplicate(df, col:str, values_to_remove):
    df = df[~df[col].isin(values_to_remove)]
    df = df.drop_duplicates(subset=[col], keep="first").reset_index(drop=True)
    return df

def get_model_topics(model, top_n=15):
    """
    Extrai a lista de palavras de modelos Gensim (LDA, LSA/LSI).
    Retorna: List[List[str]] no formato que o CoherenceModel exige.
    """
    # num_topics=-1 garante que ele pegue TODOS os tópicos do modelo
    # num_words=top_n define quantas palavras por tópico
    # formatted=False retorna a lista de tuplas (palavra, peso) em vez de string
    topics_data = model.show_topics(num_topics=-1, num_words=top_n, formatted=False)
    
    topics_list = []
    
    # O retorno é [(id_topico, [(palavra, peso), ...]), ...]
    # O Gensim nem sempre retorna os tópicos na ordem 0, 1, 2... então é bom ordenar pelo ID
    topics_data.sort(key=lambda x: x[0])
    
    for topic_id, word_weight_list in topics_data:
        # Extraímos apenas a palavra, ignorando o peso
        words = [word for word, weight in word_weight_list]
        topics_list.append(words)
        
    return topics_list

import pandas as pd
from src import utils

bases = ['lsa', 'lda', 'nmf', 'bertopic']

dataset_list = []
for model in bases:
    df = pd.read_excel(f'data/topicos_{model}.xlsx')

    if model == 'bertopic':
        # Deleta a coluna 'docs' se ela existir
        if 'docs' in df.columns:
            df = df.drop(columns=['docs'])

    # Adiciona a coluna com o nome do modelo na primeira posição (índice 0)
    df.insert(0, 'model', model)

    dataset_list.append(df)

df_concat = pd.concat(dataset_list, ignore_index=True)
print(df_concat)

df_concat.to_csv('output/topic_modeling/all_models_topics.csv', index=False, encoding='utf-8-sig')
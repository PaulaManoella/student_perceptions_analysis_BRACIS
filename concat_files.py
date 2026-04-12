import pandas as pd
from src import utils

bases = ['lsa', 'lda', 'nmf', 'bertopic']

dataset_list = []
for model in bases:
    df = pd.read_excel(f'data/topicos_{model}.xlsx')
    dataset_list.append(df)

df_concat = utils.concat_dataset(dataset_list)
print(df_concat)

df_concat.to_csv('output/topic_modeling/all_models_topics.csv', index=False, sep=';')
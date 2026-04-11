import pandas as pd
from gensim.corpora import Dictionary
from gensim.models import TfidfModel
from gensim.matutils import corpus2csc
from gensim.models.phrases import Phrases, Phraser
from src import pre_proces, transformacao

def merge_comentario_columns(df, df_cols:list):
    df['ID'] = df.index
    
    df_long = pd.melt(  df,
                        id_vars=['ID'],
                        value_vars=df_cols,
                        var_name='id',
                        value_name='comentario')
    
    df_long['label'] = df_long['id'].apply(
    lambda x: 1 if "POSITIVO" in x else (2 if "GERAL" in x else 0))

    df_long = df_long[['ID', 'comentario', 'label']]
    
    return df_long

def make_dict(df_col):
    dicionario = Dictionary(df_col)
    
    # defininindo palavras adicionais para serem removidas do dicionario
    words_to_remove = ['nada', 'nenhum', 'nenhuma', 'declarar','aluno','professor']
    
    # pegando id das palavras do dicionario
    words_ids = [
    dicionario.token2id[word]
    for word in words_to_remove
    if word in dicionario.token2id]
    
    # filtrando dicionario para remover palavras adicionais    
    dicionario.filter_tokens(bad_ids=words_ids)
    
    """
    usando 'filter_extremes' para:
        - remover palavras que aparecem em menos de 5 documentos
        - remover palavras que aparecem em mais de 50% dos documentos
    """
    # dicionario.filter_extremes(no_below=5, no_above=0.5)
    
    return dicionario

def make_corpus(dicionario, coment_col):
   return [dicionario.doc2bow(text) for text in coment_col]

def make_tfidf(BoW):
    tfidf_model = TfidfModel(BoW)
    tfidf_corpus = tfidf_model[BoW]
    
    return tfidf_corpus

def make_esparse_matrix(tfidf_corpus, dict):
    return corpus2csc(tfidf_corpus, len(dict)).T
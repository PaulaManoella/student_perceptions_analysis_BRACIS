import pandas as pd
from nltk import tokenize
import nltk
from string import punctuation
import re
import spacy
import unicodedata

nlp = spacy.load("pt_core_news_sm")

nltk.download('stopwords')
stop_words = nltk.corpus.stopwords.words('portuguese')
stop_words.extend(['nao','nenhum','ha','nenhuma'])

def normalize_string(txt):
    """ Remove acentos e coloca em minúsculo para comparação robusta. """
    if not isinstance(txt, str): return ""
    # Remove acentos (ex: 'Cássio' -> 'Cassio')
    txt = ''.join(c for c in unicodedata.normalize('NFD', txt) if unicodedata.category(c) != 'Mn')
    return txt.lower().strip()

def punct_and_number_remove(df_column):
  df_column = df_column.fillna('').astype(str)
  
  new_column_punct=[]

  token_punct =  tokenize.WordPunctTokenizer()

  for comment in df_column:
      tokens = token_punct.tokenize(comment)
      new_comment = []
      for token in tokens:
        if token not in punctuation:
          token = re.sub(r'\d', '', token)
          new_comment.append(token.lower())
      new_column_punct.append(' '.join(new_comment))

  return new_column_punct

def stopwords_docentes(nomes_docentes):
    """
    Cria lista de stopwords dos nomes DOS docentes (apenas partes individuais).
    """
    stopwords_docentes = set()
    
    for nome in nomes_docentes:
        if not nome or nome == 'nan':
            continue
            
        # Normaliza o nome
        nome_norm = normalize_string(nome)
        
        # Adiciona APENAS as partes do nome (sem nome completo)
        partes = nome_norm.split()
        for parte in partes:
            if len(parte) > 2:  # Evita preposições como "da", "de"
                stopwords_docentes.add(parte)
    
    return sorted(list(stopwords_docentes))  # Ordenado para consistência

def stopwords_remove(df_column, nomes_docentes=None):
    """
    Remove stopwords padrão e opcionalmente nomes dos docentes de uma coluna de texto.
    
    Args:
        df_column: Coluna com os textos
        nomes_docentes: Lista com nomes dos docentes (opcional)
    """
    token_espaco = tokenize.WhitespaceTokenizer()
    
    # Combina stopwords padrão com stopwords dos docentes
    all_stopwords = stop_words.copy()
    if nomes_docentes:
        stopwords_nomes_docentes = stopwords_docentes(nomes_docentes)
        all_stopwords.extend(stopwords_nomes_docentes)
    
    # Remove duplicatas
    all_stopwords = list(set(all_stopwords))
    
    comentario_sw = []
    
    for comment in df_column:
        comment = comment.lower()
        tokens = token_espaco.tokenize(comment)
        new_comment = [token for token in tokens if token not in all_stopwords]
        comentario_sw.append(' '.join(new_comment))
    
    return comentario_sw

def remove_docentes_stopwords(df_column, nomes_docentes):
    """
    Remove apenas nomes dos docentes como stopwords de uma coluna de texto.
    
    Args:
        df_column: Coluna com os textos
        nomes_docentes: Lista com nomes dos docentes
    """
    token_espaco = tokenize.WhitespaceTokenizer()
    
    # Cria stopwords dos nomes dos docentes
    stopwords_nomes_docentes = stopwords_docentes(nomes_docentes)
    
    comentario_sw = []
    
    for comment in df_column:
        comment = comment.lower()
        tokens = token_espaco.tokenize(comment)
        new_comment = [token for token in tokens if token not in stopwords_nomes_docentes]
        comentario_sw.append(' '.join(new_comment))
    
    return comentario_sw

def lemmatization(df_column):
    comentarios_lem = []

    for doc in nlp.pipe(df_column, batch_size=1000):
        # Adiciona .lower() para garantir minúsculas
        lemmas = [token.lemma_.lower() for token in doc if not token.is_punct and not token.is_space]
        comentarios_lem.append(lemmas)

    return comentarios_lem

# def pre_processamento(df):
#     df_merge = merge_df_columns(df)
#     df_merge['comentario pontuacao'] = punct_and_number_remove(df_merge['comentario'])
#     df_merge = df_merge[df_merge['comentario pontuacao'] != 'nan']
#     df_merge['comentario stopword'] = stopwords_remove(df_merge['comentario pontuacao'])
#     df_merge['comentario lematizacao'] = lemmatization(df_merge['comentario stopword'])
    
    return df_merge

def pre_processamento(df, lista_docentes):  
    colunas = [' COMENTARIO POSITIVO', ' COMENTARIO NEGATIVO', ' COMENTARIO GERAL']
    
    for col in colunas:        
        col_punct = f"{col} pontuacao"
        df[col_punct] = punct_and_number_remove(df[col])
        
        col_docentes = f"{col} stopwords_docente"
        df[col_docentes] = remove_docentes_stopwords(df[col_punct], lista_docentes)
        
        col_sw = f"{col} stopword_geral"
        df[col_sw] = stopwords_remove(df[col_punct], lista_docentes)
        
        col_lem = f"{col} lematizacao"
        df[col_lem] = lemmatization(df[col_sw])

    return df

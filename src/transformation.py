"""
Data Transformation Module.

This module provides data restructuring functionality and Gensim-based vectorization 
models. It includes merging categorical feedback columns and explicitly constructing 
Dictionaries, Bag of Words (BoW), and TF-IDF matrices (including sparse formats) 
to be used natively by LSA, LDA, and NMF baseline topic models.
"""

# Third-Party Imports
import pandas as pd
from gensim.corpora import Dictionary
from gensim.matutils import corpus2csc
from gensim.models import TfidfModel


def merge_comentario_columns(df: pd.DataFrame, df_cols: list) -> pd.DataFrame:
    """
    Melts distinct comment columns into a single long-format DataFrame 
    and applies numeric labels corresponding to the comment category.
    """
    df['ID'] = df.index

    df_long = pd.melt(
        df,
        id_vars=['ID'],
        value_vars=df_cols,
        var_name='source_col',
        value_name='comentario'
    )

    # Apply sentiment label: 0=NEGATIVE, 1=POSITIVE, 2=GENERAL
    df_long['label'] = df_long['source_col'].apply(
        lambda col_name: 1 if "POSITIVO" in col_name else (2 if "GERAL" in col_name else 0)
    )

    return df_long[['ID', 'comentario', 'label']]


def make_dict(df_col: pd.Series) -> Dictionary:
    """
    Constructs a Gensim Dictionary representing the vocabulary of the corpus 
    and removes specific extraneous or uninformative bad words.
    """
    gensim_dict = Dictionary(df_col)

    # Define additional specific manual stopwords to be removed from the dictionary
    words_to_remove = ['nada', 'nenhum', 'nenhuma', 'declarar', 'aluno', 'professor']

    # Extract structural dictionary IDs for these specific words
    words_ids = [
        gensim_dict.token2id[word]
        for word in words_to_remove
        if word in gensim_dict.token2id
    ]

    # Filter dictionary to remove these additional words
    gensim_dict.filter_tokens(bad_ids=words_ids)

    return gensim_dict


def make_corpus(gensim_dict: Dictionary, coment_col: pd.Series) -> list:
    """
    Transforms the corpus into a Bag-of-Words (BoW) vector representation.
    """
    return [gensim_dict.doc2bow(text) for text in coment_col]


def make_tfidf(bow_corpus: list) -> list:
    """
    Generates a TF-IDF representation of the textual corpus based on the BoW input.
    """
    tfidf_model = TfidfModel(bow_corpus)

    return tfidf_model[bow_corpus]


def make_esparse_matrix(tfidf_corpus: list, gensim_dict: Dictionary):
    """
    Converts the TF-IDF corpus directly into a SciPy Sparse Matrix for 
    compatibility with NMF model.
    """
    return corpus2csc(tfidf_corpus, len(gensim_dict)).T

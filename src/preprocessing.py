"""
Data Preprocessing Module.

This module is responsible for cleaning and preparing student comments text 
by normalizing strings, removing punctuation, handling stop words (including 
teacher's names), and applying lemmatization.
"""

# Third-Party Imports
import re
import unicodedata
from string import punctuation

import nltk
import pandas as pd
import spacy
from nltk import tokenize

# Initialize Portuguese natural language processing model
nlp = spacy.load("pt_core_news_sm")

# Setup NLTK stopwords
nltk.download('stopwords', quiet=True)
stop_words = nltk.corpus.stopwords.words('portuguese')
stop_words.extend(['nao', 'nenhum', 'ha', 'nenhuma'])


def normalize_string(text: str) -> str:
    """
    Strips accents and converts text to lowercase for string comparison.
    """
    if not isinstance(text, str):
        return ""
    # Remove accents (e.g. 'Cássio' -> 'Cassio')
    text = ''.join(c for c in unicodedata.normalize('NFD', text) if unicodedata.category(c) != 'Mn')
    return text.lower().strip()


def punct_and_number_remove(df_column: pd.Series) -> list:
    """
    Removes all punctuation and numeric characters from a column of text.
    """
    df_column = df_column.fillna('').astype(str)

    new_column_punct = []
    token_punct = tokenize.WordPunctTokenizer()

    for comment in df_column:
        tokens = token_punct.tokenize(comment)
        new_comment = []
        for token in tokens:
            if token not in punctuation:
                token = re.sub(r'\d', '', token)
                new_comment.append(token.lower())
        new_column_punct.append(' '.join(new_comment))

    return new_column_punct


def stopwords_docentes(teacher_names: list) -> list:
    """
    Creates a dedicated stopwords list using the names of the teachers 
    (tokenizing full names).
    """
    teacher_stopwords_set = set()

    for name in teacher_names:
        if not name or name == 'nan':
            continue

        # Normalize the teacher's name
        normalized_name = normalize_string(name)

        # Add ONLY the individual parts of the name (not the full name string)
        name_parts = normalized_name.split()
        for part in name_parts:
            if len(part) > 2:  # Avoid prepositions like "da", "de"
                teacher_stopwords_set.add(part)

    return sorted(list(teacher_stopwords_set))


def stopwords_remove(df_column: pd.Series, teacher_names: list | None = None) -> list:
    """
    Removes standard Portuguese stopwords and optionally teacher names 
    from a column of text.
    
    Args:
        df_column: DataFrame column containing text comments.
        teacher_names: Optional list of teacher names to append to stopwords.
    """
    # Instantiate the tokenizer
    space_tokenizer = tokenize.WhitespaceTokenizer()

    # Combine standard stopwords with teacher's name stopwords
    all_stopwords = stop_words.copy()
    if teacher_names:
        dynamic_teacher_stopwords = stopwords_docentes(teacher_names)
        all_stopwords.extend(dynamic_teacher_stopwords)

    # Drop duplicates
    all_stopwords = list(set(all_stopwords))

    comments_without_stopwords = []

    for comment in df_column:
        comment = comment.lower()
        tokens = space_tokenizer.tokenize(comment)
        new_comment = [token for token in tokens if token not in all_stopwords]
        comments_without_stopwords.append(' '.join(new_comment))

    return comments_without_stopwords


def remove_docentes_stopwords(df_column: pd.Series, teacher_names: list) -> list:
    """
    Removes ONLY the teacher names (as stopwords) from a column of text.
        
    Args:
        df_column: DataFrame column containing text comments.
        teacher_names: List of teacher names.
    """
    space_tokenizer = tokenize.WhitespaceTokenizer()

    # Create the isolated stopwords list for teachers
    isolated_teacher_stopwords = stopwords_docentes(teacher_names)

    comments_without_teacher_names = []

    for comment in df_column:
        comment = comment.lower()
        tokens = space_tokenizer.tokenize(comment)
        new_comment = [token for token in tokens if token not in isolated_teacher_stopwords]
        comments_without_teacher_names.append(' '.join(new_comment))

    return comments_without_teacher_names


def lemmatization(df_column: pd.Series) -> list:
    """
    Applies lemmatization to the input document text.
    """
    lemmatized_comments = []

    for doc in nlp.pipe(df_column, batch_size=1000):
        # Adding .lower() to ensure all lemmas are cast downwards
        lemmas = [token.lemma_.lower() for token in doc if not token.is_punct and not token.is_space]
        lemmatized_comments.append(lemmas)

    return lemmatized_comments


def pre_processamento(df: pd.DataFrame, teacher_list: list) -> pd.DataFrame:
    """
    Primary interface to execute the full text cleaning pipeline across 
    positive, negative, and general feedback columns.
    """
    target_columns = [' COMENTARIO POSITIVO', ' COMENTARIO NEGATIVO', ' COMENTARIO GERAL']

    for col in target_columns:
        col_punct = f"{col} pontuacao"
        df[col_punct] = punct_and_number_remove(df[col])

        col_teacher_removed = f"{col} stopwords_docente"
        df[col_teacher_removed] = remove_docentes_stopwords(df[col_punct], teacher_list)

        col_general_stopwords = f"{col} stopword_geral"
        df[col_general_stopwords] = stopwords_remove(df[col_punct], teacher_list)

        col_lemmatized = f"{col} lematizacao"
        df[col_lemmatized] = lemmatization(df[col_general_stopwords])

    return df

"""
Utilities Module.

This module provides common helper functions used across the project for
file ingestion, dataset concatenation, removing duplicates and nulls, and
extracting raw topics from Gensim distribution models.
"""

# Standard Imports
from pathlib import Path

# Third-Party Imports
import pandas as pd

# Paths Setup
ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"


def read_dataset() -> list:
    """
    Iterates sequentially over predefined internal datasets (csv/xlsx),
    reads them into pandas DataFrames, standardizes the first column name, 
    and returns a list of DataFrames.
    """
    bases = ['18.2', '18.4', '19.2', '19.4', '22.2', '22.4', '23.2', '23.4', '24.2', '24.4']
    evaluations_by_year = []

    for year in bases:
        csv_path = DATA / f"avalia-20{year}.csv"
        xlsx_path = DATA / f"avalia-20{year}.xlsx"

        # Note: 24.2 uses XLSX due to parsing anomalies with CSV encodings.
        if year == '24.2':
            evaluation_df = pd.read_excel(xlsx_path)
        else:
            evaluation_df = pd.read_csv(csv_path, delimiter=';', encoding='latin1', on_bad_lines='skip')

        evaluation_df.columns.values[0] = 'ANO-PERIODO'
        evaluations_by_year.append(evaluation_df)

    return evaluations_by_year


def concat_dataset(dataset_list: list) -> pd.DataFrame:
    """
    Concatenates a list of Pandas DataFrames into a single unified DataFrame 
    ignoring the original indices.
    """
    return pd.concat(dataset_list, ignore_index=True)


def drop_null_duplicate(df: pd.DataFrame, target_column: str, values_to_remove: list) -> pd.DataFrame:
    """
    Filters rows where the value in the specified target column is present in 
    the removal list, and removes duplicates keeping only the first occurrence.
    """
    df = df[~df[target_column].isin(values_to_remove)]

    return df.drop_duplicates(subset=[target_column], keep="first").reset_index(drop=True)


def get_model_topics(model, top_n: int = 15) -> list:
    """
    Extracts the keyword lists out of the native LSA and LDA Gensim models.
    Ensures the topic IDs are naturally sorted because Gensim can scramble
    output order.
    
    Returns:
        A list of lists (List[List[str]]) containing strings in the format 
        required natively by the CoherenceModel.
    """
    # num_topics = -1 to fetch all available topics mapped in the model memory.
    # num_words = top_n establishes exact keyword length output for each topic array.
    # formatted = False secures lists of raw tuple pairs (word, probability value).
    topics_data = model.show_topics(num_topics=-1, num_words=top_n, formatted=False)

    topics_list = []

    # Sort by topic ID
    topics_data.sort(key=lambda x: x[0])

    for word_weight_list in topics_data:
        words = [word for word, weight in word_weight_list]
        topics_list.append(words)

    return topics_list

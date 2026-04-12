"""
Data Selection.

This module provides helper functions to filter the raw assessment dataset
by specific columns, campus, academic unit, course, or specific generated labels.
It also includes an interactive terminal filter function.
"""

# Thridy-Party Imports
import pandas as pd


def columns_filter(dataframe: pd.DataFrame, selected_columns: list) -> pd.DataFrame:
    """
    Filters a DataFrame to keep only the specified columns.
    """
    columns = selected_columns
    return dataframe[columns]


def filter_by_campus(df: pd.DataFrame, campus_input: str) -> pd.DataFrame:
    """
    Filters the DataFrame by the specified campus name.
    """
    return df[df['CAMPUS CENTRO DO CURSO'] == campus_input.upper()]


def filter_by_und_acad(df: pd.DataFrame, unit_input: str) -> pd.DataFrame:
    """
    Filters the DataFrame by the academic unit acronym.
    """
    return df[df['CENTRO DO CURSO SIGLA'] == unit_input.upper()]


def filter_by_curso(df: pd.DataFrame, course_input: str) -> pd.DataFrame:
    """
    Filters the DataFrame by the specific course name.
    """
    return df[df['CURSO'] == course_input.upper()]

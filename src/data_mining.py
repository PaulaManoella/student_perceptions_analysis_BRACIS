"""
Data Mining Module.

This module provides functions to initialize and run topic modeling algorithms,
including Latent Semantic Analysis (LSA), Latent Dirichlet Allocation (LDA), 
Non-Negative Matrix Factorization (NMF), and BERTopic.
"""

# Thridy-Party Imports
from typing import Any, Dict, Tuple
import numpy as np
import pandas as pd
from gensim.models import LsiModel, LdaModel
from gensim.corpora import Dictionary
from sklearn.decomposition import NMF
from bertopic import BERTopic
from gensim.models.coherencemodel import CoherenceModel
import rbo


def LSA_model(tfidf_corpus: list, vocab_dict: Dictionary, n_topics: int, seed: int) -> LsiModel:
    """
    Initializes and trains a LSA model using a TF-IDF corpus.

    Args:
        tfidf_corpus (list): The TF-IDF vectorized corpus.
        vocab_dict (Dictionary): The Gensim dictionary mapping word IDs to words.
        n_topics (int): The number of topics to extract.
        seed (int): The random seed for reproducibility.

    Returns:
        LsiModel: The trained LSA model.
    """
    return LsiModel(
        corpus=tfidf_corpus,
        id2word=vocab_dict,
        num_topics=n_topics,
        random_seed=seed
    )


def print_lsa_topics(lsa_model: LsiModel, n_words: int) -> None:
    """
    Prints the top words for each topic extracted by the LSA model.

    Args:
        lsa_model (LsiModel): The trained LSA model.
        n_words (int): The number of top words to print per topic.
    """
    print("Topics extracted by the LSA model:")
    for topic_id, topic in lsa_model.print_topics(num_words=n_words):
        print(f"Topic #{topic_id}: {topic}")


def LDA_model(corpus: list, vocab_dict: Dictionary, n_topics: int, n_passes: int, seed: int) -> LdaModel:
    """
    Initializes and trains a Latent Dirichlet Allocation (LDA) model using a Bag of Words corpus.

    Args:
        corpus (list): The Bag of Words (BoW) vectorized corpus.
        vocab_dict (Dictionary): The Gensim dictionary mapping word IDs to words.
        n_topics (int): The number of topics to extract.
        n_passes (int): Number of passes through the corpus during training.
        seed (int): The random seed for reproducibility.

    Returns:
        LdaModel: The trained LDA model.
    """
    return LdaModel(
        corpus=corpus,
        id2word=vocab_dict,
        num_topics=n_topics,
        passes=n_passes,
        random_state=seed
    )


def NMF_model(n_topics: int, tfidf_sparse: Any, seed: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Initializes and trains a Non-Negative Matrix Factorization (NMF) model.

    Args:
        n_topics (int): The number of topics (components) to extract.
        tfidf_sparse (Any): The sparse TF-IDF document-term matrix.
        seed (int): The random seed for reproducibility.

    Returns:
        Tuple[np.ndarray, np.ndarray]: A tuple containing:
            - W (np.ndarray): The document-topic matrix.
            - H (np.ndarray): The topic-term matrix (components).
    """
    nmf_engine = NMF(
        n_components=n_topics,
        random_state=seed,
        init='nndsvd',       
        solver='cd',
        max_iter=500,
        tol=0.01,
        verbose=0
    )
    W = nmf_engine.fit_transform(tfidf_sparse)
    H = nmf_engine.components_

    return W, H


def BERTopic_model(df_col: pd.Series, params_dict: Dict[str, Any]) -> Tuple[list, np.ndarray, BERTopic, list]:
    """
    Initializes and trains a BERTopic neural topic model, detecting and reducing outliers.

    Args:
        df_col (pd.Series): The pandas Series containing the text documents.
        params_dict (Dict[str, Any]): A dictionary of parameters to initialize BERTopic.

    Returns:
        Tuple[list, np.ndarray, BERTopic, list]: A tuple containing:
            - topics (list): The initial topics assigned to documents.
            - probs (np.ndarray): The probabilities of document assignments to topics.
            - bertopic_model (BERTopic): The trained BERTopic model object.
            - new_topics (list): The updated topics after outlier reduction.
    """
    bertopic_model = BERTopic(**params_dict)
    
    topics, probs = bertopic_model.fit_transform(df_col)

    # Reduce topic outliers using embedding distribution strategies
    new_topics = bertopic_model.reduce_outliers(
        df_col, topics, strategy="embeddings"
    )
            
    # Update the topic representations after assigning outliers to topics
    bertopic_model.update_topics(
        df_col,
        topics=new_topics,
        vectorizer_model=params_dict['vectorizer_model'],
        top_n_words=params_dict.get('top_n_words', 10)
    )
    
    return topics, probs, bertopic_model, new_topics


def coherence_score(topic_words, df_col_token, model_dict):
    
    coherence_model = CoherenceModel(
            topics=topic_words,
            texts=df_col_token,
            dictionary=model_dict,
            coherence='c_v'
    )
    
    return coherence_model.get_coherence()


def calculate_inter_model_diversity(topics_model_a: list, topics_model_b: list) -> float:
    """
    Calculates the average diversity (IRBO) between two models.
    The higher the value (closer to 1), the more different the models are.
    """
    if not topics_model_a or not topics_model_b:
        return 0.0  # Prevents error if list is empty

    rbo_scores = []
    
    # For each topic in A, find the "twin" (best match) in B
    for i, topic_a in enumerate(topics_model_a):
        best_match_score = 0
        
        for j, topic_b in enumerate(topics_model_b):
            # Ignore self-comparison if the same identical list of topics is passed (Intra-Model Diversity)
            if topics_model_a is topics_model_b and i == j:
                continue

            # RBO p=0.9: High importance for the top of the ranking
            score = rbo.RankingSimilarity(topic_a, topic_b).rbo(p=0.9)
            if score > best_match_score:
                best_match_score = score
        
        rbo_scores.append(best_match_score)
    
    # Average similarity of the best matches
    avg_similarity = np.mean(rbo_scores)
    
    # IRBO = 1 - Similarity (Transforms into a "Difference" metric)
    diversity_irbo = 1 - avg_similarity
    return diversity_irbo

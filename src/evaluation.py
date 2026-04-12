"""
Data Evaluation and Visualization Module.

This module provides tools for evaluating topic models, generating vector embeddings,
computing semantic similarity matrices across diverse topics, detecting communities
using the Louvain algorithm, and plotting inter-model projections using UMAP.
"""

# Thridy-Party Imports
import community.community_louvain as community_louvain
import networkx as nx
import numpy as np
import pandas as pd
import rbo
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
from gensim.models.coherencemodel import CoherenceModel

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


def generate_topic_vector(topic_keywords: list, model_emb) -> np.ndarray:
    """
    Receives a list of topic keywords and returns the mean vector (centroid) 
    of that topic using the given embedding model.
    """
    if not topic_keywords:
        return np.zeros(model_emb.get_sentence_embedding_dimension())

    word_embeddings = model_emb.encode(topic_keywords)
    return np.mean(word_embeddings, axis=0)


def generate_cosine_matrix(topics_vectors: dict) -> tuple:
    """
    Constructs a global cosine similarity matrix comparing vectors of all models.
    """
    global_vectors = []
    global_labels = []

    # Unpack the dictionary containing all topic vectors
    for model_name, vectors in topics_vectors.items():
        for i, vec in enumerate(vectors):
            global_vectors.append(vec)
            # Create labels to identify the model and topic. ex: 'LDA T0'
            global_labels.append(f"{model_name.upper()}_{i}")

    # Convert the vector list into a numpy matrix (topics as rows, embedding dimensions as columns)
    x_global = np.array(global_vectors)

    # Print matrix structure
    print(f"✅ Global Matrix Created: {x_global.shape} (Topics x Dimensions)")

    # Calculate similarity
    sim_matrix_global = cosine_similarity(x_global)

    # Transform into DataFrame for better structured data layout
    df_cos_sim = pd.DataFrame(
        sim_matrix_global,
        index=global_labels,
        columns=global_labels
    )

    return df_cos_sim, sim_matrix_global


def generate_graph(labels: list, sim_matrix_global: np.ndarray, threshold: float) -> nx.Graph:
    """
    Builds a similarity network graph connecting topics based on a minimum similarity threshold.
    """
    G = nx.Graph()

    # 1. Build the Graph (Nodes and Edges)
    for i, label in enumerate(labels):
        model_name = label.split('_')[0]
        G.add_node(i, label=label, model=model_name)

    rows, cols = sim_matrix_global.shape
    for i in range(rows):
        for j in range(i + 1, cols):
            sim_score = sim_matrix_global[i, j]
            if sim_score >= threshold:
                G.add_edge(i, j, weight=sim_score)

    print(f"Graph initialized with {G.number_of_nodes()} nodes and {G.number_of_edges()} strong connections (threshold >= {threshold}).")
    return G


def generate_communities(labels: list, sim_matrix_global: np.ndarray, threshold: float = 0.15, min_models_common: int = 3) -> tuple:
    """
    Builds a topic similarity graph and detects communities, 
    segregating them into Common Topics and Exclusive Topics.
    """
    # 1. Graph Generation
    G = generate_graph(labels, sim_matrix_global, threshold)

    # 2. Community Detection (Louvain)
    partition = community_louvain.best_partition(G, weight='weight', random_state=42)

    # Organize results into a DataFrame
    results = []
    for node_id, community_id in partition.items():
        results.append({
            "Node_ID": node_id,
            "Community_ID": community_id,
            "Topic_Label": G.nodes[node_id]['label'],
            "Model": G.nodes[node_id]['model']
        })
    df_communities = pd.DataFrame(results)

    # 3. Cluster Analysis (Common vs Exclusive)
    # Count how many unique models exist within each community
    comm_composition = df_communities.groupby("Community_ID")['Model'].nunique()

    print("\n" + "="*50)
    print("🎯 1. COMMON TOPICS (SUPER-TOPICS)")
    print("="*50)
    common_comms = comm_composition[comm_composition >= min_models_common].index

    if len(common_comms) == 0:
        print(f"No super-topic found with at least {min_models_common} models.")
    else:
        for comm_id in common_comms:
            group = df_communities[df_communities['Community_ID'] == comm_id]
            models_present = group['Model'].unique()
            print(f"\nSuper-Topic [Community {comm_id}]:")
            print(f"  Present models: {', '.join(models_present)}")
            print(f"  Topics: {group['Topic_Label'].values}")

    print("\n" + "="*50)
    print("🔒 2. EXCLUSIVE TOPICS (MONO-MODEL)")
    print("="*50)
    exclusive_comms = comm_composition[comm_composition == 1].index

    if len(exclusive_comms) == 0:
        print("No exclusive community found.")
    else:
        for comm_id in exclusive_comms:
            group = df_communities[df_communities['Community_ID'] == comm_id]
            model_name = group['Model'].iloc[0]
            topics = group['Topic_Label'].values
            print(f"Community {comm_id} is EXCLUSIVE to model: {model_name}")
            print(f"  Topics: {', '.join(topics)}")

    return df_communities, G


def plot_metric_across_models(
    x_values: list,
    models_metrics: dict,
    title: str,
    xlabel: str,
    ylabel: str,
    legend_title: str = "Model",
    save_path: str = None
) -> None:
    """
    Plots a generic line chart to compare metric scores across different models.
    Can be used for both Coherence Score (C_v) and Topic Diversity (IRBO).
    
    Args:
        x_values (list): Values for the x-axis (e.g., range of topics).
        models_metrics (dict): Dictionary mapping model name to their respective metric scores list.
                               Example: {'LSA': [0.6, 0.5, ...], 'LDA': [0.9, ...]}
        title (str): The plot title.
        xlabel (str): Label for the x-axis.
        ylabel (str): Label for the y-axis.
        legend_title (str): Title for the legend.
        save_path (str, optional): If provided, saves the figure to the specified path.
    """
    plt.figure(figsize=(14, 7))
    
    for model_name, metrics in models_metrics.items():
        plt.plot(x_values, metrics, marker='o', markersize=4, label=model_name)
    
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    
    # Ensure ticks match x_values
    plt.xticks(x_values)
    
    plt.grid(True, linestyle='-', alpha=0.4, color='lightgrey')
    plt.legend(title=legend_title, loc='lower right')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        print(f"Plot saved to '{save_path}'")
        
    plt.show()
    plt.close()

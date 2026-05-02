"""
Data Evaluation and Visualization Module.

This module provides tools for evaluating topic models, generating vector embeddings,
computing semantic similarity matrices across diverse topics, detecting communities
using the Louvain algorithm, and plotting inter-model projections using UMAP.
"""

# Third-Party Imports
import community.community_louvain as community_louvain
import networkx as nx
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import umap
from sklearn.metrics.pairwise import cosine_similarity


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
    Plots a line chart to compare metric scores across different models.
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


def plot_umap_projection(
    all_topic_vectors: dict,
    save_path: str = "umap_projection.pdf",
    n_neighbors: int = 15,
    min_dist: float = 0.2,
    random_state: int = 42
) -> None:
    """
    Projects all topic vectors from multiple models into a 2D space using UMAP
    and generates a scatter plot for inter-model visual comparison.

    Args:
        all_topic_vectors (dict): Dictionary mapping model names (str) to lists
                                  of topic embedding vectors (np.ndarray).
                                  Example: {'lda': [vec0, vec1, ...], 'nmf': [...]}
        save_path (str): File path to save the resulting plot.
        n_neighbors (int): UMAP parameter controlling the balance between local
                           and global structure preservation.
        min_dist (float): UMAP parameter controlling point spread in the embedding.
        random_state (int): Random seed for reproducibility.
    """
    # Unpack all topic vectors into flat lists
    all_vectors = []
    all_labels = []   # Model name (e.g., 'LDA', 'BERTopic')
    all_indices = []  # Topic ID (e.g., 'T0', 'T1')

    for model_name, vectors in all_topic_vectors.items():
        for i, vec in enumerate(vectors):
            all_vectors.append(vec)
            all_labels.append(model_name.upper())
            all_indices.append(f"T{i}")

    # Build input matrix (topics x embedding dimensions)
    X = np.array(all_vectors)
    print(f"UMAP input matrix: {X.shape} (Topics x Dimensions)")

    # Fit UMAP projection
    umap_model = umap.UMAP(
        n_components=2,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        metric="cosine",
        random_state=random_state,
        init="spectral"
    )
    X_embedded = umap_model.fit_transform(X)

    # Assemble DataFrame for plotting
    df_plot = pd.DataFrame({
        "x": X_embedded[:, 0],
        "y": X_embedded[:, 1],
        "Model": all_labels,
        "Topic": all_indices
    })

    # Plot
    plt.figure(figsize=(14, 9))
    sns.set_style("whitegrid")

    palette = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]
    sns.scatterplot(
        data=df_plot,
        x="x",
        y="y",
        hue="Model",
        style="Model",
        s=180,
        alpha=0.8,
        palette=palette
    )

    # Annotate each point with its topic ID
    for i in range(df_plot.shape[0]):
        plt.text(
            df_plot.x.iloc[i] + 0.04,
            df_plot.y.iloc[i] + 0.04,
            df_plot.Topic.iloc[i],
            fontsize=11,
            alpha=0.9
        )

    plt.title("UMAP Projection of Inter-Model Topics", fontsize=16)
    plt.xlabel("Dimension 1", fontsize=12)
    plt.ylabel("Dimension 2", fontsize=12)
    plt.legend(
        bbox_to_anchor=(1, 1.011),
        loc="upper left",
        title="Model",
        title_fontsize=11
    )

    plt.savefig(save_path, dpi=600, bbox_inches="tight")
    print(f"UMAP plot saved to '{save_path}'")
    plt.show()
    plt.close()


def plot_super_topics_network(
    df_communities: pd.DataFrame,
    graph: nx.Graph,
    min_models_common: int = 3,
    save_path: str = "super_topics_network.pdf"
) -> None:
    """
    Visualizes the common topic communities (super-topics) as side-by-side
    network graphs, where each subplot represents one super-topic.

    Nodes are colored by their originating model, and edges represent
    above-threshold cosine similarity connections.

    Args:
        df_communities (pd.DataFrame): DataFrame returned by `generate_communities`,
                                       containing columns: Node_ID, Community_ID,
                                       Topic_Label, Model.
        graph (nx.Graph): The similarity network graph returned by `generate_communities`.
        min_models_common (int): Minimum number of distinct models required for a
                                 community to be classified as a common super-topic.
        save_path (str): File path to save the resulting plot.
    """
    # Define a consistent color palette per model
    model_colors = {
        "LSA": "#1f77b4",       # Blue
        "LDA": "#ff7f0e",       # Orange
        "NMF": "#2ca02c",       # Green
        "BERTopic": "#e377c2",  # Pink
    }

    # Identify common communities (super-topics)
    comm_composition = df_communities.groupby("Community_ID")["Model"].nunique()
    common_comm_ids = sorted(
        comm_composition[comm_composition >= min_models_common].index
    )

    if len(common_comm_ids) == 0:
        print(f"No super-topic found with at least {min_models_common} models. Skipping plot.")
        return

    n_plots = len(common_comm_ids)
    fig, axes = plt.subplots(1, n_plots, figsize=(8 * n_plots, 9))

    # Ensure axes is always iterable (even for a single subplot)
    if n_plots == 1:
        axes = [axes]

    for ax, comm_id in zip(axes, common_comm_ids):
        # Filter nodes belonging to this community
        group = df_communities[df_communities["Community_ID"] == comm_id]
        node_ids = group["Node_ID"].tolist()

        # Extract the subgraph for this community
        subgraph = graph.subgraph(node_ids).copy()

        # Build node colors and labels from the subgraph
        node_colors = []
        node_labels = {}
        for node_id in subgraph.nodes():
            row = group[group["Node_ID"] == node_id].iloc[0]
            model_name = row["Model"]
            node_colors.append(model_colors.get(model_name, "#999999"))
            node_labels[node_id] = row["Topic_Label"]

        # Compute layout
        pos = nx.spring_layout(subgraph, seed=42, k=0.5, iterations=60)

        # Draw edges
        nx.draw_networkx_edges(
            subgraph, pos, ax=ax,
            edge_color="#cccccc", alpha=0.6, width=1.2
        )

        # Draw nodes
        nx.draw_networkx_nodes(
            subgraph, pos, ax=ax,
            node_color=node_colors,
            node_size=600,
            alpha=0.85,
            edgecolors="white",
            linewidths=0.8
        )

        # Draw labels
        nx.draw_networkx_labels(
            subgraph, pos, labels=node_labels, ax=ax,
            font_size=8, font_weight="bold"
        )

        ax.set_title(
            f"Super-Topic {comm_id}: {len(node_ids)} topics",
            fontsize=14, fontweight="bold"
        )
        ax.axis("off")

    plt.tight_layout(rect=[0, 0.05, 1, 1])
    plt.savefig(save_path, dpi=600, bbox_inches="tight")
    print(f"Super-topics network plot saved to '{save_path}'")
    plt.show()
    plt.close()


def get_dominant_topics(model, corpus: list, model_type: str = "gensim") -> list:
    """
    Determines the dominant (most representative) topic for each document.

    Args:
        model: The trained model. For 'gensim' (LSA/LDA), a Gensim model.
               For 'nmf', the W (document-topic) matrix (np.ndarray).
        corpus (list): The vectorized corpus (TF-IDF for LSA, BoW for LDA).
                       Ignored when model_type='nmf'.
        model_type (str): One of 'gensim' (for LSA/LDA) or 'nmf'.

    Returns:
        list: A list of dominant topic IDs (int), one per document.
    """
    if model_type == "nmf":
        # model is the W matrix (documents x topics)
        return model.argmax(axis=1).tolist()

    # Gensim models (LSA / LDA)
    dominant_topics = []
    for doc in corpus:
        topic_dist = model[doc]
        if topic_dist:
            # For LSA use absolute values (can be negative), for LDA use raw probabilities
            dominant = max(topic_dist, key=lambda x: abs(x[1]))[0]
        else:
            dominant = -1
        dominant_topics.append(dominant)

    return dominant_topics


def generate_top_n_frequency_table(
    model_topic_data: dict,
    top_n: int = 3,
    save_path: str = None
) -> pd.DataFrame:
    """
    Generates a consolidated table showing the top-N most frequent topics
    per model, including frequency percentages and keywords.

    Args:
        model_topic_data (dict): Dictionary mapping model names to a dict with:
            - 'assignments' (list[int]): Dominant topic ID per document.
            - 'keywords' (pd.DataFrame): DataFrame with a 'keywords' column
                                          (list of words), indexed by topic ID.
            Example:
                {
                    'LSA': {'assignments': [0, 2, 0, ...], 'keywords': df_lsa_topics},
                    'LDA': {'assignments': [1, 1, 3, ...], 'keywords': df_lda_topics},
                }
        top_n (int): Number of top topics to return per model.
        save_path (str, optional): If provided, saves the table as CSV.

    Returns:
        pd.DataFrame: Table with columns [Model, Freq_Pct, Topic_ID, Keywords].
    """
    rows = []

    for model_name, data in model_topic_data.items():
        assignments = data["assignments"]
        df_keywords = data["keywords"]
        total_docs = len(assignments)

        # Count documents per topic
        topic_counts = pd.Series(assignments).value_counts()

        # Select top-N topics by frequency
        top_topics = topic_counts.head(top_n)

        for topic_id, count in top_topics.items():
            freq_pct = round((count / total_docs) * 100, 1)

            # Retrieve keywords for this topic
            if topic_id >= 0 and topic_id < len(df_keywords):
                kw_list = df_keywords.iloc[topic_id]["keywords"]
                if isinstance(kw_list, list):
                    keywords_str = ", ".join(kw_list[:8])  # Show top 8 keywords
                else:
                    keywords_str = str(kw_list)
            else:
                keywords_str = "N/A"

            rows.append({
                "Model": model_name,
                "Freq_Pct": freq_pct,
                "Topic_ID": topic_id,
                "Keywords": keywords_str
            })

    df_table = pd.DataFrame(rows)

    if save_path:
        df_table.to_csv(save_path, index=False, encoding="utf-8-sig")
        print(f"\nTable saved to '{save_path}'")

    return df_table

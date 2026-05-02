"""
Inter-model Alignment Pipeline Script.

This script processes the generated topics from four topic modeling algorithms
and calculates their semantic similarity using a Portuguese sentence-transformer
model. It then constructs a similarity network and extracts topic communities
across models using the Louvain algorithm
"""


# Third-party Imports
import pandas as pd
from sentence_transformers import SentenceTransformer

# Local Imports
from src import evaluation

# -----------------------------------------------------------------------------
# Configuration & Constants
# -----------------------------------------------------------------------------
SIMILARITY_THRESHOLD = 0.94
TOPIC_KEYWORD_COLUMN = 'keywords'
EMBEDDING_MODEL_NAME = "PORTULAN/serafim-335m-portuguese-pt-sentence-encoder-ir"


def run_alignment() -> None:
    """
    Main function executing the inter-model topic alignment process.
    """
    # Load generated topics from all models
    df_lsa_topics = pd.read_pickle('src/data/df_lsa_topics.pkl')
    df_lda_topics = pd.read_pickle('src/data/df_lda_topics.pkl')
    df_nmf_topics = pd.read_pickle('src/data/df_nmf_topics.pkl')
    df_bertopic_topics = pd.read_pickle('src/data/df_bertopic_topics.pkl')

    # Group topics by their respective models for embeddings generation
    topics_by_model = {
        "lsa": df_lsa_topics,
        "lda": df_lda_topics,
        "nmf": df_nmf_topics,
        "bertopic": df_bertopic_topics,
    }

    # Dictionary to store vector embeddings for all topics across all models
    all_topic_vectors = {}

    # Initialize the sentence embedding model
    embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)

    # Generate vector embeddings for all topics across all models
    for model_name, df_topics in topics_by_model.items():
        vectors = []
        for keywords in df_topics[TOPIC_KEYWORD_COLUMN]:
            vec = evaluation.generate_topic_vector(keywords, embedding_model)
            vectors.append(vec)
        all_topic_vectors[model_name] = vectors

    # Generate a cosine similarity matrix encompassing all valid model combinations
    df_cosine_similarity, cosine_matrix = evaluation.generate_cosine_matrix(all_topic_vectors)

    # Generate labels for the similarity matrix matching the total number of topics
    labels = []
    labels += [f"LSA_Topic_{i}" for i in range(25)]      # Indices 0 to 24
    labels += [f"LDA_Topic_{i}" for i in range(13)]      # Indices 25 to 37
    labels += [f"NMF_Topic_{i}" for i in range(13)]      # Indices 38 to 50
    labels += [f"BERTopic_Topic_{i}" for i in range(14)] # Indices 51 to 64

    # Apply Louvain community detection to find commom topics across models and exclusive topics
    df_communities, network_graph = evaluation.generate_communities(
        labels,
        cosine_matrix,
        threshold=SIMILARITY_THRESHOLD,
        min_models_common=4
    )

    print("\n✅ Inter-model alignment execution completed successfully!\n")

    print(f"\n --- Communities Detected --- \n{df_communities}")

    # Generate UMAP 2D projection of all topic vectors across models
    evaluation.plot_umap_projection(
        all_topic_vectors,
        save_path="output/figure/umap_projection.pdf"
    )

    # Plot the common topic communities (super-topics) as network graphs
    evaluation.plot_super_topics_network(
        df_communities,
        network_graph,
        min_models_common=4,
        save_path="output/figure/super_topics.pdf"
    )

    # ----- Generate Top-3 Most Frequent Topics Table -----
    # Load document-to-topic assignments generated during modeling (main.py)
    assignments_by_model = pd.read_pickle('src/data/all_topic_assignments.pkl')

    # Map keyword DataFrames to their corresponding model names
    keywords_by_model = {
        "LSA": df_lsa_topics,
        "LDA": df_lda_topics,
        "NMF": df_nmf_topics,
        "BERTopic": df_bertopic_topics,
    }

    # Reconstruct the full data structure expected by the frequency table function
    model_topic_data = {
        model: {
            "assignments": assignments_by_model[model],
            "keywords": keywords_by_model[model]
        }
        for model in assignments_by_model
    }

    evaluation.generate_top_n_frequency_table(
        model_topic_data,
        top_n=3,
        save_path="output/topic_modeling/top3_topics_frequency.csv"
    )


if __name__ == "__main__":
    run_alignment()

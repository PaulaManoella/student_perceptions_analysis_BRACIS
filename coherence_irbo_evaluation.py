"""
Defining k topics .

This script processes the generated topics from four topic modeling algorithms
and calculates their semantic similarity using a Portuguese sentence-transformer
model. It then constructs a similarity network and extracts topic communities
across models using the Louvain algorithm
"""

# Thridy-Party Imports
import nltk
import pandas as pd
from bertopic.representation import MaximalMarginalRelevance
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import CountVectorizer
from umap import UMAP
from gensim.corpora import Dictionary

# Load Imports
from main import load_and_filter_data
from src import data_mining as data_mining
from src import preprocessing as pre_processing
from src import selection as selection
from src import transformation as transformation
from src import evaluation as evaluation
from src import utils


# -----------------------------------------------------------------------------
# Configuration & Constants
# -----------------------------------------------------------------------------
RANDOM_SEED = 2025

# Setup NLTK
nltk.download('stopwords', quiet=True)
nltk.download('punkt_tab', quiet=True)
portuguese_stopwords = nltk.corpus.stopwords.words('portuguese')

# Domain-specific stopwords
academic_stopwords = [
    'nenhum', 'ha', 'ser', 'ter', 'fazer', 'ir', 'estar',
    'pode', 'pra', 'tava', 'so', 'porque', 'entao', 'la', 'aqui',
    'professor', 'professora', 'professores', 'docente',
    'aluno', 'alunos', 'discente', 'estudante', 'estudantes',
    'disciplina', 'materia', 'universidade', 'ufpa', 'curso', 'turma',
    'aulas', 'semestre', 'periodo', 'de', 'nao', 'que', 'em', 'da', 'com',
    'para', 'do', 'os', 'um', 'sempre', 'uma', 'o', 'a', 'e', 'ao', 'dos',
    'das', 'no', 'na', 'nos', 'nas', 'se', 'eu', 'me', 'meu', 'minha',
    'seu', 'sua', 'esse', 'essa', 'aula', 'pois', 'forma'
]

custom_stopwords = list(set(portuguese_stopwords))
custom_stopwords.extend(academic_stopwords)

# BERTopic hyperparameters configuration
vectorizer_model = CountVectorizer(stop_words=custom_stopwords, min_df=1)
embedding_model = SentenceTransformer("PORTULAN/serafim-335m-portuguese-pt-sentence-encoder-ir")
umap_model = UMAP(min_dist=0.0, metric="cosine", random_state=RANDOM_SEED)
representation_model = MaximalMarginalRelevance(diversity=0.4)

def load_and_filter_data() -> pd.DataFrame:
    """
    Loads the dataset and applies filters to select the campus, academic unit
    and course target.

    Returns:
        pd.DataFrame: Filtered dataset for the specific unit/course.
    """
    # Load raw dataset
    df_raw = utils.concat_dataset(utils.read_dataset())

    # Select relevant columns
    target_columns = [
        'ANO-PERIODO', 'CENTRO DO CURSO', 'CAMPUS CENTRO DO CURSO',
        'CENTRO DO CURSO SIGLA', 'CURSO', 'DOCENTE', 'COMPONENTE',
        ' COMENTARIO POSITIVO', ' COMENTARIO NEGATIVO', ' COMENTARIO GERAL'
    ]
    df_selected = selection.columns_filter(df_raw, target_columns)

    # Filter by campus, academic unit and course
    df_campus = selection.filter_by_campus(df_selected, 'belem')
    df_academic_unit = selection.filter_by_und_acad(df_campus, 'icen')
    df_course = selection.filter_by_curso(df_academic_unit, 'sistemas de informacao')

    return df_course

def run_pipeline() -> None:
    """
    Main function executing the complete data processing and topic modeling pipeline.
    """
    # Load the filtered dataset
    df_course = load_and_filter_data()

    # Get a list of unique teachers name to remove from comments
    teacher_names_list = df_course['DOCENTE'].unique().tolist()

    # Pre-process the data
    df_preprocessed = pre_processing.pre_processamento(df_course, teacher_names_list)

    # Merge comment columns for classic topic modeling algorithms LSA, LDA and NMF
    df_comments_lemmatized = transformation.merge_comentario_columns(
        df_preprocessed,
        [
            ' COMENTARIO POSITIVO lematizacao',
            ' COMENTARIO NEGATIVO lematizacao',
            ' COMENTARIO GERAL lematizacao'
        ]
    )

    # Merge comment columns for neural topic modeling algorithm BERTopic
    df_comments_bertopic = transformation.merge_comentario_columns(
        df_preprocessed,
        [
            ' COMENTARIO POSITIVO stopwords_docente',
            ' COMENTARIO NEGATIVO stopwords_docente',
            ' COMENTARIO GERAL stopwords_docente'
        ]
    )

    # Define values to be removed from comments
    null_values = ['nan', '']

    # Drop null or duplicated values in the 'comentario' column
    df_lemmatized = utils.drop_null_duplicate(df_comments_lemmatized, 'comentario', null_values)
    df_bertopic = utils.drop_null_duplicate(df_comments_bertopic, 'comentario', null_values)

    # Generate Dictionary, Bag of Words (BoW), and TF-IDF (sparse and non-sparse) matrix for classic models
    lemmatized_dict = transformation.make_dict(df_lemmatized["comentario"])
    bow_corpus = transformation.make_corpus(lemmatized_dict, df_lemmatized["comentario"])
    tfidf_matrix = transformation.make_tfidf(bow_corpus)
    tfidf_sparse = transformation.make_esparse_matrix(tfidf_matrix, lemmatized_dict)

    # Start topic modeling with chosen baseline models
    models_to_run = ['lsa', 'lda', 'nmf', 'bertopic']
    range_n_topics = range(2, 26)

    # Dictionaries to store metrics
    all_coherence_scores = {model: [] for model in models_to_run}
    all_irbo_scores = {model: [] for model in models_to_run}

    # Texts for coherence (LSA, LDA, NMF usually use lemmatized texts list of lists)
    lemmatized_texts = df_lemmatized["comentario"].tolist()
    
    # Texts for BERTopic coherence (tokenize df_bertopic)
    bertopic_texts = [str(text).split() for text in df_bertopic["comentario"].tolist()]
    bertopic_dict = Dictionary(bertopic_texts)

    for model_name in models_to_run:
        print(f"\n=== MODEL {model_name.upper()} ===")

        for topic_num in range_n_topics:
            model_topics = []
            
            if model_name == 'lsa':
                lsa_model = data_mining.LSA_model(tfidf_matrix, lemmatized_dict, topic_num, RANDOM_SEED)
                model_topics = utils.get_model_topics(lsa_model)

            elif model_name == 'lda':
                lda_model = data_mining.LDA_model(
                    corpus=bow_corpus,
                    vocab_dict=lemmatized_dict,
                    n_topics=topic_num,
                    n_passes=30,
                    seed=RANDOM_SEED
                )
                model_topics = utils.get_model_topics(lda_model)

            elif model_name == 'nmf':
                _, h_matrix = data_mining.NMF_model(topic_num, tfidf_sparse, RANDOM_SEED)
                feature_names = list(lemmatized_dict.values())
                for topic_weights in h_matrix:
                    top_indices = topic_weights.argsort()[:-15 - 1:-1]
                    model_topics.append([feature_names[i] for i in top_indices])

            elif model_name == 'bertopic':
                umap_iter = UMAP(min_dist=0.0, metric="cosine", random_state=RANDOM_SEED)
                bertopic_params = {
                    "vectorizer_model": vectorizer_model,
                    "representation_model": representation_model,
                    "umap_model": umap_iter,
                    "language": "portuguese",
                    "min_topic_size": 2,
                    "nr_topics": topic_num,
                    "verbose": False,
                    "embedding_model": embedding_model,
                    "calculate_probabilities": False,
                    "top_n_words": 15,
                }
                topics, probs, b_model, new_topics = data_mining.BERTopic_model(
                    df_bertopic["comentario"], bertopic_params
                )
                topic_word_dict = b_model.get_topics()
                for topic_id in sorted(topic_word_dict.keys()):
                    if topic_id == -1:
                        continue
                    t_words = [w for w, _ in topic_word_dict[topic_id]]
                    if len(t_words) >= 2:
                        model_topics.append(t_words)

            # Evaluate Coherence
            if model_name == 'bertopic':
                co_score = data_mining.coherence_score(model_topics, bertopic_texts, bertopic_dict)
            else:
                co_score = data_mining.coherence_score(model_topics, lemmatized_texts, lemmatized_dict)
            all_coherence_scores[model_name].append(co_score)

            # Evaluate Diversity
            div_score = data_mining.calculate_inter_model_diversity(model_topics, model_topics)
            all_irbo_scores[model_name].append(div_score)

            print(f"[{model_name.upper()}] Topics: {topic_num} -> Coherence: {co_score:.4f} | IRBO: {div_score:.4f}")

        print(f"✅ {model_name.upper()} completed!\n")

    # Generate and save plots
    print("Generating Coherence / IRBO plots...")
    
    evaluation.plot_metric_across_models(
        x_values=list(range_n_topics),
        models_metrics=all_coherence_scores,
        title="Coherence (Cv) score per model and topics number",
        xlabel="Topics number",
        ylabel="Coherence (Cv) score",
        legend_title="Model",
        save_path="output/figure/coherence_plot.pdf"
    )

    evaluation.plot_metric_across_models(
        x_values=list(range_n_topics),
        models_metrics=all_irbo_scores,
        title="Topic diversity (IRBO) score per model and topics number",
        xlabel="Topics number",
        ylabel="Topic diversity (IRBO) score",
        legend_title="Model",
        save_path="output/figure/irbo_plot.pdf"
    )

    print("✅ Pipeline Completed!")


if __name__ == "__main__":
    run_pipeline()

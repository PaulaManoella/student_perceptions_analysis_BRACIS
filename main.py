"""
Main pipeline script for topic modeling analysis of student feedback.

This script performs data preprocessing, cleaning, transformation and
applies several topic modeling algorithms (LSA, NMF, BERTopic) to qualitative
feedback from students. It is designed to be highly reproducible and anonymized
for public distribution.
"""

# Thridy-Party Imports
import nltk
import pandas as pd
from bertopic.representation import MaximalMarginalRelevance
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import CountVectorizer
from umap import UMAP

from src import data_mining as data_mining
from src import evaluation as evaluation
from src import preprocessing as pre_processing
from src import selection as selection
from src import transformation as transformation

# Local Imports
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

    return selection.filter_by_curso(df_academic_unit, 'sistemas de informacao')


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
    all_models_results = {}
    all_topic_assignments = {}  # Stores dominant topic per document for each model

    for model_name in models_to_run:
        print(f"\n=== MODEL {model_name.upper()} ===")

        # Start LSA model
        if model_name == 'lsa':
            n_topics = 25
            print(f"--- N TOPICS: {n_topics} ---")

            lsa_model = data_mining.LSA_model(tfidf_matrix, lemmatized_dict, n_topics, RANDOM_SEED)
            model_topics = utils.get_model_topics(lsa_model)

            df_lsa_topics = pd.DataFrame({'keywords': model_topics})
            df_lsa_topics.to_pickle('src/data/df_lsa_topics.pkl')
            all_models_results['LSA'] = df_lsa_topics

            # Compute dominant topic per document for frequency analysis
            lsa_assignments = evaluation.get_dominant_topics(lsa_model, tfidf_matrix, model_type="gensim")
            all_topic_assignments['LSA'] = {'assignments': lsa_assignments, 'keywords': df_lsa_topics}
            print("✅ LSA completed!\n")

         # Start LDA model
        elif model_name == 'lda':
            n_topics = 13
            print(f"--- N TOPICS: {n_topics} ---")

            lda_model = data_mining.LDA_model(bow_corpus, lemmatized_dict, n_topics, 30, RANDOM_SEED)

            model_topics = utils.get_model_topics(lda_model)

            df_lda_topics = pd.DataFrame({'keywords': model_topics})
            df_lda_topics.to_pickle('src/data/df_lda_topics.pkl')
            all_models_results['LDA'] = df_lda_topics

            # Compute dominant topic per document for frequency analysis
            lda_assignments = evaluation.get_dominant_topics(lda_model, bow_corpus, model_type="gensim")
            all_topic_assignments['LDA'] = {'assignments': lda_assignments, 'keywords': df_lda_topics}
            print("✅ LDA completed!\n")

         # Start NMF model
        elif model_name == 'nmf':
            n_topics = 13
            print(f"--- N TOPICS: {n_topics} ---")

            w_matrix, h_matrix = data_mining.NMF_model(n_topics, tfidf_sparse, RANDOM_SEED)

            feature_names = list(lemmatized_dict.values())
            topic_words = []

            for topic in h_matrix:
                top_indices = topic.argsort()[:-15 - 1:-1]
                topic_words.append([feature_names[i] for i in top_indices])

            df_nmf_topics = pd.DataFrame({'keywords': topic_words})
            df_nmf_topics.to_pickle('src/data/df_nmf_topics.pkl')
            all_models_results['NMF'] = df_nmf_topics

            # Compute dominant topic per document using the W (document-topic) matrix
            nmf_assignments = evaluation.get_dominant_topics(w_matrix, corpus=[], model_type="nmf")
            all_topic_assignments['NMF'] = {'assignments': nmf_assignments, 'keywords': df_nmf_topics}
            print("✅ NMF completed!\n")

         # Start BERTopic model
        elif model_name == 'bertopic':
            n_topics = 14

            print(f"--- N TOPICS: {n_topics} ---")

            umap_iter = UMAP(min_dist=0.0, metric="cosine", random_state=RANDOM_SEED)
            bertopic_params = {
                "vectorizer_model": vectorizer_model,
                "representation_model": representation_model,
                "umap_model": umap_iter,
                "language": "portuguese",
                "min_topic_size": 2,
                "nr_topics": n_topics,
                "verbose": False,
                "embedding_model": embedding_model,
                "calculate_probabilities": False,
                "top_n_words": 15,
            }

            topics, probs, b_model, new_topics = data_mining.BERTopic_model(
                df_bertopic["comentario"], bertopic_params
            )

            topic_info_df = b_model.get_topic_info()
            df_bertopic_results = pd.DataFrame({
                'keywords': topic_info_df['Representation'],
                'docs': topic_info_df['Representative_Docs']
            })

            topic_word_dict = b_model.get_topics()
            words_list = []

            for topic_id in sorted(topic_word_dict.keys()):
                if topic_id == -1:
                    continue  # Ignore outlier topic

                t_words = [w for w, _ in topic_word_dict[topic_id]]
                if len(t_words) >= 2:
                    words_list.append(t_words)

            df_bertopic_results.to_pickle('src/data/df_bertopikc_topics.pkl')
            all_models_results['BERTopic'] = df_bertopic_results[['keywords']]

            # Use post-outlier-reduction topic assignments for frequency analysis
            all_topic_assignments['BERTopic'] = {
                'assignments': new_topics,
                'keywords': df_bertopic_results[['keywords']]
            }
            print("✅ BERTopic completed!\n")

    # ----- Consolidating and saving all results -----
    # Convert dict to a single DataFrame with a new 'model' column
    for model_name, df in all_models_results.items():
        if 'model' not in df.columns:
            df.insert(0, 'model', model_name)

    df_all_topics = pd.concat(all_models_results.values(), ignore_index=True)

    # Save as CSV (best for universal access without external Excel packages)
    df_all_topics.to_csv('output/topic_modeling/all_models_topics.csv', index=False, encoding='utf-8-sig')

    print("✅ Consolidated topics saved to all_models_topics.csv!\n")

    # Save dominant topic assignments per model for inter-model frequency analysis
    assignments_only = {
        model: data["assignments"] for model, data in all_topic_assignments.items()
    }
    pd.to_pickle(assignments_only, 'src/data/all_topic_assignments.pkl')
    print("✅ Topic assignments saved to all_topic_assignments.pkl!")


if __name__ == "__main__":
    run_pipeline()

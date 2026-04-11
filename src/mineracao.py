from gensim.models import LsiModel
from gensim.models import LdaModel
from sklearn.decomposition import NMF
from bertopic import BERTopic

def LSA_model(tfidf_corpus, dict, n_topics, seed):
    return LsiModel(tfidf_corpus, 
                    id2word=dict, 
                    num_topics=n_topics,
                    random_seed=seed)

def print_lsa_topics(lsa_model, n_words):
    print("Tópicos extraídos pelo modelo LSA:")
    for topico_id, topico in lsa_model.print_topics(num_words=n_words):
        print(f"Tópico #{topico_id}: {topico}")

def LDA_model(corpus, dict, n_topics, n_passes, seed):
    return LdaModel(
    corpus=corpus,
    id2word=dict,
    num_topics=n_topics,
    passes=n_passes,
    random_state=seed
)

def NMF_model(n_topics, tfidf_esparse, seed):
    nmf_engine = NMF(
                    n_components=n_topics,
                    random_state=seed,
                    init='nndsvd',       
                    solver='cd',
                    max_iter=500,
                    tol=0.01,
                    verbose=0
                )
    W = nmf_engine.fit_transform(tfidf_esparse)
    H = nmf_engine.components_

    return W, H

def BERTopic_model(df_col, params_dict):
    bertopic_model = BERTopic(**params_dict)
    
    topics, probs = bertopic_model.fit_transform(df_col)

    new_topics = bertopic_model.reduce_outliers(
                            df_col, topics, strategy="embeddings")
            
    bertopic_model.update_topics(
                            df_col,
                            topics=new_topics,
                            vectorizer_model=params_dict['vectorizer_model'],
                            top_n_words=params_dict.get('top_n_words', 10))
    
    return topics, probs, bertopic_model, new_topics
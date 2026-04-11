import pandas as pd
import numpy as np
import community.community_louvain as community_louvain 
import umap
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics.pairwise import cosine_similarity
import networkx as nx
from collections import Counter
from typing import List

def freq_topicos_nmf(W, df_topics, col_keywords="keywords"):
    """
    W: matriz documento x topico (NMF fit_transform)
    df_topics: DataFrame com uma linha por topico e coluna 'keywords'
    col_keywords: nome da coluna com as palavras do topico
    """
    # 1) topico dominante por documento
    topicos_dominantes = np.argmax(W, axis=1)

    # 2) contagem por topico
    valores, contagens = np.unique(topicos_dominantes, return_counts=True)
    mapa_contagem = dict(zip(valores, contagens))

    # 3) total para calcular %
    total_docs = W.shape[0] if W.shape[0] > 0 else 1

    # 4) monta saida na ordem dos topicos do df
    linhas = []
    for idx, row in df_topics.reset_index(drop=True).iterrows():
        count = mapa_contagem.get(idx, 0)
        freq_percent = (count / total_docs) * 100
        linhas.append({
            "topico": idx,
            "keywords": row[col_keywords],
            "count": count,
            "freq_percent": freq_percent
        })

    return pd.DataFrame(linhas)

def calculate_topic_frequencies(topic_keywords_column: pd.Series) -> pd.DataFrame:
    """
    Calculate the frequency and percentage of each unique topic based on its keywords.
    
    This function assumes each entry in the input column is a list of strings representing
    the keywords for the dominant topic of that document. It groups by unique keyword sets,
    counts occurrences, and computes percentages.
    
    Args:
        topic_keywords_column (pd.Series): A pandas Series where each value is a list of strings
                                           (e.g., ['word1', 'word2']) representing the topic keywords
                                           for a document. Non-list or empty values are skipped.
    
    Returns:
        pd.DataFrame: A DataFrame with columns:
                      - 'topic_keywords': List of keywords for the topic.
                      - 'frequency': Number of documents assigned to this topic.
                      - 'percentage': Percentage of total documents (formatted to 1 decimal place).
                      Sorted by frequency (descending).
    
    Raises:
        ValueError: If the input is not a pandas Series or contains invalid data.
    
    Example:
        # Assuming df['topic_keywords'] is a column of lists like [['a', 'b'], ['c', 'd'], ['a', 'b']]
        freq_df = calculate_topic_frequencies(df['topic_keywords'])
        print(freq_df)
        # Output:
        #   topic_keywords  frequency  percentage
        # 0     [a, b]            2        66.7
        # 1     [c, d]            1        33.3
    """
    if not isinstance(topic_keywords_column, pd.Series):
        raise ValueError("Input must be a pandas Series.")
    
    # Filter out non-list or empty lists, and convert to tuples for hashing/grouping
    valid_entries = topic_keywords_column.dropna().apply(lambda x: tuple(x) if isinstance(x, list) and x else None).dropna()
    
    if valid_entries.empty:
        return pd.DataFrame(columns=['topic_keywords', 'frequency', 'percentage'])
    
    # Count frequencies of each unique topic (as tuple)
    freq_counter = Counter(valid_entries)
    total_docs = len(valid_entries)
    
    # Build the result DataFrame
    data = []
    for topic_tuple, freq in freq_counter.items():
        percentage = (freq / total_docs) * 100
        data.append({
            'topic_keywords': list(topic_tuple),  # Convert back to list for readability
            'frequency': freq,
            'percentage': round(percentage, 1)  # Matches your notebook's formatting
        })
    
    df_result = pd.DataFrame(data).sort_values(by='frequency', ascending=False).reset_index(drop=True)
    return df_result

def generate_topic_vector(topic_keywords, model_emb):
    """
    Recebe um tópico (ex: ['aula', 'professor'])
    Retorna o vetor médio (centroide) desse tópico.
    """
    if not topic_keywords:
        return np.zeros(model_emb.get_sentence_embedding_dimension())
    
    word_embeddings = model_emb.encode(topic_keywords)
    return np.mean(word_embeddings, axis=0)

def plot_umap_vis(df_plot_umap:pd.DataFrame, path):
    plt.figure(figsize=(14, 9))
    sns.set_style("whitegrid") # Fundo limpo e acadêmico

    # Paleta de cores personalizada
    palette = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']

    scatter = sns.scatterplot(
        data=df_plot_umap, 
        x='x', 
        y='y', 
        hue='Modelo',      # Cores diferentes por modelo
        style='Modelo',    # Formas diferentes por modelo
        s=180,             # Tamanho dos pontos
        alpha=0.8,         # Transparência
        palette=palette)
    
    for i in range(df_plot_umap.shape[0]):
        plt.text(
            df_plot_umap.x[i]+0.04, 
            df_plot_umap.y[i]+0.04, 
            df_plot_umap.Tópico[i], 
            fontsize=11, 
            alpha=0.9)
    
    plt.title('Projeção UMAP dos Tópicos Inter-Modelos', fontsize=16)
    plt.xlabel('Dimensão 1', fontsize=12)
    plt.ylabel('Dimensão 2', fontsize=12)
    plt.legend(bbox_to_anchor=(1, 1.011), loc='upper left', title='Modelo', title_fontsize=11) 
    # plt.show()

    plt.savefig(path, dpi=600)

    # plt.tight_layout()
    
def generate_umap_view(topicos_vetores, path):
    all_vectors = []
    all_labels = []   
    all_indices = [] 

    for model_name, vectors in topicos_vetores.items():
        for i, vec in enumerate(vectors):
            all_vectors.append(vec)
            all_labels.append(model_name.upper())
            all_indices.append(f"T{i}")  
    
    X = np.array(all_vectors)

    # UMAP CONFIG
    umap_vis = umap.UMAP(
        n_components=2, 
        n_neighbors=15,   # Padrão bom para equilibrar local/global
        min_dist=0.2,     # <--- MUDANÇA: 0.1 ou 0.2 para evitar sobreposição total
        metric="cosine",  # CRÍTICO: Manter cosine para texto
        random_state=42,  # Garante que o gráfico seja igual sempre
        init='spectral'   # Inicialização padrão do UMAP (muito estável)
    )

    X_embedded = umap_vis.fit_transform(X)

    df_plot_umap = pd.DataFrame({
    'x': X_embedded[:, 0],
    'y': X_embedded[:, 1],
    'Modelo': all_labels,
    'Tópico': all_indices})

    plot_umap_vis(df_plot_umap, path)

def generate_cosine_matrix(topicos_vetores):
    global_vectors = []
    global_labels = []
    model_origins = [] 

    # desempacotando o diciionario com todos os vetores dos topicos
    for model_name, vectors in topicos_vetores.items():
        for i, vec in enumerate(vectors):
            global_vectors.append(vec)
            # criando labels para identificar o modelo e topico. ex: 'LDA T0
            global_labels.append(f"{model_name.upper()} {i}")
            model_origins.append(model_name.upper())

    # converte a lsita de vetores em uma amtriz numpy, onde cada linha é um TÓPICO e cada coluna é uma DIMENSÃO do embedding
    X_global = np.array(global_vectors)

    # imprimindo estrutura da matriz
    print(f"✅ Matriz Global Criada: {X_global.shape} (Tópicos x Dimensões)")

    # calculando similaridade para matriz 65x65
    sim_matrix_global = cosine_similarity(X_global)

    # transformando em df para organizar os dados
    df_cos_sim = pd.DataFrame(
        sim_matrix_global, 
        index=global_labels, 
        columns=global_labels)

    return df_cos_sim, sim_matrix_global

def louvain_comunidade_exclusivos(labels, sim_matrix_global, threshold=0.15, min_models_common=3):
    """
    Constrói um grafo de similaridade entre tópicos e detecta comunidades,
    separando-as em Tópicos em Comum e Tópicos Exclusivos.
    """
    G = nx.Graph()

    # 1. Construir o Grafo (Nós e Arestas)
    for i, label in enumerate(labels):
        model_name = label.split('_')[0]
        G.add_node(i, label=label, model=model_name)

    rows, cols = sim_matrix_global.shape
    for i in range(rows):
        for j in range(i + 1, cols):
            sim_score = sim_matrix_global[i, j]
            if sim_score >= threshold:
                G.add_edge(i, j, weight=sim_score)

    print(f"Grafo criado com {G.number_of_nodes()} nós e {G.number_of_edges()} conexões fortes (limiar >= {threshold}).")

    # 2. Detecção de Comunidades (Louvain)
    partition = community_louvain.best_partition(G, weight='weight', random_state=42)

    # Organizar resultados em um DataFrame
    results = []
    for node_id, community_id in partition.items():
        results.append({
            "Node_ID": node_id,
            "Community_ID": community_id,
            "Topic_Label": G.nodes[node_id]['label'],
            "Model": G.nodes[node_id]['model']
        })
    df_communities = pd.DataFrame(results)

    # 3. Análise dos Agrupamentos (Comum vs Exclusivo)
    # Conta quantos modelos únicos existem dentro de cada comunidade
    comm_composition = df_communities.groupby("Community_ID")['Model'].nunique()

    print("\n" + "="*50)
    print("🎯 1. TÓPICOS EM COMUM (SUPER-TÓPICOS)")
    print("="*50)
    common_comms = comm_composition[comm_composition >= min_models_common].index
    
    if len(common_comms) == 0:
        print(f"Nenhum super-tópico encontrado com pelo menos {min_models_common} modelos.")
    else:
        for comm_id in common_comms:
            group = df_communities[df_communities['Community_ID'] == comm_id]
            models_present = group['Model'].unique()
            print(f"\nSuper-Tópico [Comunidade {comm_id}]:")
            print(f"  Modelos presentes: {', '.join(models_present)}")
            print(f"  Tópicos: {group['Topic_Label'].values}")

    print("\n" + "="*50)
    print("🔒 2. TÓPICOS EXCLUSIVOS (MONO-MODELO)")
    print("="*50)
    exclusive_comms = comm_composition[comm_composition == 1].index
    
    if len(exclusive_comms) == 0:
        print("Nenhuma comunidade exclusiva encontrada.")
    else:
        for comm_id in exclusive_comms:
            group = df_communities[df_communities['Community_ID'] == comm_id]
            model_name = group['Model'].iloc[0]
            topics = group['Topic_Label'].values
            print(f"Comunidade {comm_id} é EXCLUSIVA do modelo: {model_name}")
            print(f"  Tópicos: {', '.join(topics)}")

    print("\n" + "="*50)
    print("👻 3. TÓPICOS ISOLADOS (GRAU = 0)")
    print("="*50)
    isolates = [G.nodes[n]['label'] for n in G.nodes() if G.degree(n) == 0]
    print(f"Total de tópicos totalmente isolados (Exclusivos 'Fortes'): {len(isolates)}")
    if isolates:
        # Mostra apenas os 10 primeiros para não poluir muito a tela
        print(f"  Exemplos: {', '.join(isolates[:10])}{'...' if len(isolates) > 10 else ''}")

    return df_communities, G
import ast
import re
from collections import defaultdict

import umap
import torch
import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from transformers import BertTokenizer, BertModel, BertForMaskedLM
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score

from extract_cluster_data import extract_cluster_data_first, extract_cluster_data_recursion
from compute_optimal_clusters import compute_optimal_clusters_by_elbow_gmm
from read_data import read_data
from metric import compute_cluster_com, compute_cluster_rel_first, compute_cluster_rel_recursion, compute_cluster_dif, \
    compute_cluster_dis
from ask_llm_for_summary import ask_llm_for_summary_first_inner_glm4, ask_llm_for_summary_recursion_inner_glm4


def main():
    file_path = ""  # Enter the path to your Excel file here
    df = read_data(file_path)
    if df is None:
        return

    model_name = "bert-base-chinese"  # Enter the path to your local large language model here
    tokenizer = BertTokenizer.from_pretrained(model_name)
    model = BertModel.from_pretrained(model_name)
    GPU = 6
    if GPU:
        device = torch.device('cuda:{}'.format(GPU))
    else:
        device = torch.device('cpu')
    model = model.to(device)
    model.eval()

    def get_embedding(text):
        inputs = tokenizer(text, padding=True, truncation=True, return_tensors='pt', max_length=512).to(device)
        with torch.no_grad():
            outputs = model(**inputs)
        embedding = outputs.last_hidden_state
        attention_mask = inputs['attention_mask']
        mask = attention_mask.unsqueeze(-1).expand(embedding.size())
        embedding = embedding * mask
        embedding = embedding.sum(dim=1) / mask.sum(dim=1)
        return embedding.cpu().numpy()[0]

    columns_to_embed = ['Requirements', 'name', 'Keywords', 'Domains']

    def extract_name(x):
        try:
            name_all = ast.literal_eval(str(x))
            if isinstance(name_all, list):
                name = name_all[0]
            else:
                name = str(x)
        except (ValueError, SyntaxError):
            name = str(x)
        return name

    for column in columns_to_embed:
        if column == 'name':
            df[column + '_embedding'] = df[column].apply(lambda x: get_embedding(extract_name(x)))
        else:
            df[column + '_embedding'] = df[column].apply(lambda x: get_embedding(str(x)))
        print(f"finish embedding for {column}")

    print(df[['Requirements_embedding', 'name_embedding', 'Keywords_embedding', 'Domains_embedding']].head())

    embedding_data = []
    for column in columns_to_embed:
        embeddings_col = np.array(df[column + '_embedding'].tolist())
        embedding_data.append(embeddings_col)

    embedding_data = np.hstack(embedding_data)
    print(embedding_data.shape)
    embedding_data = embedding_data.reshape(embedding_data.shape[0], -1)
    print(embedding_data.shape)

    scaler = StandardScaler()
    embedding_data_scaled = scaler.fit_transform(embedding_data)

    n_neighbors = int((len(embedding_data_scaled) - 1) ** 0.5)
    n_components = min(5, len(embedding_data_scaled) - 2)

    umap_model = umap.UMAP(
        n_neighbors=n_neighbors,
        n_components=n_components,
        min_dist=0.1,
        random_state=42,
        metric="cosine"
    )

    embedding_data_umap = umap_model.fit_transform(embedding_data_scaled)

    print("Dimensions after dimensionality reduction: ", embedding_data_umap.shape[1])

    max_clusters = int(np.floor(embedding_data_umap.shape[0]))
    optimal_clusters, silhouettes = compute_optimal_clusters_by_elbow_gmm(embedding_data_umap, max_clusters=max_clusters - 1,
                                                                    min_clusters=2)

    print(f"Optimal number of clusters: {optimal_clusters}")

    gmm = GaussianMixture(n_components=optimal_clusters, random_state=42)
    gmm.fit(embedding_data_umap)
    labels = gmm.predict(embedding_data_umap)

    cluster_tree = defaultdict(list)

    cluster_info = []
    for cluster_num in range(optimal_clusters):
        cluster_indices = np.where(labels == cluster_num)[0]
        print(f"Cluster {cluster_num + 1} contains samples with indices: {cluster_indices.tolist()}")

        cluster_data = extract_cluster_data_first(cluster_indices, df)
        cluster_summary = ask_llm_for_summary_first_inner_glm4(cluster_data, GPU='6')

        match_name = re.search(r'cluster_name：([^\n，]+)', cluster_summary)
        match_description = re.search(r'description：([^\n]+)', cluster_summary)

        cluster_name = match_name.group(1) if match_name else None
        cluster_description = match_description.group(1) if match_description else None

        print(f"Cluster {cluster_num + 1} cluster_name:")
        print(cluster_name)
        print(f"Cluster {cluster_num + 1} cluster_description:")
        print(cluster_description)
        print("=" * 80)

        cluster_info.append({
            'cluster_name_1': cluster_name,
            'cluster_description_1': cluster_description
        })

        cluster_tree[0].append({
            'cluster_name_1': cluster_name,
            'cluster_description_1': cluster_description,
            'cluster_indices': cluster_indices.tolist()
        })

    df_cluster_1 = pd.DataFrame(cluster_info)
    print("Final DataFrame with cluster names and descriptions:")
    print(df_cluster_1)

    df_cluster_1.to_csv('cluster_summary_1.csv', index=False)

    # SC
    silhouette_avg = silhouette_score(embedding_data_scaled, labels)
    mapped_silhouette_score = (silhouette_avg + 1) / 2
    print(f"Mapped Silhouette Score for the clustering (range [0, 1]): {mapped_silhouette_score:.4f}")

    # com
    cluster_com = compute_cluster_com(embedding_data_scaled, labels, optimal_clusters)
    # for cluster in cluster_com:
    #     print(f"Cluster {cluster['cluster_num']} average cosine similarity (com): {cluster['average_cosine_similarity']:.4f}")
    average_com = np.mean([cluster['average_cosine_similarity'] for cluster in cluster_com])
    print(f"Average cosine similarity (com) across all clusters: {average_com:.4f}")

    # rel
    cluster_relevance, average_relevance = compute_cluster_rel_first(embedding_data_scaled, labels, optimal_clusters, df, cluster_info)
    # for cluster in cluster_relevance:
    #     print(
    #         f"Cluster {cluster['cluster_num']} - Average cosine similarity with requirements: {cluster['average_cosine_similarity_with_requirements']:.4f}")
    print(f"Average cosine similarity with requirements across all clusters: {average_relevance:.4f}")

    # dif
    name_similarities, description_similarities, total_diff = compute_cluster_dif(embedding_data_scaled, labels, optimal_clusters, cluster_info, level=0)
    # print("\nCluster Name Differences:")
    # for i in range(optimal_clusters):
    #     for j in range(i + 1, optimal_clusters):
    #         similarity = name_similarities.pop(0)
    #         print(f"Mapped cosine similarity between cluster names {i + 1} and {j + 1}: {similarity:.4f}")
    # print("\nCluster Description Differences:")
    # for i in range(optimal_clusters):
    #     for j in range(i + 1, optimal_clusters):
    #         similarity = description_similarities.pop(0)
    #         print(f"Mapped cosine similarity between cluster descriptions {i + 1} and {j + 1}: {similarity:.4f}")
    print(f"\nTotal difference value (dif) across all clusters: {total_diff:.4f}")

    # dis
    cluster_distribution_values, mean_size, std_size, average_distribution_value = compute_cluster_dis(embedding_data_scaled, labels, optimal_clusters)
    # for i, distribution_value in enumerate(cluster_distribution_values):
    #     print(f"Cluster {i} distribution value: {distribution_value}")
    print(f"Average distribution value across all clusters: {average_distribution_value:.4f}")

    if optimal_clusters <= 8:
        print(f"Recursion stopped at level 0 because optimal clusters are <= 8.")
        return cluster_tree

    recursion_level = 1

    def recursive_clustering(df_input, recursion_level, cluster_tree=None):
        """
        Perform recursive clustering and return a tree structure with clustering results.

        Args:
        - df_input: DataFrame containing the current clustering results.
        - recursion_level: Current level of recursion.
        - cluster_tree: The tree structure that stores clustering results (used to build the hierarchy).

        Returns:
        - Updated cluster_tree with new clustering results.
        """
        print(f"Recursion Level {recursion_level} - Starting clustering...")

        cluster_name_column = f'cluster_name_{recursion_level}'
        cluster_description_column = f'cluster_description_{recursion_level}'

        if cluster_name_column not in df_input.columns or cluster_description_column not in df_input.columns:
            raise ValueError(
                f"Error: Previous recursion did not generate '{cluster_name_column}' or '{cluster_description_column}' columns. Cannot proceed!")

        cluster_info_recursion = []

        for column in [cluster_name_column, cluster_description_column]:
            df_input[column + f'_embedding_{recursion_level}'] = df_input[column].apply(lambda x: get_embedding(str(x)))
            print(f"Finish embedding for {column} at recursion level {recursion_level}")

        embedding_data_recursion = []
        for column in [cluster_name_column, cluster_description_column]:
            embedding_column = column + f'_embedding_{recursion_level}'
            if embedding_column in df_input.columns:
                embeddings_col_recursion = np.array(df_input[embedding_column].tolist())
                embedding_data_recursion.append(embeddings_col_recursion)
            else:
                print(f"Warning: {embedding_column} not found in columns.")

        embedding_data_recursion = np.hstack(embedding_data_recursion)
        print(f"Shape of embedding data at recursion level {recursion_level}: {embedding_data_recursion.shape}")

        scaler = StandardScaler()
        embedding_data_recursion_scaled = scaler.fit_transform(embedding_data_recursion)

        n_neighbors = int((len(embedding_data_recursion_scaled) - 1) ** 0.5)
        n_components = min(5, len(embedding_data_recursion_scaled) - 2)

        umap_model = umap.UMAP(
            n_neighbors=n_neighbors,
            n_components=n_components,
            min_dist=0.1,
            random_state=42,
            metric="cosine"
        )

        embedding_data_recursion_umap = umap_model.fit_transform(embedding_data_recursion_scaled)

        print("Dimensions after dimensionality reduction: ", embedding_data_recursion_umap.shape[1])

        max_clusters_recursion = int(np.floor(embedding_data_recursion_umap.shape[0]))
        optimal_clusters_recursion, silhouettes_recursion = compute_optimal_clusters_by_elbow_gmm(embedding_data_recursion_umap, max_clusters=max_clusters_recursion - 1, min_clusters=2)

        print(f"Optimal number of clusters: {optimal_clusters_recursion}")

        gmm = GaussianMixture(n_components=optimal_clusters_recursion, random_state=42)
        gmm.fit(embedding_data_recursion_umap)
        labels_recursion = gmm.predict(embedding_data_recursion_umap)

        cluster_info_recursion = []
        for cluster_num in range(optimal_clusters_recursion):
            cluster_indices = np.where(labels_recursion == cluster_num)[0]
            print(f"Cluster {cluster_num + 1} contains samples with indices: {cluster_indices.tolist()}")

            cluster_data_recursion = extract_cluster_data_recursion(cluster_indices, df_input, recursion_level)
            cluster_summary_recursion = ask_llm_for_summary_recursion_inner_glm4(cluster_data_recursion, GPU='6')

            match_name = re.search(r'cluster_name：([^\n，]+)', cluster_summary_recursion)
            match_description = re.search(r'description：([^\n]+)', cluster_summary_recursion)

            cluster_name = match_name.group(1) if match_name else None
            cluster_description = match_description.group(1) if match_description else None

            print(f"Cluster {cluster_num + 1} cluster_name:")
            print(cluster_name)
            print(f"Cluster {cluster_num + 1} cluster_description:")
            print(cluster_description)

            cluster_info_recursion.append({
                f'cluster_name_{recursion_level + 1}': cluster_name,
                f'cluster_description_{recursion_level + 1}': cluster_description,
                'cluster_indices': cluster_indices.tolist()
            })

        if recursion_level not in cluster_tree:
            cluster_tree[recursion_level] = []

        cluster_tree[recursion_level].extend(cluster_info_recursion)

        df_cluster_recursion = pd.DataFrame(cluster_info_recursion)
        print(f"DataFrame for recursion level {recursion_level}:")
        print(df_cluster_recursion)

        df_cluster_recursion.to_csv(f'cluster_summary_{recursion_level + 1}.csv', index=False)

        # SC
        silhouette_avg = silhouette_score(embedding_data_recursion_scaled, labels_recursion)
        mapped_silhouette_score = (silhouette_avg + 1) / 2  # 映射到 [0, 1] 区间
        print(f"Recursion Level {recursion_level} - Mapped Silhouette Score for the clustering (range [0, 1]): {mapped_silhouette_score:.4f}")

        # com
        cluster_com = compute_cluster_com(embedding_data_recursion_scaled, labels_recursion, optimal_clusters_recursion)
        # for cluster in cluster_com:
        #     print(f"Recursion Level {recursion_level} - Cluster {cluster['cluster_num']} average cosine similarity (com): {cluster['average_cosine_similarity']:.4f}")
        average_com = np.mean([cluster['average_cosine_similarity'] for cluster in cluster_com])
        print(f"Average cosine similarity (com) across all clusters: {average_com:.4f}")

        # rel
        cluster_relevance, average_relevance = compute_cluster_rel_recursion(embedding_data_recursion_scaled, labels_recursion, optimal_clusters_recursion, df_input, cluster_info_recursion, recursion_level)
        # print("\nCluster Relevance:")
        # for cluster in cluster_relevance:
        #     print(f"Cluster {cluster['cluster_num']} - Average cosine similarity with sample descriptions: {cluster['average_cosine_similarity_with_sample_descriptions']:.4f}")
        print(f"Average cosine similarity with sample descriptions across all clusters: {average_relevance:.4f}")

        # dif
        name_similarities, description_similarities, total_diff = compute_cluster_dif(embedding_data_recursion_scaled, labels_recursion, optimal_clusters_recursion, cluster_info_recursion, recursion_level)
        # print("\nCluster Name Differences:")
        # for i in range(optimal_clusters_recursion):
        #     for j in range(i + 1, optimal_clusters_recursion):
        #         similarity = name_similarities.pop(0)
        #         print(f"Mapped cosine similarity between cluster names {i + 1} and {j + 1}: {similarity:.4f}")
        # print("\nCluster Description Differences:")
        # for i in range(optimal_clusters_recursion):
        #     for j in range(i + 1, optimal_clusters_recursion):
        #         similarity = description_similarities.pop(0)
        #         print(f"Mapped cosine similarity between cluster descriptions {i + 1} and {j + 1}: {similarity:.4f}")
        print(f"\nTotal difference value (dif) across all clusters: {total_diff:.4f}")

        # dis
        cluster_distribution_values, mean_size, std_size, average_distribution_value = compute_cluster_dis(embedding_data_recursion_scaled, labels_recursion, optimal_clusters_recursion)
        # for i, distribution_value in enumerate(cluster_distribution_values):
        #     print(f"Cluster {i} distribution value: {distribution_value}")
        print(f"Average distribution value across all clusters: {average_distribution_value:.4f}")

        if optimal_clusters_recursion <= 8:
            print(f"Recursion stopped at level {recursion_level} because optimal clusters are <= 8.")
            return cluster_tree

        return recursive_clustering(df_cluster_recursion, recursion_level + 1, cluster_tree)

    def print_cluster_tree(cluster_tree):
        """
        Print the cluster tree structure in reverse order, starting from the deepest level.

        Args:
        - cluster_tree: The tree structure containing clustering information.
        """
        def print_recursive(level, max_level, clusters):
            if not clusters:
                return

            cluster_name_field = f'cluster_name_{level + 1}'
            cluster_description_field = f'cluster_description_{level + 1}'

            for i, cluster in enumerate(clusters):
                cluster_name = cluster.get(cluster_name_field, 'N/A')
                cluster_description = cluster.get(cluster_description_field, 'N/A')

                print(f"{' ' * 4 * (max_level - level)}Level {level + 1} - Cluster {i + 1}:")
                print(f"{' ' * 4 * (max_level - level + 1)}Name: {cluster_name}")
                print(f"{' ' * 4 * (max_level - level + 1)}Description: {cluster_description}")
                print(f"{' ' * 4 * (max_level - level + 1)}Indices: {cluster.get('cluster_indices', 'N/A')}")

            print(' ')
            if level - 1 in cluster_tree:
                print_recursive(level - 1, max_level, cluster_tree[level - 1])

        max_level = max(cluster_tree.keys()) if cluster_tree else 0
        print_recursive(max_level, max_level, cluster_tree.get(max_level, []))

    cluster_tree = recursive_clustering(df_cluster_1, recursion_level=1, cluster_tree=cluster_tree)
    print("\nCluster Tree Structure:")
    print_cluster_tree(cluster_tree)

if __name__ == "__main__":
    main()

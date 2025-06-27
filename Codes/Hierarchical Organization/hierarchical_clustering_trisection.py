import ast
import re
from collections import defaultdict
import torch
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics import silhouette_score
import scipy.cluster.hierarchy as sch
from transformers import BertTokenizer, BertModel, BertForMaskedLM
from scipy.cluster.hierarchy import fcluster
from read_data import read_data
from metric import compute_cluster_dis
from ask_llm_for_summary import ask_llm_for_summary_first_inner, ask_llm_for_summary_first_inner_glm4


def main():
    file_path = ""
    df = read_data(file_path)
    if df is None:
        return

    model_name = "bert-base-chinese"
    tokenizer = BertTokenizer.from_pretrained(model_name)
    model = BertModel.from_pretrained(model_name)
    GPU = 7
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


    linked = sch.linkage(embedding_data, method='ward')
    max_distance = np.max(linked[:, 2])
    print(f"Maximum distance in the dendrogram: {max_distance}")
    thresholds = [max_distance * (i / 15) for i in range(2, 0, -1)]
    print(f"Thresholds for multiple cuts: {thresholds}")

    labels_per_threshold = {}
    clusters_per_threshold = {}
    splits_per_threshold = {}

    for i, threshold in enumerate(thresholds):
        labels = fcluster(linked, threshold, criterion='distance')
        labels_per_threshold[f'cut_{i + 1}_threshold_{threshold}'] = labels
        clusters = {}
        for idx, label in enumerate(labels):
            if label not in clusters:
                clusters[label] = []
            clusters[label].append(idx)

        clusters_per_threshold[f'cut_{i + 1}_threshold_{threshold}'] = clusters

        if i > 0:
            prev_clusters = clusters_per_threshold[f'cut_{i}_threshold_{thresholds[i - 1]}']
            split_info = []
            for prev_cluster_id, prev_members in prev_clusters.items():
                new_split = []
                for cluster_id, members in clusters.items():
                    if any(member in prev_members for member in members):
                        new_split.append(cluster_id)
                if new_split:
                    split_info.append((prev_cluster_id, new_split))

            splits_per_threshold[f'cut_{i + 1}_threshold_{threshold}'] = split_info

            print(f"Previous layer splitting information:")
            for prev_cluster_id, new_clusters in split_info:
                print(f"Previous cluster {prev_cluster_id} was split into the following clusters: {new_clusters}")

    for key, clusters in clusters_per_threshold.items():
        print(f"\n{key} number of cluster labels: {len(clusters)}")

    df['Cluster_Label'] = labels

    min_threshold = thresholds[-1]
    second_threshold = thresholds[-2]

    labels_min_threshold = labels_per_threshold['cut_2_threshold_' + str(min_threshold)]
    num_cluster_min = len(set(labels_min_threshold))
    labels_second_threshold = labels_per_threshold['cut_1_threshold_' + str(second_threshold)]
    num_cluster_second = len(set(labels_second_threshold))
    df['Cluster_Label_1'] = labels_min_threshold
    df['Cluster_Label_2'] = labels_second_threshold

    cluster_tree = defaultdict(list)

    cluster_info = []

    print(f"================================ the first clustering ======================================")

    for cluster_id in set(labels_min_threshold):
        cluster_data = df[df[f'Cluster_Label_1']  == cluster_id].to_dict(orient='records')
        cluster_indices = df.index[df[f'Cluster_Label_1'] == cluster_id].tolist()
        summary = ask_llm_for_summary_first_inner_glm4(cluster_data, GPU='7')
        match_name = re.search(r'cluster_name：([^\n，]+)', summary)
        match_description = re.search(r'description：([^\n，]+)', summary)
        cluster_name = match_name.group(1) if match_name else None
        cluster_description = match_description.group(1) if match_description else None

        print(f"Cluster {cluster_id} cluster_name:")
        print(cluster_name)
        print(f"Cluster {cluster_id} cluster_description:")
        print(cluster_description)
        print(f"Cluster {cluster_id} cluster_indices:")
        print(cluster_indices)
        print("=" * 80)

        cluster_info.append({
            'cluster_name_1': cluster_name,
            'cluster_description_1': cluster_description
        })

        cluster_tree[0].append({
            'cluster_name_1': cluster_name,
            'cluster_description_1': cluster_description,
            'cluster_indices_1': cluster_indices
        })

    df_cluster_1 = pd.DataFrame(cluster_info)
    print("Final DataFrame with cluster names and descriptions:")
    print(df_cluster_1)

    #SC
    silhouette_avg = silhouette_score(embedding_data, labels_min_threshold)
    mapped_silhouette_score = (silhouette_avg + 1) / 2
    print(f"Silhouette score of the first clustering: {mapped_silhouette_score:.4f}")

    # com
    optimal_clusters = len(set(labels_min_threshold))
    cluster_com = []
    for cluster_num in range(optimal_clusters):
        cluster_indices = np.where(labels_min_threshold == cluster_num + 1)[0]
        cluster_embeddings = embedding_data[cluster_indices]
        print(f"Processing Cluster {cluster_num}: {len(cluster_embeddings)} samples, indices: {cluster_indices}")
        if len(cluster_embeddings) == 1:
            cluster_com.append({
                'cluster_num': cluster_num + 1,
                'average_cosine_similarity': 1.0
            })
            continue
        cosine_similarities = []
        for i in range(len(cluster_embeddings)):
            for j in range(i + 1, len(cluster_embeddings)):
                vec_i = cluster_embeddings[i]
                vec_j = cluster_embeddings[j]
                norm_i = np.linalg.norm(vec_i)
                norm_j = np.linalg.norm(vec_j)
                if norm_i > 0 and norm_j > 0:
                    cosine_similarity = np.dot(vec_i, vec_j) / (norm_i * norm_j)
                    cosine_similarities.append(cosine_similarity)
                else:
                    print(f"One of the vectors is a zero vector: {vec_i if norm_i == 0 else vec_j}")

        if not cosine_similarities:
            print(f"Cluster {cluster_num} has no valid cosine similarities.")
            continue

        mapped_cosine_similarities = [(sim + 1) / 2 for sim in cosine_similarities]
        average_cosine_similarity = np.mean(mapped_cosine_similarities)
        cluster_com.append({
            'cluster_num': cluster_num + 1,
            'average_cosine_similarity': average_cosine_similarity
        })

    overall_average_com = np.nanmean([com['average_cosine_similarity'] for com in cluster_com])
    for com in cluster_com:
        print(f"Cluster {com['cluster_num']} average cosine similarity: {com['average_cosine_similarity']:.4f}")
    print(f"Overall average cosine similarity (com) across all clusters: {overall_average_com:.4f}")


    #rel
    def compute_cluster_rel_first(embedding_data_scaled, labels, optimal_clusters, df, cluster_info):
        cluster_relevance = []
        requirments_embeddings = [get_embedding(str(requirement)) for requirement in df['Requirements']]
        requirements_embeddings = np.array(requirments_embeddings)
        for cluster_num in range(optimal_clusters):
            cluster_indices = np.where(labels == cluster_num + 1)[0]
            cluster_name = cluster_info[cluster_num].get('cluster_name_1', '')
            cluster_description = cluster_info[cluster_num].get('cluster_description_1', '')
            cluster_description_embedding = get_embedding(str(cluster_description))
            similarities = []
            for idx in cluster_indices:
                req_embedding = requirements_embeddings[idx]
                cosine_similarity = np.dot(req_embedding, cluster_description_embedding) / (np.linalg.norm(req_embedding) * np.linalg.norm(cluster_description_embedding))
                similarities.append(cosine_similarity)
            mapped_similarities =  [(sim + 1) / 2 for sim in similarities]
            average_similarity = np.mean(mapped_similarities)
            cluster_relevance.append({
                'cluster_num': cluster_num + 1,
                'average_cosine_similarity_with_requirements': average_similarity
            })
        average_relevance = np.mean([cluster['average_cosine_similarity_with_requirements'] for cluster in cluster_relevance])
        return cluster_relevance, average_relevance

    optimal_clusters = len(set(labels_min_threshold))
    cluster_relevance, average_relevance = compute_cluster_rel_first(embedding_data, labels_min_threshold, optimal_clusters, df, cluster_info)
    for relevance in cluster_relevance:
        print(f"Cluster {relevance['cluster_num']} average cosine similarity with requirements: {relevance['average_cosine_similarity_with_requirements']:.4f}")
    print(f"Overall average cosine similarity with requirements (rel) across all clusters: {average_relevance:.4f}")

    #dif
    def compute_cluster_dif_first(embedding_data_scaled, labels, optimal_clusters, cluster_info, level):
        cluster_name_embeddings = []
        cluster_description_embeddings = []
        for cluster_num in range(optimal_clusters):
            cluster_name_key = f'cluster_name_{level + 1}'
            cluster_description_key = f'cluster_description_{level + 1}'
            cluster_name = cluster_info[cluster_num].get(cluster_name_key, '')
            cluster_description = cluster_info[cluster_num].get(cluster_description_key, '')
            cluster_name_embedding = get_embedding(str(cluster_name))
            cluster_description_embedding = get_embedding(str(cluster_description))
            cluster_name_embeddings.append(cluster_name_embedding)
            cluster_description_embeddings.append(cluster_description_embedding)
        name_cosine_similarities = []
        for i in range(optimal_clusters):
            for j in range(i + 1, optimal_clusters):
                dot_product = np.dot(cluster_name_embeddings[i], cluster_name_embeddings[j])
                norm_i = np.linalg.norm(cluster_name_embeddings[i])
                norm_j = np.linalg.norm(cluster_name_embeddings[j])
                cosine_similarity = dot_product / (norm_i * norm_j)
                mapped_similarity = (1 - cosine_similarity) / 2
                name_cosine_similarities.append(mapped_similarity)
        description_cosine_similarities = []
        for i in range(optimal_clusters):
            for j in range(i + 1, optimal_clusters):
                dot_product = np.dot(cluster_description_embeddings[i], cluster_description_embeddings[j])
                norm_i = np.linalg.norm(cluster_description_embeddings[i])
                norm_j = np.linalg.norm(cluster_description_embeddings[j])
                cosine_similarity = dot_product / (norm_i * norm_j)
                mapped_similarity = (1 - cosine_similarity) / 2
                description_cosine_similarities.append(mapped_similarity)
        if name_cosine_similarities:
            average_name_similarity = np.mean(name_cosine_similarities)
        else:
            average_name_similarity = 0
        if description_cosine_similarities:
            average_description_similarity = np.mean(description_cosine_similarities)
        else:
            average_description_similarity = 0
        total_diff = (average_name_similarity + average_description_similarity)  / 2
        return name_cosine_similarities, description_cosine_similarities, total_diff

    level = 0
    name_differences, description_differences, total_differences = compute_cluster_dif_first(embedding_data, labels_min_threshold, optimal_clusters, cluster_info, level)
    print(f"\nTotal differences across all clusters: {total_differences:.4f}")

    #dis
    def compute_cluster_dis(labels, optimal_clusters):
        cluster_distribution_values = []
        cluster_sizes = [np.sum(labels == cluster_num) for cluster_num in range(optimal_clusters)]
        mean_size = np.mean(cluster_sizes)
        std_size = np.std(cluster_sizes)
        for cluster_num in range(optimal_clusters):
            cluster_size = cluster_sizes[cluster_num]
            lower_bound = mean_size - 2 * std_size
            upper_bound = mean_size + 2 * std_size
            if lower_bound <= cluster_size <= upper_bound:
                cluster_distribution_values.append(1)
            else:
                cluster_distribution_values.append(0)
        average_distribution_value = np.mean(cluster_distribution_values)
        return cluster_distribution_values, mean_size, std_size, average_distribution_value


    cluster_distribution_values, mean_size, std_size, average_distribution_value = compute_cluster_dis(labels_min_threshold, optimal_clusters)
    print("\ndistribution value of each cluster:")
    for i,value in enumerate(cluster_distribution_values):
        print(f"Cluster {i + 1}: {value}")
    print(f"Average cluster distribution value: {average_distribution_value:.4f}")


    print(f"================================ the second clustering ======================================")

    for cluster_id in set(labels_second_threshold):
        cluster_data = df[df[f'Cluster_Label_2']  == cluster_id].to_dict(orient='records')
        cluster_indices = df.index[df['Cluster_Label_2'] == cluster_id].tolist()
        summary = ask_llm_for_summary_first_inner_glm4(cluster_data, GPU='7')
        match_name = re.search(r'cluster_name：([^\n，]+)', summary)
        match_description = re.search(r'description：([^\n]+)', summary)
        cluster_name = match_name.group(1) if match_name else None
        cluster_description = match_description.group(1) if match_description else None

        print(f"Cluster {cluster_id} cluster_name:")
        print(cluster_name)
        print(f"Cluster {cluster_id} cluster_description:")
        print(cluster_description)
        print(f"Cluster {cluster_id} cluster_indices:")
        print(cluster_indices)
        print("=" * 80)

        cluster_info.append({
            'cluster_name_2': cluster_name,
            'cluster_description_2': cluster_description
        })

        cluster_tree[0].append({
            'cluster_name_2': cluster_name,
            'cluster_description_2': cluster_description,
            'cluster_indices_2': cluster_indices
        })

    df_cluster_2 = pd.DataFrame(cluster_info)
    print("Final DataFrame with cluster names and descriptions:")
    print(df_cluster_2)

    #SC
    silhouette_avg = silhouette_score(embedding_data, labels_second_threshold)
    mapped_silhouette_score = (silhouette_avg + 1) / 2
    print(f"Silhouette score of the second clustering: {mapped_silhouette_score:.4f}")

    # com
    optimal_clusters = len(set(labels_second_threshold))
    cluster_com = []
    for cluster_num in range(optimal_clusters):
        cluster_indices = np.where(labels_second_threshold == cluster_num + 1)[0]
        cluster_embeddings = embedding_data[cluster_indices]
        print(f"Processing Cluster {cluster_num}: {len(cluster_embeddings)} samples, indices: {cluster_indices}")
        if len(cluster_embeddings) == 1:
            cluster_com.append({
                'cluster_num': cluster_num + 1,
                'average_cosine_similarity': 1.0
            })
            continue
        cosine_similarities = []
        for i in range(len(cluster_embeddings)):
            for j in range(i + 1, len(cluster_embeddings)):
                vec_i = cluster_embeddings[i]
                vec_j = cluster_embeddings[j]
                norm_i = np.linalg.norm(vec_i)
                norm_j = np.linalg.norm(vec_j)
                if norm_i > 0 and norm_j > 0:
                    cosine_similarity = np.dot(vec_i, vec_j) / (norm_i * norm_j)
                    cosine_similarities.append(cosine_similarity)
                else:
                    print(f"One of the vectors is a zero vector: {vec_i if norm_i == 0 else vec_j}")

        if not cosine_similarities:
            print(f"Cluster {cluster_num} has no valid cosine similarities.")
            continue

        mapped_cosine_similarities = [(sim + 1) / 2 for sim in cosine_similarities]
        average_cosine_similarity = np.mean(mapped_cosine_similarities)
        cluster_com.append({
            'cluster_num': cluster_num + 1,
            'average_cosine_similarity': average_cosine_similarity
        })

    overall_average_com = np.nanmean([com['average_cosine_similarity'] for com in cluster_com])
    for com in cluster_com:
        print(f"Cluster {com['cluster_num']} average cosine similarity: {com['average_cosine_similarity']:.4f}")
    print(f"Overall average cosine similarity (com) across all clusters: {overall_average_com:.4f}")

    #rel
    def compute_cluster_rel_second(labels, optimal_clusters, df, cluster_info):
        cluster_relevance = []
        requirments_embeddings = [get_embedding(str(requirement)) for requirement in df['Requirements']]
        requirements_embeddings = np.array(requirments_embeddings)
        for cluster_num in range(optimal_clusters):
            cluster_indices = np.where(labels == cluster_num + 1)[0]
            cluster_name = cluster_info[cluster_num].get('cluster_name_2', '')
            cluster_description = cluster_info[cluster_num].get('cluster_description_2', '')
            cluster_description_embedding = get_embedding(str(cluster_description))
            similarities = []
            for idx in cluster_indices:
                req_embedding = requirements_embeddings[idx]
                cosine_similarity = np.dot(req_embedding, cluster_description_embedding) / (np.linalg.norm(req_embedding) * np.linalg.norm(cluster_description_embedding))
                similarities.append(cosine_similarity)
            mapped_similarities =  [(sim + 1) / 2 for sim in similarities]
            average_similarity = np.mean(mapped_similarities)
            cluster_relevance.append({
                'cluster_num': cluster_num + 1,
                'average_cosine_similarity_with_requirements': average_similarity
            })
        average_relevance = np.mean([cluster['average_cosine_similarity_with_requirements'] for cluster in cluster_relevance])
        return cluster_relevance, average_relevance

    optimal_clusters = len(set(labels_second_threshold))
    cluster_relevance, average_relevance = compute_cluster_rel_second(labels_second_threshold, optimal_clusters, df, cluster_info)
    for relevance in cluster_relevance:
        print(f"Cluster {relevance['cluster_num']} average cosine similarity with requirements: {relevance['average_cosine_similarity_with_requirements']:.4f}")
    print(f"Overall average cosine similarity with requirements (rel) across all clusters: {average_relevance:.4f}")

    #dif
    def compute_cluster_dif_second(optimal_clusters, cluster_info, level, num_cluster_min):
        cluster_name_embeddings = []
        cluster_description_embeddings = []
        for cluster_num in range(optimal_clusters):
            cluster_name_key = f'cluster_name_{level + 1}'
            cluster_description_key = f'cluster_description_{level + 1}'
            cluster_name = cluster_info[cluster_num + num_cluster_min].get(cluster_name_key, '')
            cluster_description = cluster_info[cluster_num + num_cluster_min].get(cluster_description_key, '')
            cluster_name_embedding = get_embedding(str(cluster_name))
            cluster_description_embedding = get_embedding(str(cluster_description))
            cluster_name_embeddings.append(cluster_name_embedding)
            cluster_description_embeddings.append(cluster_description_embedding)
        name_cosine_similarities = []
        for i in range(optimal_clusters):
            for j in range(i + 1, optimal_clusters):
                dot_product = np.dot(cluster_name_embeddings[i], cluster_name_embeddings[j])
                norm_i = np.linalg.norm(cluster_name_embeddings[i])
                norm_j = np.linalg.norm(cluster_name_embeddings[j])
                cosine_similarity = dot_product / (norm_i * norm_j)
                mapped_similarity = (1 - cosine_similarity) / 2
                name_cosine_similarities.append(mapped_similarity)
        description_cosine_similarities = []
        for i in range(optimal_clusters):
            for j in range(i + 1, optimal_clusters):
                dot_product = np.dot(cluster_description_embeddings[i], cluster_description_embeddings[j])
                norm_i = np.linalg.norm(cluster_description_embeddings[i])
                norm_j = np.linalg.norm(cluster_description_embeddings[j])
                cosine_similarity = dot_product / (norm_i * norm_j)
                mapped_similarity = (1 - cosine_similarity) / 2
                description_cosine_similarities.append(mapped_similarity)
        if name_cosine_similarities:
            average_name_similarity = np.mean(name_cosine_similarities)
        else:
            average_name_similarity = 0
        if description_cosine_similarities:
            average_description_similarity = np.mean(description_cosine_similarities)
        else:
            average_description_similarity = 0
        total_diff = (average_name_similarity + average_description_similarity)  / 2
        return name_cosine_similarities, description_cosine_similarities, total_diff

    level = 1
    name_differences, description_differences, total_differences = compute_cluster_dif_second(optimal_clusters, cluster_info, level, num_cluster_min)
    print("Differences between the names of each cluster:")
    for i, dif in enumerate(name_differences):
        print(f"Cluster pair {i}: {dif:.4f}")
    print("Differences between the descriptions of each cluster:")
    for i, dif in enumerate(description_differences):
        print(f"Cluster pair {i}: {dif:.4f}")
    print(f"\nTotal differences across all clusters: {total_differences:.4f}")

    #dis
    cluster_distribution_values, mean_size, std_size, average_distribution_value = compute_cluster_dis(labels, optimal_clusters)
    print("\ndistribution value of each cluster:")
    for i,value in enumerate(cluster_distribution_values):
        print(f"Cluster {i + 1}: {value}")
    print(f"Average cluster distribution value: {average_distribution_value:.4f}")

if __name__ == "__main__":
    main()

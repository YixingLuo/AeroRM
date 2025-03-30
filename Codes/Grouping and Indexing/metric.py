import numpy as np
from safetensors import torch
from sentence_transformers import SentenceTransformer
from torch import cosine_similarity
import ast
import re
from collections import defaultdict
from pprint import pprint

import umap
import torch
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from matplotlib import pyplot as plt
import scipy.cluster.hierarchy as sch
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from sentence_transformers import SentenceTransformer
from transformers import BertTokenizer, BertModel, BertForMaskedLM
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score

model_name = "bert-base-chinese"  # Enter the path to your local large language model here
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertModel.from_pretrained(model_name)
GPU = 5
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

def compute_cluster_com(embedding_data_scaled, labels, optimal_clusters):
    cluster_com = []
    for cluster_num in range(optimal_clusters):
        cluster_indices = np.where(labels == cluster_num)[0]
        cluster_embeddings = embedding_data_scaled[cluster_indices]
        if len(cluster_embeddings) == 1:
            cluster_com.append({
                'cluster_num': cluster_num + 1,
                'average_cosine_similarity': 1.0
            })
            continue
        cosine_similarities = []
        for i in range(len(cluster_embeddings)):
            for j in range(i + 1, len(cluster_embeddings)):
                cosine_similarity = np.dot(cluster_embeddings[i], cluster_embeddings[j]) / (np.linalg.norm(cluster_embeddings[i]) * np.linalg.norm(cluster_embeddings[j]))
                cosine_similarities.append(cosine_similarity)
        mapped_cosine_similarities = [(sim + 1) / 2 for sim in cosine_similarities]
        average_cosine_similarity = np.mean(mapped_cosine_similarities)
        cluster_com.append({
            'cluster_num': cluster_num + 1,
            'average_cosine_similarity': average_cosine_similarity
        })
    return cluster_com

def compute_cluster_rel_first(embedding_data_scaled, labels, optimal_clusters, df, cluster_info):
    cluster_relevance = []
    requirements_embeddings = [get_embedding(str(x)) for x in df['Requirements']]
    requirements_embeddings = np.array(requirements_embeddings)
    for cluster_num in range(optimal_clusters):
        cluster_indices = np.where(labels == cluster_num)[0]
        cluster_name = cluster_info[cluster_num].get('cluster_name_1', '')
        cluster_description = cluster_info[cluster_num].get('cluster_description_1', '')
        cluster_description_embedding = get_embedding(str(cluster_description))
        similarities = []
        for idx in cluster_indices:
            req_embedding = requirements_embeddings[idx]
            cosine_similarity = np.dot(req_embedding, cluster_description_embedding) / (np.linalg.norm(req_embedding) * np.linalg.norm(cluster_description_embedding))
            similarities.append(cosine_similarity)
        mapped_similarities = [(sim + 1) / 2 for sim in similarities]
        average_similarity = np.mean(mapped_similarities)
        cluster_relevance.append({
            'cluster_num': cluster_num + 1,
            'average_cosine_similarity_with_requirements': average_similarity
        })
    average_relevance = np.mean([cluster['average_cosine_similarity_with_requirements'] for cluster in cluster_relevance])
    return cluster_relevance, average_relevance

def compute_cluster_rel_recursion(embedding_data_scaled, labels, optimal_clusters, df, cluster_info, level):
    cluster_relevance = []
    cluster_name_field = f'cluster_name_{level + 1}'
    cluster_description_field = f'cluster_description_{level + 1}'
    for cluster_num in range(optimal_clusters):
        cluster_indices = np.where(labels == cluster_num)[0]
        cluster_name = cluster_info[cluster_num].get(cluster_name_field, '')
        cluster_description = cluster_info[cluster_num].get(cluster_description_field, '')
        cluster_description_embedding = get_embedding(str(cluster_description))
        similarities = []
        for idx in cluster_indices:
            cluster_name_idx_field = f'cluster_name_{level}'
            cluster_description_idx_field = f'cluster_description_{level}'
            sample_cluster_description = df.loc[idx, cluster_description_idx_field]
            sample_description_embedding = get_embedding(str(sample_cluster_description))
            cosine_similarity = np.dot(sample_description_embedding, cluster_description_embedding) / (np.linalg.norm(sample_description_embedding) * np.linalg.norm(cluster_description_embedding))
            similarities.append(cosine_similarity)
        mapped_similarities = [(sim + 1) / 2 for sim in similarities]
        average_similarity = np.mean(mapped_similarities)
        cluster_relevance.append({
            'cluster_num': cluster_num + 1,
            'average_cosine_similarity_with_sample_descriptions': average_similarity
        })
    average_relevance = np.mean([cluster['average_cosine_similarity_with_sample_descriptions'] for cluster in cluster_relevance])
    return cluster_relevance, average_relevance

def compute_cluster_dif(embedding_data_scaled, labels, optimal_clusters, cluster_info, level):
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
    average_name_similarity = np.mean(name_cosine_similarities)
    average_description_similarity = np.mean(description_cosine_similarities)
    total_diff = (average_name_similarity + average_description_similarity)  / 2
    return name_cosine_similarities, description_cosine_similarities, total_diff

def compute_cluster_dis(embedding_data_scaled, labels, optimal_clusters):
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

import numpy as np
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture

def compute_optimal_clusters_by_biclike(embedding_data_scaled, max_clusters, min_clusters, early_stop_threshold = 10):
    """
    Calculate the BIC-like value based on K-means clustering (estimated through sum of squared residuals)
    """
    bics = []
    rss_values = []
    bic_part1 = []
    bic_part2 = []
    consecutive_increases = 0

    for n_clusters in range(min_clusters, max_clusters + 1):
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        kmeans.fit(embedding_data_scaled)

        labels = kmeans.labels_
        centers = kmeans.cluster_centers_

        RSS = np.sum((embedding_data_scaled - centers[labels]) ** 2)
        if RSS == 0:
            continue

        N = embedding_data_scaled.shape[0]
        d = embedding_data_scaled.shape[1]
        n = n_clusters
        part1 = n * d * np.log(N)
        part2 = N * np.log(RSS / N)
        bic = part1 + part2
        bics.append(bic)
        rss_values.append(RSS)
        bic_part1.append(part1)
        bic_part2.append(part2)

        if len(bics) > early_stop_threshold:
            recent_bics = bics[-early_stop_threshold:]
            if all(recent_bics[i] < recent_bics[i+1] for i in range(len(recent_bics)-1)):
                consecutive_increases += 1
            else:
                consecutive_increases = 0

            if consecutive_increases >= 1:
                print("Early stopping triggered!")
                break

    if not bics:
        return None, None

    optimal_clusters = np.argmin(bics) + min_clusters
    return optimal_clusters, bics

def compute_optimal_clusters_by_aiclike(embedding_data_scaled, max_clusters, min_clusters, early_stop_threshold = 5):
    """
    Calculate the AIC-like value based on K-means clustering (estimated through sum of squared residuals)
    """
    aics = []
    rss_values = []
    aic_part1 = []
    aic_part2 = []
    consecutive_increases = 0

    for n_clusters in range(min_clusters, max_clusters + 1):
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        kmeans.fit(embedding_data_scaled)

        labels = kmeans.labels_
        centers = kmeans.cluster_centers_

        RSS = np.sum((embedding_data_scaled - centers[labels]) ** 2)

        if RSS == 0:
            continue

        N = embedding_data_scaled.shape[0]
        d = embedding_data_scaled.shape[1]
        n = n_clusters
        part1 = 2 * (n * d)
        part2 = N * np.log(RSS / N)
        aic = part1 + part2
        aics.append(aic)
        rss_values.append(RSS)
        aic_part1.append(part1)
        aic_part2.append(part2)

        if len(aics) > early_stop_threshold:
            recent_aics = aics[-early_stop_threshold:]
            if all(recent_aics[i] < recent_aics[i+1] for i in range(len(recent_aics)-1)):
                consecutive_increases += 1
            else:
                consecutive_increases = 0

            if consecutive_increases >= 1:
                print("Early stopping triggered!")
                break

    if not aics:
        return None, None

    optimal_clusters = np.argmin(aics) + min_clusters
    return optimal_clusters, aics

def compute_optimal_clusters_by_bic(embedding_data_scaled, max_clusters, min_clusters, early_stop_threshold = 10):
    """
    Calculate the BIC value based on Gaussian Mixture Model (GMM) clustering
    """
    bics = []
    log_likelihoods = []
    consecutive_increases = 0

    for n_clusters in range(min_clusters, max_clusters + 1):

        gmm = GaussianMixture(n_components=n_clusters, random_state=42, n_init=10)
        gmm.fit(embedding_data_scaled)
        bic = gmm.bic(embedding_data_scaled)
        log_likelihood = gmm.score(embedding_data_scaled) * len(embedding_data_scaled)
        bics.append(bic)
        log_likelihoods.append(log_likelihood)

        print(f"Number of clusters: {n_clusters}, BIC: {bic}, Log-likelihood: {log_likelihood}")

        if len(bics) > early_stop_threshold:
            recent_bics = bics[-early_stop_threshold:]
            if all(recent_bics[i] < recent_bics[i+1] for i in range(len(recent_bics)-1)):
                consecutive_increases += 1
            else:
                consecutive_increases = 0

            if consecutive_increases >= 1:
                print("Early stopping triggered!")
                break

    optimal_clusters = np.argmin(bics) + min_clusters

    return optimal_clusters, bics

def compute_optimal_clusters_by_aic(embedding_data_scaled, max_clusters, min_clusters, early_stop_threshold = 5):
    """
    Calculate the AIC value based on Gaussian Mixture Model (GMM) clustering
    """
    aics = []
    log_likelihoods = []
    consecutive_increases = 0

    for n_clusters in range(min_clusters, max_clusters + 1):
        gmm = GaussianMixture(n_components=n_clusters, random_state=42, n_init=10)
        gmm.fit(embedding_data_scaled)

        aic = gmm.aic(embedding_data_scaled)
        log_likelihood = gmm.score(embedding_data_scaled) * len(embedding_data_scaled)

        aics.append(aic)
        log_likelihoods.append(log_likelihood)

        print(f"Number of clusters: {n_clusters}, AIC: {aic}, Log-likelihood: {log_likelihood}")

        if len(aics) > early_stop_threshold:
            recent_aics = aics[-early_stop_threshold:]
            if all(recent_aics[i] < recent_aics[i+1] for i in range(len(recent_aics)-1)):
                consecutive_increases += 1
            else:
                consecutive_increases = 0

            if consecutive_increases >= 1:
                print("Early stopping triggered!")
                break

    optimal_clusters = np.argmin(aics) + min_clusters

    return optimal_clusters, aics

def compute_optimal_clusters_by_elbow_2(embedding_data_scaled, max_clusters, min_clusters):
    """
    Calculate the optimal number of clusters for K-means clustering based on the elbow method.
    The elbow point is identified by connecting the first and last endpoints, then finding the point in the middle that is below the line and farthest from it.
    """
    wcss = []
    for n_clusters in range(min_clusters, max_clusters + 1):
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        kmeans.fit(embedding_data_scaled)

        wcss.append(kmeans.inertia_)

    plt.figure(figsize=(8, 6))
    plt.plot(range(min_clusters, max_clusters + 1), wcss, marker='o', color='b', linestyle='-', markersize=6)
    plt.title('Elbow Method for Optimal K')
    plt.xlabel('Number of Clusters')
    plt.ylabel('WCSS (Within-cluster sum of squares)')
    plt.grid(True)
    plt.show()

    if len(wcss) <= 3:
        print("Warning: Not enough data points to compute the elbow.")
        return min_clusters, wcss

    start_point = np.array([min_clusters, wcss[0]])
    end_point = np.array([max_clusters, wcss[-1]])

    def point_to_line_distance(point, start, end):
        line_vec = end - start
        point_vec = point - start
        t = np.dot(point_vec, line_vec) / np.dot(line_vec, line_vec)
        t = np.clip(t, 0, 1)
        projection = start + t * line_vec

        if point[1] < projection[1]:
            return np.linalg.norm(point - projection)
        else:
            return 0

    distances = [point_to_line_distance(np.array([i, wcss[i - min_clusters]]), start_point, end_point)
                 for i in range(min_clusters + 1, max_clusters)]

    elbow_point = np.argmax(distances) + min_clusters + 1

    return elbow_point, wcss


def compute_optimal_clusters_by_elbow_gmm(embedding_data_scaled, max_clusters, min_clusters):
    """
    Calculate the optimal number of clusters for GMM clustering based on log-likelihood.
    The elbow point is identified by connecting the first and last endpoints, then finding the point in the middle that is below the line and farthest from it.
    """
    log_likelihoods = []

    for n_clusters in range(min_clusters, max_clusters + 1):
        gmm = GaussianMixture(n_components=n_clusters, random_state=42)
        gmm.fit(embedding_data_scaled)

        log_likelihoods.append(gmm.score(embedding_data_scaled) * len(embedding_data_scaled))

    plt.figure(figsize=(8, 6))
    plt.plot(range(min_clusters, max_clusters + 1), log_likelihoods, marker='o', color='b', linestyle='-', markersize=6)
    plt.title('Elbow Method for Optimal K (using Log-Likelihood)')
    plt.xlabel('Number of Clusters')
    plt.ylabel('Log-Likelihood')
    plt.grid(True)
    plt.show()

    if len(log_likelihoods) <= 3:
        print("Warning: Not enough data points to compute the elbow.")
        return min_clusters, log_likelihoods

    start_point = np.array([min_clusters, log_likelihoods[0]])
    end_point = np.array([max_clusters, log_likelihoods[-1]])

    def point_to_line_distance(point, start, end):
        line_vec = end - start
        point_vec = point - start
        t = np.dot(point_vec, line_vec) / np.dot(line_vec, line_vec)
        t = np.clip(t, 0, 1)
        projection = start + t * line_vec

        if point[1] > projection[1]:
            return np.linalg.norm(point - projection)
        else:
            return 0

    distances = [point_to_line_distance(np.array([i, log_likelihoods[i - min_clusters]]), start_point, end_point)
                 for i in range(min_clusters + 1, max_clusters)]

    elbow_point = np.argmax(distances) + min_clusters + 1
    return elbow_point, log_likelihoods


def compute_optimal_clusters_by_silhouette(embedding_data_scaled, max_clusters, min_clusters):
    """
    Calculate the optimal number of clusters for K-means clustering based on the silhouette score.
    """
    silhouette_scores = []

    for n_clusters in range(min_clusters, max_clusters + 1):
        kmeans = KMeans(n_clusters=n_clusters, init="k-means++", random_state=42)
        kmeans.fit(embedding_data_scaled)
        score = silhouette_score(embedding_data_scaled, kmeans.labels_)
        silhouette_scores.append(score)

    plt.figure(figsize=(8, 6))
    plt.plot(range(min_clusters, max_clusters + 1), silhouette_scores, marker='o', color='b', linestyle='-',
             markersize=6)
    plt.title('Silhouette Score for Optimal K')
    plt.xlabel('Number of Clusters')
    plt.ylabel('Silhouette Score')
    plt.grid(True)
    plt.show()

    optimal_clusters = np.argmax(silhouette_scores) + min_clusters

    return optimal_clusters, silhouette_scores


def compute_optimal_clusters_by_silhouette_gmm(embedding_data_scaled, max_clusters, min_clusters):
    """
    Calculate the optimal number of clusters for GMM clustering based on the silhouette score.
    """
    silhouette_scores = []

    for n_clusters in range(min_clusters, max_clusters + 1):
        gmm = GaussianMixture(n_components=n_clusters, random_state=42)
        gmm.fit(embedding_data_scaled)
        labels = gmm.predict(embedding_data_scaled)
        score = silhouette_score(embedding_data_scaled, labels)
        silhouette_scores.append(score)

        print(f"Number of clusters: {n_clusters}, Silhouette score: {score:.4f}")

    plt.figure(figsize=(8, 6))
    plt.plot(range(min_clusters, max_clusters + 1), silhouette_scores, marker='o', color='b', linestyle='-',
             markersize=6)
    plt.title('Silhouette Score for Optimal K (using GMM)')
    plt.xlabel('Number of Clusters')
    plt.ylabel('Silhouette Score')
    plt.grid(True)
    plt.show()

    optimal_clusters = np.argmax(silhouette_scores) + min_clusters

    return optimal_clusters, silhouette_scores

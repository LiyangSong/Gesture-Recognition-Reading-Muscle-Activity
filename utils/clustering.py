import itertools

import numpy as np
import pandas as pd
import matplotlib as mpl
import umap
from kneed import KneeLocator
from matplotlib import pyplot as plt
from scipy.spatial.distance import pdist, squareform
from scipy.stats import linregress
from sklearn.cluster import KMeans, DBSCAN
import hdbscan.validity as dbcv_hdbscan
from sklearn.manifold import trustworthiness
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score, adjusted_rand_score, \
    rand_score
from sklearn.metrics.cluster import contingency_matrix
from sklearn.neighbors import KDTree
import seaborn as sns


def umap_dim_red(cap_x_df: pd.DataFrame, n_neighbors: int, min_dist: float, metric: str, n_components: int) -> dict:
    """
    Performs UMAP dimensionality reduction.
    Prepares data for clustering by reducing it to a lower-dimensional space
    while preserving its intrinsic structure.
    """

    cap_x_df_copy = cap_x_df.drop(columns='id').copy()

    transformer = umap.UMAP(
        n_neighbors=n_neighbors,
        n_components=n_components,
        metric=metric,
        min_dist=min_dist
    )

    embedding = transformer.fit_transform(cap_x_df_copy)

    twness = trustworthiness(
        squareform(pdist(cap_x_df_copy)),
        squareform(pdist(embedding))
    )

    results_dict = {
        'embedding': embedding,
        'n_neighbors': n_neighbors,
        'min_dist': min_dist,
        'metric': metric,
        'n_components': n_components,
        'trustworthiness': twness
    }

    return results_dict


def clustering(results_dict: dict) -> dict | None:
    """
    Coordinates the clustering process by applying k-means and DBSCAN,
    then selecting the best method based on various metrics.
    """

    embedding = results_dict['embedding'].copy()

    n_clusters_list = list(range(2, 16))
    k_means_internal_indices_df = perform_k_means_and_get_internal_indices(embedding, n_clusters_list)
    n_clusters_found = get_k_means_best_n_clusters(k_means_internal_indices_df)

    n_clusters_db_score_is_min = k_means_internal_indices_df \
        .loc[k_means_internal_indices_df['davies_bouldin_score'].idxmin(), 'n_clusters']
    print(f'n_clusters_db_score_is_min={n_clusters_db_score_is_min}')

    n_clusters_ch_score_is_max = k_means_internal_indices_df \
        .loc[k_means_internal_indices_df['calinski_harabasz_score'].idxmax(), 'n_clusters']
    print(f'n_clusters_ch_score_is_max={n_clusters_ch_score_is_max}')

    n_clusters_silhouette_score_is_max = k_means_internal_indices_df \
        .loc[k_means_internal_indices_df['silhouette_score'].idxmax(), 'n_clusters']
    print(f'n_clusters_silhouette_score_is_max={n_clusters_silhouette_score_is_max}')

    test_result = test_k_means_result(n_clusters_found, n_clusters_db_score_is_min,
                           n_clusters_ch_score_is_max, n_clusters_silhouette_score_is_max)

    if test_result != 0:
        n_clusters_found = n_clusters_db_score_is_min if test_result == 2 else n_clusters_found
        index_found = k_means_internal_indices_df[k_means_internal_indices_df['n_clusters'] == n_clusters_found].index

        return {
            'algo': 'k_means',
            'n_clusters_found': n_clusters_found,
            'n_clusters_db_score_is_min': n_clusters_db_score_is_min,
            'n_clusters_ch_score_is_max': n_clusters_ch_score_is_max,
            'n_clusters_silhouette_score_is_max': n_clusters_silhouette_score_is_max,
            'silhouette_score': k_means_internal_indices_df.loc[index_found, 'silhouette_score'].values[0],
            'hopkins_statistic': k_means_internal_indices_df.loc[index_found, 'hopkins_statistic'].values[0],
            'umap_n_neighbors': results_dict['n_neighbors'],
            'umap_min_dist': results_dict['min_dist'],
            'umap_metric': results_dict['metric'],
            'umap_n_components': results_dict['n_components'],
            'trustworthiness': results_dict['trustworthiness'],
            'fitted_k_means': k_means_internal_indices_df.loc[index_found, 'fitted_k_means'].values[0],
            'embedding': embedding,
            'cluster_labels': k_means_internal_indices_df.loc[index_found, 'cluster_labels'].values[0]
        }

    else:
        print(f'\nTry applying DBSCAN method:')

        embedding = results_dict['embedding'].copy()
        k_list = [3, 4, 5, 6]
        eps_factor_list = np.arange(0.5, 1.8, 0.1)

        dbscan_internal_indices_df = perform_dbscan_and_get_internal_indices(
            embedding,
            k_list,
            eps_factor_list,
            results_dict['metric']
        )

        index_found = get_dbscan_best_index(dbscan_internal_indices_df)

        if index_found is None:
            return None

        n_clusters_found = dbscan_internal_indices_df.loc[index_found, 'n_clusters']
        eps = dbscan_internal_indices_df.loc[index_found, 'f_eps']
        dbscan_min_samples = dbscan_internal_indices_df.loc[index_found, 'min_samples']

        print(
            f'\033[92mSucceed\033[0m to find n_clusters={n_clusters_found}, eps={eps}, min_samples={dbscan_min_samples}')

        return {
            'algo': 'dbscan',
            'eps': eps,
            'dbscan_min_samples': dbscan_min_samples,
            'n_clusters_found': n_clusters_found,
            'validity_index': dbscan_internal_indices_df.loc[index_found, 'validity_index'],
            'hopkins_statistic': dbscan_internal_indices_df.loc[index_found, 'hopkins_statistic'],
            'umap_n_neighbors': results_dict['n_neighbors'],
            'umap_min_dist': results_dict['min_dist'],
            'umap_metric': results_dict['metric'],
            'umap_n_components': results_dict['n_components'],
            'trustworthiness': results_dict['trustworthiness'],
            'fitted_dbscan': dbscan_internal_indices_df.loc[index_found, 'fitted_dbscan'],
            'embedding': embedding,
            'cluster_labels': dbscan_internal_indices_df.loc[index_found, 'cluster_labels']
        }


def perform_k_means_and_get_internal_indices(cap_x: np.ndarray, n_clusters_list: list[int],
                                             random_state: int = 42) -> pd.DataFrame:
    """
    Performs k-means clustering and return the internal indices dataframe.
    """

    internal_indices = []

    for n_clusters in n_clusters_list:
        cap_x_copy = cap_x.copy()
        kmeans = KMeans(
            n_clusters=n_clusters,
            n_init='auto',
            random_state=random_state
        )
        kmeans.fit_predict(cap_x_copy)

        internal_indices.append({
            'n_clusters': n_clusters,
            'inertia': kmeans.inertia_,
            'calinski_harabasz_score': calinski_harabasz_score(cap_x_copy, kmeans.labels_),
            'davies_bouldin_score': davies_bouldin_score(cap_x_copy, kmeans.labels_),
            'silhouette_score': silhouette_score(cap_x_copy, kmeans.labels_),
            'hopkins_statistic': get_hopkins_statistic(cap_x_copy),
            'fitted_k_means': kmeans,
            'cluster_labels': kmeans.labels_
        })

    return pd.DataFrame(internal_indices)


def get_k_means_best_n_clusters(k_means_internal_indices_df: pd.DataFrame, plot: bool = False) -> int | None:
    """
    Identifies the optimal number of clusters for k-means clustering using the elbow method.
    """

    n_clusters_found = KneeLocator(
        x=k_means_internal_indices_df['n_clusters'],
        y=k_means_internal_indices_df['inertia'],
        S=1.0,
        curve='convex',
        direction='decreasing'
    ).elbow

    if plot:
        sns.lineplot(k_means_internal_indices_df, x='n_clusters', y='inertia')
        plt.axvline(n_clusters_found, color="r", linestyle="--")
        plt.grid()
        plt.show()

    if n_clusters_found is not None:
        print(f'\033[92mSucceed\033[0m to find an elbow at {n_clusters_found} in the inertia curve')
        if test_elbow_slope_change(k_means_internal_indices_df, n_clusters_found) != 0:
            n_clusters_found = None
    else:
        print('\033[91mFailed\033[0m to find an elbow in the inertia curve')

    return n_clusters_found


def test_k_means_result(
        n_clusters_found: int,
        n_clusters_db_score_is_min: int,
        n_clusters_ch_score_is_max: int,
        n_clusters_silhouette_score_is_max: int
) -> int:
    """
    Validates the k-means clustering results by comparing the found number of clusters against other metrics.
    """

    if n_clusters_found is not None and \
            n_clusters_found == n_clusters_db_score_is_min and \
            n_clusters_found == n_clusters_ch_score_is_max and \
            n_clusters_found == n_clusters_silhouette_score_is_max:
        print('K-Means first test \033[92mpassed\033[0m.')
        return 1
    elif n_clusters_db_score_is_min == n_clusters_ch_score_is_max and \
            n_clusters_db_score_is_min == n_clusters_silhouette_score_is_max:
        print('K-Means first test \033[91mfailed\033[0m')
        print('K-Means second test \033[92mpassed\033[0m')
        return 2
    else:
        print('K-Means first test \033[91mfailed\033[0m')
        print('K-Means second test \033[91mfailed\033[0m')
        return 0


def test_elbow_slope_change(results_df: pd.DataFrame, n_clusters: int, min_elbow_slope_diff: int = 30) -> int:
    """
    Tests the significance of the change in inertia slope at the elbow point to validate k-means results.
    """

    n_clusters_m1 = results_df.loc[results_df.n_clusters == n_clusters - 1, :].values.flatten()
    n_clusters_at_elbow = results_df.loc[results_df.n_clusters == n_clusters, :].values.flatten()
    n_clusters_p1 = results_df.loc[results_df.n_clusters == n_clusters + 1, :].values.flatten()

    slope_before_elbow = linregress(
        [n_clusters_m1[0], n_clusters_at_elbow[0]],
        [n_clusters_m1[1], n_clusters_at_elbow[1]]
    ).slope

    slope_after_elbow = linregress(
        [n_clusters_at_elbow[0], n_clusters_p1[0]],
        [n_clusters_at_elbow[1], n_clusters_p1[1]]
    ).slope

    if (slope_after_elbow - slope_before_elbow) < min_elbow_slope_diff:
        print("Elbow slope change test \033[91mfailed\033[0m ")
        return -1
    else:
        print("Elbow slope change test \033[92mpassed\033[0m")
        return 0


def get_hopkins_statistic(cap_x: np.ndarray) -> float:
    """
    Calculates the Hopkins statistic to measure the clusterability of the dataset.
    """

    randomly_distributed_data = get_randomly_distributed_data(cap_x)
    cap_x_nn_dist_list = get_nearest_neighbor_distance(cap_x, 1)
    randomly_distributed_data_nn_dist_list = get_nearest_neighbor_distance(randomly_distributed_data, 1)

    return sum(cap_x_nn_dist_list) / (sum(randomly_distributed_data_nn_dist_list) + sum(cap_x_nn_dist_list))


def get_randomly_distributed_data(cap_x: np.ndarray, random_state: int = 42) -> np.ndarray:
    """
    Generates a uniformly distributed dataset based on the range of the original data.
    """

    data_max = cap_x.max(axis=0)
    data_min = cap_x.min(axis=0)

    np.random.seed(random_state)
    rand_x = np.random.uniform(low=data_min[0], high=data_max[0], size=(cap_x.shape[0], 1))
    rand_y = np.random.uniform(low=data_min[1], high=data_max[1], size=(cap_x.shape[0], 1))

    return np.concatenate((rand_x, rand_y), axis=1)


def get_nearest_neighbor_distance(cap_x: np.ndarray, k: int) -> list:
    """
    Calculates the distance to the kth nearest neighbor for each point in the dataset.
    """

    nn_dist_list = []
    kdt = KDTree(cap_x, metric='l2')

    for i in range(cap_x.shape[0]):
        dist, _ = kdt.query(cap_x[i, :].reshape(1, -1), k + 1)
        nn_dist_list.append(dist[0, -1])

    return nn_dist_list


def perform_dbscan_and_get_internal_indices(
        cap_x: np.ndarray,
        k_list: list,
        eps_factor_list: np.ndarray,
        metric: str = 'euclidean') -> pd.DataFrame:
    """
    Performs DBSCAN clustering and return the internal indices dataframe.
    """

    internal_indices = []

    eps_k_df = get_eps_k_df(cap_x, k_list)
    max_eps = eps_k_df.eps.values.max()
    min_samples = eps_k_df.loc[eps_k_df.eps == max_eps, 'k'].values[0]
    print(eps_k_df)
    print(f'max_eps: {max_eps}, min_samples: {min_samples}')

    for factor in eps_factor_list:
        cap_x_copy = cap_x.copy()
        f_eps = factor * max_eps

        metric = 'cityblock' if metric == 'manhattan' else 'euclidean'

        dbscan = DBSCAN(
            eps=f_eps,
            min_samples=min_samples,
            metric=metric
        )
        dbscan.fit(cap_x_copy)

        clusters = np.unique(dbscan.labels_)
        n_clusters = clusters[clusters != -1].shape[0]
        if n_clusters < 1:
            continue

        try:
            validity_index = dbcv_hdbscan.validity_index(cap_x_copy.astype(np.float64), dbscan.labels_)
        except ValueError as e:
            print(e)
            validity_index = np.nan

        if np.isnan(validity_index):
            continue

        internal_indices.append({
            'max_eps': max_eps,
            'f_eps': f_eps,
            'min_samples': min_samples,
            'n_clusters': n_clusters,
            'validity_index': validity_index,
            'hopkins_statistic': get_hopkins_statistic(cap_x_copy),
            'fitted_dbscan': dbscan,
            'cluster_labels': dbscan.labels_
        })

    return pd.DataFrame(internal_indices)


def get_dbscan_best_index(dbscan_internal_indices_df: pd.DataFrame) -> int | None:
    """
    Determines the best DBSCAN result based on the validity index.
    """

    try:
        n_clusters_max = dbscan_internal_indices_df \
            .loc[dbscan_internal_indices_df['validity_index'].idxmin(), 'n_clusters']
        dbscan_internal_indices_df = dbscan_internal_indices_df[
            dbscan_internal_indices_df['n_clusters'] <= n_clusters_max]
        index_found = dbscan_internal_indices_df['validity_index'].idxmax()
    except KeyError:
        print(f'\033[91mFailed\033[0m to find valid n_clusters')
        index_found = None

    return index_found


def get_eps_k_df(cap_x: np.ndarray, k_list: list[int], drop_first_percent: float = 0.10, plot: bool = False) -> pd.DataFrame:
    """
    Determines the optimal epsilon value for DBSCAN by analyzing the distribution of kth nearest neighbor distances.
    """

    nn_dist_kth_dict = {}
    df_row_dict_list = []
    for k in k_list:
        nn_dist_kth_dict[k] = sorted(get_nearest_neighbor_distance(cap_x, k))
        y = nn_dist_kth_dict[k][int(drop_first_percent * cap_x.shape[0]):]
        eps_idx = KneeLocator(
            x=list(range(len(y))),
            y=y,
            S=3,
            curve='convex',
            direction='increasing'
        ).knee
        eps = y[eps_idx]

        index = int(drop_first_percent * cap_x.shape[0]) + eps_idx
        df_row_dict_list.append({
            'index': index,
            'k': k,
            'eps': eps
        })

        if plot:
            label = f'k={k} eps={eps:.2f}'
            line = sns.lineplot(
                x=list(range(len(nn_dist_kth_dict[k]))),
                y=nn_dist_kth_dict[k],
                label=label
            )
            line_color = line.lines[-1].get_color()
            plt.axvline(index, c=line_color, linestyle="--")

    if plot:
        plt.ylabel('eps')
        plt.xlabel('kth neighbor distances')
        plt.legend()
        plt.grid()
        plt.show()

    return pd.DataFrame(df_row_dict_list)


def select_and_plot_best_k_means(results_df: pd.DataFrame) -> any:
    """
    Selects the best k-means clustering result based on silhouette score and generates a plot.
    """

    best_k_means_index = results_df[results_df['algo'] == 'k_means']['silhouette_score'].idxmax()
    best_k_means = results_df.loc[best_k_means_index, 'fitted_k_means']
    best_k_means_embedding = results_df.loc[best_k_means_index, 'embedding']

    if results_df.loc[best_k_means_index, 'umap_n_components'] == 2:
        plot_k_means(best_k_means, best_k_means_embedding)

    return best_k_means, best_k_means_index


def plot_k_means(k_means: any, cap_x: np.ndarray) -> None:
    """
    Plots the k-means clustering results, showing clusters and their centroids.
    """

    labels = k_means.labels_
    centroids = k_means.cluster_centers_
    n_clusters = len(np.unique(labels))

    for cluster in range(n_clusters):
        cluster_points = cap_x[labels == cluster]

        plt.scatter(
            cluster_points[:, 0],
            cluster_points[:, 1],
            label=f'Cluster {cluster}',
            s=20
        )

    plt.scatter(
        centroids[:, 0],
        centroids[:, 1],
        marker='x',
        s=100,
        c='black',
        label='Centroids'
    )

    plt.xlabel('$x_1$')
    plt.ylabel('$x_2$', rotation=0)
    plt.title(f'n_clusters={n_clusters}\nK-Means Clustering')
    plt.grid(True)
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.show()


def select_and_plot_best_dbscan(results_df: pd.DataFrame) -> any:
    """
    Selects the best DBSCAN clustering result based on validity index and generates a plot.
    """

    best_dbscan_index = results_df[results_df['algo'] == 'dbscan']['validity_index'].idxmax()
    best_dbscan = results_df.loc[best_dbscan_index, 'fitted_dbscan']
    best_dbscan_embedding = results_df.loc[best_dbscan_index, 'embedding']

    if results_df.loc[best_dbscan_index, 'umap_n_components'] == 2:
        plot_dbscan(best_dbscan, best_dbscan_embedding)

    return best_dbscan, best_dbscan_index


def plot_dbscan(dbscan: any, cap_x: np.ndarray) -> None:
    """
    Plots the DBSCAN clustering results, highlighting core, border, and noise points.
    """

    core_mask = np.zeros_like(dbscan.labels_, dtype=bool)
    core_mask[dbscan.core_sample_indices_] = True
    anomalies_mask = dbscan.labels_ == -1
    border_mask = ~(core_mask | anomalies_mask)

    cores = dbscan.components_
    anomalies = cap_x[anomalies_mask]
    non_cores = cap_x[border_mask]

    if cores.shape[0] > 0:
        plt.scatter(cores[:, 0], cores[:, 1], marker='*', s=20, c=dbscan.labels_[core_mask], label='core')

    if anomalies.shape[0] > 0:
        plt.scatter(anomalies[:, 0], anomalies[:, 1], c="r", marker="x", s=100, label='anomalies')

    if non_cores.shape[0] > 0:
        plt.scatter(non_cores[:, 0], non_cores[:, 1], c=dbscan.labels_[border_mask], marker=".", label='border')

    clusters = np.unique(dbscan.labels_)
    n_clusters = len(clusters[clusters != -1])

    plt.xlabel("$x_1$")
    plt.ylabel("$x_2$", rotation=0)
    plt.title(f"eps={dbscan.eps:.2f}, min_samples={dbscan.min_samples}, n_clusters={n_clusters}\nDBSCAN Clustering")
    plt.grid()
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.show()


def select_latent_manifold(results_df: pd.DataFrame, best_k_means_index: int, best_dbscan_index: int):
    """
    Selects the best clustering result between k-means and DBSCAN based on clustering metrics.
    """

    if best_k_means_index is None:
        return best_dbscan_index

    if best_dbscan_index is None:
        return best_k_means_index

    return best_k_means_index if results_df.loc[best_k_means_index, 'silhouette_score'] > \
                                 results_df.loc[best_dbscan_index, 'validity_index'] else best_dbscan_index


def get_latent_manifold_details(results_df: pd.DataFrame, best_index: int) -> pd.DataFrame:
    """
    Retrieves details of the selected clustering result for further analysis or reporting.
    """

    try:
        n_clusters_db_score_is_min = results_df.loc[best_index, 'n_clusters_db_score_is_min']
        n_clusters_ch_score_is_max = results_df.loc[best_index, 'n_clusters_ch_score_is_max']
        n_clusters_silhouette_score_is_max = results_df.loc[best_index, 'n_clusters_silhouette_score_is_max']
        silhouette_score = results_df.loc[best_index, 'silhouette_score']
    except KeyError:
        n_clusters_db_score_is_min = None
        n_clusters_ch_score_is_max = None
        n_clusters_silhouette_score_is_max = None
        silhouette_score = None

    try:
        eps = results_df.loc[best_index, 'eps']
        dbscan_min_samples = results_df.loc[best_index, 'dbscan_min_samples']
        validity_index = results_df.loc[best_index, 'validity_index']
    except KeyError:
        eps = None
        dbscan_min_samples = None
        validity_index = None

    return pd.DataFrame([{
        'number of classes in the digits data set': 10,
        'UMAP n_components': results_df.loc[best_index, 'umap_n_components'],
        'UMAP min_dist': results_df.loc[best_index, 'umap_min_dist'],
        'UMAP n_neighbors': results_df.loc[best_index, 'umap_n_neighbors'],
        'UMAP metric': results_df.loc[best_index, 'umap_metric'],
        'trustworthiness': results_df.loc[best_index, 'trustworthiness'],
        'clustering algorithm': results_df.loc[best_index, 'algo'],
        'number of clusters found': results_df.loc[best_index, 'n_clusters_found'],
        'validity index or silhouette score': validity_index if results_df.loc[best_index, 'algo'] == 'dbscan' else silhouette_score,
        # 'n_clusters_db_score_is_min': n_clusters_db_score_is_min,
        # 'n_clusters_ch_score_is_max': n_clusters_ch_score_is_max,
        # 'n_clusters_silhouette_score_is_max': n_clusters_silhouette_score_is_max,
        # 'hopkins_statistic': results_df.loc[best_index, 'hopkins_statistic'],
        # 'eps': eps,
        # 'dbscan_min_samples': dbscan_min_samples,
        # 'validity_index': validity_index
    }])


def get_external_indices(labels_true: np.ndarray, labels_pred: np.ndarray, algo: str) -> pd.DataFrame:

    labels_true_copy = labels_true.copy()
    labels_pred_copy = labels_pred.copy()
    non_noise_indices = None

    if algo == 'dbscan':
        return_dict = remove_noise_data_objects_from_labels(labels_true_copy, labels_pred_copy)
        labels_true_copy = return_dict['labels_true']
        labels_pred_copy = return_dict['labels_pred']
        non_noise_indices = return_dict['non_noise_indices']

    rand_score_ = rand_score(labels_true_copy, labels_pred_copy)
    print('rand_score: ', rand_score_)

    adjusted_rand_score_ = adjusted_rand_score(labels_true_copy, labels_pred_copy)
    print('adjusted_rand_score: ', adjusted_rand_score_)

    if len(labels_pred_copy) <= 5:
        best_cluster_label_permutation = get_best_cluster_label_permutation(labels_true_copy, labels_pred_copy)
    else:
        best_cluster_label_permutation = get_approx_best_cluster_label_permutation(labels_true_copy, labels_pred_copy)

    print('best_contingency_matrix:\n', best_cluster_label_permutation['best_contingency_matrix'])

    return pd.DataFrame([{
        'rand_score': rand_score_,
        'adjusted_rand_score': adjusted_rand_score_,
        'best_perm_labels_pred': best_cluster_label_permutation['best_perm_labels_pred'],
        'best_contingency_matrix': best_cluster_label_permutation['best_contingency_matrix'],
        'non_noise_indices': non_noise_indices
    }])


def get_best_cluster_label_permutation(labels_true: np.ndarray, labels_pred: np.ndarray) -> dict:
    best_permutation_mapping = None
    best_perm_labels_pred = None
    best_contingency_matrix = None

    max_contingency_matrix_trace = 0
    for cluster_labels in itertools.permutations(set(labels_pred)):

        mapping = dict(zip(set(labels_pred), cluster_labels))
        perm_labels_pred = [mapping[label] for label in labels_pred]

        contingency_matrix_ = contingency_matrix(labels_true, perm_labels_pred)
        contingency_matrix_trace = np.trace(contingency_matrix_)

        if contingency_matrix_trace > max_contingency_matrix_trace:
            max_contingency_matrix_trace = contingency_matrix_trace
            best_permutation_mapping = mapping
            best_perm_labels_pred = perm_labels_pred
            best_contingency_matrix = contingency_matrix_

    return {
        'best_permutation_mapping': best_permutation_mapping,
        'best_perm_labels_pred': best_perm_labels_pred,
        'best_contingency_matrix': best_contingency_matrix
    }


def get_approx_best_cluster_label_permutation(labels_true: np.ndarray, labels_pred: np.ndarray) -> dict:
    """
    This algorithm assume the max count in one contingency matrix represents
    the largest group of items that have the same true label and were assigned the same predicted label.
    It iteratively removes current largest count in contingency matrix to find a
    relatively good permutation result, which has less computational complexity than
    the exhaustive permutation method.
    """

    best_permutation_mapping = {}
    best_perm_labels_pred = np.full_like(labels_pred, fill_value=-1)

    # Compute the initial contingency matrix
    initial_matrix = contingency_matrix(labels_true, labels_pred)
    true_labels = np.unique(labels_true)
    pred_labels = np.unique(labels_pred)

    for _ in range(min(len(true_labels), len(pred_labels))):
        # Find the indices of the max count in the contingency matrix
        true_idx, pred_idx = divmod(initial_matrix.argmax(), initial_matrix.shape[1])

        # The corresponding true and pred label at the max count position
        true_label = true_labels[true_idx]
        pred_label = pred_labels[pred_idx]

        # Update best_permutation_mapping and best_perm_labels_pred
        best_permutation_mapping[pred_label] = true_label
        best_perm_labels_pred[labels_pred == pred_label] = true_label

        # Zero out the current row and column
        initial_matrix[true_idx, :] = 0
        initial_matrix[:, pred_idx] = 0

    # Get final best_contingency matrix
    best_contingency_matrix = contingency_matrix(labels_true, best_perm_labels_pred)

    return {
        'best_permutation_mapping': best_permutation_mapping,
        'best_perm_labels_pred': best_perm_labels_pred,
        'best_contingency_matrix': best_contingency_matrix
    }


def remove_noise_data_objects_from_labels(labels_true: np.ndarray, labels_pred: np.ndarray) -> dict:
    labels_true = labels_true.reshape(-1, 1)
    labels_pred = labels_pred.reshape(-1, 1)
    labels = np.concatenate((labels_true, labels_pred), axis=1)

    labels = labels[~np.any(labels == -1, axis=1), :]

    return {
        'labels_true': labels[:, 0],
        'labels_pred': labels[:, 1],
        'non_noise_indices': np.where(labels_pred != -1)[0]
    }


def plot_digits(digits: np.ndarray, pred_labels: np.ndarray, digits_per_row: int = 10) -> None:
    n_digits = len(digits)
    n_cols = min(n_digits, digits_per_row)
    n_rows = (n_digits - 1) // digits_per_row + 1

    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(n_cols * 1.5, n_rows * 1.8),
        squeeze=False
    )
    axes = axes.flatten()

    for i in range(n_digits):
        ax = axes[i]
        digit = digits[i].reshape(8, 8)
        ax.imshow(digit, cmap=mpl.cm.binary)
        ax.set_title(f'Pred: {pred_labels[i]}')
        ax.axis('off')

    # Remove empty subplots at the end
    for ax in axes[n_digits:]:
        ax.remove()

    plt.tight_layout()
    plt.show()


def plot_digits_by_true_label(merged_df: pd.DataFrame, target: str):

    misclassified = merged_df[merged_df[target] != merged_df['pred_labels']]
    unique_labels = misclassified[target].unique()

    for label in unique_labels:
        instances = misclassified[merged_df[target] == label]

        most_common_pred = instances['pred_labels'].value_counts().idxmax()
        print(f"\nTrue label: {label}, most common incorrect pred: {most_common_pred}")

        digits = instances.drop(columns=['id', target, 'pred_labels']).values
        if len(digits) > 50:
            digits = digits[:50]
        pred_labels = instances['pred_labels'].values
        plot_digits(digits, pred_labels)

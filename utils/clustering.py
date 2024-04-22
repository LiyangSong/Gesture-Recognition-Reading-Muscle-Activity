import numpy as np
import pandas as pd
import umap
from kneed import KneeLocator
from matplotlib import pyplot as plt
from scipy.spatial.distance import pdist, squareform
from scipy.stats import linregress
from sklearn.cluster import KMeans, DBSCAN
import hdbscan.validity as dbcv_hdbscan
from sklearn.manifold import trustworthiness
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.neighbors import KDTree
import seaborn as sns


def umap_dim_red(
        cap_x_df: pd.DataFrame,
        n_neighbors: int = 15,
        min_dist: float = 0.1,
        metric: str = 'euclidean',
        n_components: int = 2) -> dict:
    """
    Performs UMAP dimensionality reduction.
    Prepares data for clustering by reducing it to a lower-dimensional space
    while preserving its intrinsic structure.
    """

    cap_x_df_copy = cap_x_df.copy()

    transformer = umap.UMAP(
        n_neighbors=n_neighbors,
        n_components=n_components,
        metric=metric,
        min_dist=min_dist
    )

    embedding = transformer.fit_transform(cap_x_df_copy)

    twness = trustworthiness(
        X=squareform(pdist(cap_x_df_copy)),
        X_embedded=squareform(pdist(embedding)),
        metric=metric
    )

    umap_results_dict = {
        'embedding': embedding,
        'n_neighbors': n_neighbors,
        'min_dist': min_dist,
        'metric': metric,
        'n_components': n_components,
        'trustworthiness': twness
    }

    return umap_results_dict


def clustering(umap_results_dict: dict) -> dict | None:
    """
    Coordinates the clustering process by applying k-means and DBSCAN,
    then selecting the best method based on various metrics.
    """

    embedding = umap_results_dict['embedding'].copy()
    k_means_internal_indices_df = perform_k_means_and_get_internal_indices(embedding)
    k_means_clustering_results_dict = get_k_means_clustering_results_dict(
        k_means_internal_indices_df,
        umap_results_dict
    )

    if k_means_clustering_results_dict is not None:
        return k_means_clustering_results_dict

    else:
        # k_means results not pass, try DBSCAN
        print(f'\nTry applying DBSCAN method:')

        embedding = umap_results_dict['embedding'].copy()
        dbscan_internal_indices_df = perform_dbscan_and_get_internal_indices(
            embedding,
            metric=umap_results_dict['metric']
        )
        dbscan_clustering_results_dict = get_dbscan_clustering_results_dict(
            dbscan_internal_indices_df,
            umap_results_dict
        )

        return dbscan_clustering_results_dict


def perform_k_means_and_get_internal_indices(
        embedding: np.ndarray,
        n_clusters_list: list = None,
        random_state: int = 42) -> pd.DataFrame:
    """
    Performs k-means clustering and return the internal indices dataframe.
    """

    if n_clusters_list is None:
        n_clusters_list = list(np.arange(2, 16))

    internal_indices = []

    for n_clusters in n_clusters_list:
        embedding_copy = embedding.copy()
        kmeans = KMeans(
            n_clusters=n_clusters,
            n_init='auto',
            random_state=random_state
        )
        kmeans.fit_predict(embedding_copy)

        internal_indices.append({
            'n_clusters': n_clusters,
            'inertia': kmeans.inertia_,
            'calinski_harabasz_score': calinski_harabasz_score(embedding_copy, kmeans.labels_),
            'davies_bouldin_score': davies_bouldin_score(embedding_copy, kmeans.labels_),
            'silhouette_score': silhouette_score(embedding_copy, kmeans.labels_),
            'hopkins_statistic': get_hopkins_statistic(embedding_copy),
            'fitted_k_means': kmeans,
            'cluster_labels': kmeans.labels_
        })

    return pd.DataFrame(internal_indices)


def get_k_means_clustering_results_dict(k_means_internal_indices_df: pd.DataFrame, umap_results_dict: dict) -> dict | None:
    """
    Generate results dict based on the best k_means clustering index.
    """

    best_index_results_dict = get_k_means_best_index(k_means_internal_indices_df)

    index_found = best_index_results_dict['index_found']
    if index_found is None:
        return None

    k_means_clustering_results_dict = {
        'algo': 'k_means',
        'n_clusters_found': best_index_results_dict['n_clusters_found'],
        'n_clusters_db_score_is_min': best_index_results_dict['n_clusters_db_score_is_min'],
        'n_clusters_ch_score_is_max': best_index_results_dict['n_clusters_ch_score_is_max'],
        'n_clusters_silhouette_score_is_max': best_index_results_dict['n_clusters_silhouette_score_is_max'],
        'silhouette_score': k_means_internal_indices_df.loc[index_found, 'silhouette_score'].values[0],
        'hopkins_statistic': k_means_internal_indices_df.loc[index_found, 'hopkins_statistic'].values[0],
        'umap_n_neighbors': umap_results_dict['n_neighbors'],
        'umap_min_dist': umap_results_dict['min_dist'],
        'umap_metric': umap_results_dict['metric'],
        'umap_n_components': umap_results_dict['n_components'],
        'trustworthiness': umap_results_dict['trustworthiness'],
        'fitted_k_means': k_means_internal_indices_df.loc[index_found, 'fitted_k_means'].values[0],
        'embedding': umap_results_dict['embedding'].copy(),
        'cluster_labels': k_means_internal_indices_df.loc[index_found, 'cluster_labels'].values[0]
    }

    return k_means_clustering_results_dict


def get_k_means_best_index(k_means_internal_indices_df: pd.DataFrame):
    """
    Determines the best k_means result based on the inertia elbow and other internal indices.
    """

    index_found = None
    n_clusters_found = None

    # Computer n_clusters based on Knee of the inertia curve
    n_clusters_elbow = get_k_means_elbow_n_clusters(k_means_internal_indices_df)

    # Computer n_clusters based on internal indices
    n_clusters_db_score_is_min = k_means_internal_indices_df \
        .loc[k_means_internal_indices_df['davies_bouldin_score'].idxmin(), 'n_clusters']
    print(f'n_clusters_db_score_is_min={n_clusters_db_score_is_min}')

    n_clusters_ch_score_is_max = k_means_internal_indices_df \
        .loc[k_means_internal_indices_df['calinski_harabasz_score'].idxmax(), 'n_clusters']
    print(f'n_clusters_ch_score_is_max={n_clusters_ch_score_is_max}')

    n_clusters_silhouette_score_is_max = k_means_internal_indices_df \
        .loc[k_means_internal_indices_df['silhouette_score'].idxmax(), 'n_clusters']
    print(f'n_clusters_silhouette_score_is_max={n_clusters_silhouette_score_is_max}')

    # Test n_clusters found through k_means
    test_result = test_k_means_result(n_clusters_elbow, n_clusters_db_score_is_min,
                                      n_clusters_ch_score_is_max, n_clusters_silhouette_score_is_max)

    if test_result != 0:
        n_clusters_found = n_clusters_db_score_is_min if test_result == 2 else n_clusters_elbow
        index_found = k_means_internal_indices_df[k_means_internal_indices_df['n_clusters'] == n_clusters_found].index

    return {
        'n_clusters_elbow': n_clusters_elbow,
        'n_clusters_db_score_is_min': n_clusters_db_score_is_min,
        'n_clusters_ch_score_is_max': n_clusters_ch_score_is_max,
        'n_clusters_silhouette_score_is_max': n_clusters_silhouette_score_is_max,
        'n_clusters_found': n_clusters_found,
        'index_found': index_found
    }


def get_k_means_elbow_n_clusters(k_means_internal_indices_df: pd.DataFrame, plot: bool = False) -> int | None:
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
    cap_x_nn_dist_list = get_nearest_neighbor_distance(cap_x, k=1)
    randomly_distributed_data_nn_dist_list = get_nearest_neighbor_distance(randomly_distributed_data, k=1)

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
        embedding: np.ndarray,
        k_list: list[int] = None,
        eps_factor_list: list[float] = None,
        metric: str = 'euclidean') -> pd.DataFrame:
    """
    Performs DBSCAN clustering and return the internal indices dataframe.
    """

    if k_list is None:
        k_list = [3, 4, 5, 6]

    if eps_factor_list is None:
        eps_factor_list = list(np.arange(0.5, 1.8, 0.2))

    internal_indices = []

    eps_k_df = get_eps_k_df(embedding, k_list)
    max_eps = eps_k_df['eps'].values.max()
    min_samples = eps_k_df.loc[eps_k_df.eps == max_eps, 'k'].values[0]

    print(eps_k_df)
    print(f'max_eps: {max_eps}, min_samples: {min_samples}')

    for factor in eps_factor_list:
        embedding_copy = embedding.copy()
        f_eps = factor * max_eps

        metric = 'cityblock' if metric == 'manhattan' else 'euclidean'

        dbscan = DBSCAN(
            eps=f_eps,
            min_samples=min_samples,
            metric=metric
        )
        dbscan.fit(embedding_copy)

        clusters = np.unique(dbscan.labels_)
        n_clusters = clusters[clusters != -1].shape[0]
        if n_clusters < 1:
            continue

        try:
            validity_index = dbcv_hdbscan.validity_index(
                X=embedding_copy.astype(np.float64),
                labels=dbscan.labels_,
                metric=metric
            )
        except ValueError:
            validity_index = np.nan

        if np.isnan(validity_index):
            continue

        internal_indices.append({
            'max_eps': max_eps,
            'f_eps': f_eps,
            'min_samples': min_samples,
            'n_clusters': n_clusters,
            'validity_index': validity_index,
            'hopkins_statistic': get_hopkins_statistic(embedding_copy),
            'fitted_dbscan': dbscan,
            'cluster_labels': dbscan.labels_
        })

    return pd.DataFrame(internal_indices)


def get_dbscan_clustering_results_dict(dbscan_internal_indices_df: pd.DataFrame, umap_results_dict: dict) -> dict | None:
    """
    Generate results dict based on the best DBSCAN clustering index.
    """

    index_found = get_dbscan_best_index(dbscan_internal_indices_df)
    if index_found is None:
        return None

    n_clusters_found = dbscan_internal_indices_df.loc[index_found, 'n_clusters']
    eps = dbscan_internal_indices_df.loc[index_found, 'f_eps']
    dbscan_min_samples = dbscan_internal_indices_df.loc[index_found, 'min_samples']

    print(f'\033[92mSucceed\033[0m to find n_clusters={n_clusters_found}, eps={eps}, min_samples={dbscan_min_samples}')

    dbscan_clustering_results_dict = {
        'algo': 'dbscan',
        'eps': eps,
        'dbscan_min_samples': dbscan_min_samples,
        'n_clusters_found': n_clusters_found,
        'validity_index': dbscan_internal_indices_df.loc[index_found, 'validity_index'],
        'hopkins_statistic': dbscan_internal_indices_df.loc[index_found, 'hopkins_statistic'],
        'umap_n_neighbors': umap_results_dict['n_neighbors'],
        'umap_min_dist': umap_results_dict['min_dist'],
        'umap_metric': umap_results_dict['metric'],
        'umap_n_components': umap_results_dict['n_components'],
        'trustworthiness': umap_results_dict['trustworthiness'],
        'fitted_dbscan': dbscan_internal_indices_df.loc[index_found, 'fitted_dbscan'],
        'embedding': umap_results_dict['embedding'].copy(),
        'cluster_labels': dbscan_internal_indices_df.loc[index_found, 'cluster_labels']
    }

    return dbscan_clustering_results_dict


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


def select_best_k_means_from_clustering_results(clustering_results_df: pd.DataFrame) -> (KMeans, int):
    """
    Selects the best k-means clustering result based on silhouette score and generates a plot.
    """

    best_k_means_index = clustering_results_df[clustering_results_df['algo'] == 'k_means']['silhouette_score'].idxmax()
    best_k_means = clustering_results_df.loc[best_k_means_index, 'fitted_k_means']
    best_k_means_embedding = clustering_results_df.loc[best_k_means_index, 'embedding']

    if clustering_results_df.loc[best_k_means_index, 'umap_n_components'] == 2:
        plot_k_means(best_k_means, best_k_means_embedding)

    return best_k_means, best_k_means_index


def plot_k_means(k_means: KMeans, embedding: np.ndarray) -> None:
    """
    Plots the k-means clustering results, showing clusters and their centroids.
    """

    labels = k_means.labels_
    centroids = k_means.cluster_centers_
    n_clusters = len(np.unique(labels))

    plt.figure(figsize=(8, 8))
    for cluster in range(n_clusters):
        cluster_points = embedding[labels == cluster]

        plt.scatter(
            x=cluster_points[:, 0],
            y=cluster_points[:, 1],
            label=f'Cluster {cluster}',
            s=20
        )

    plt.scatter(
        x=centroids[:, 0],
        y=centroids[:, 1],
        marker='x',
        s=100,
        c='black',
        label='Centroids'
    )

    plt.xlabel('$x_1$', fontsize=14)
    plt.ylabel('$x_2$', rotation=0, fontsize=14)
    plt.title(f'n_clusters={n_clusters}\nK-Means Clustering', fontsize=14)
    plt.grid(True)
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.show()


def select_best_dbscan_from_clustering_results(clustering_results_df: pd.DataFrame) -> (DBSCAN, int):
    """
    Selects the best DBSCAN clustering result based on validity index and generates a plot.
    """

    best_dbscan_index = clustering_results_df[clustering_results_df['algo'] == 'dbscan']['validity_index'].idxmax()
    best_dbscan = clustering_results_df.loc[best_dbscan_index, 'fitted_dbscan']
    best_dbscan_embedding = clustering_results_df.loc[best_dbscan_index, 'embedding']

    if clustering_results_df.loc[best_dbscan_index, 'umap_n_components'] == 2:
        plot_dbscan(best_dbscan, best_dbscan_embedding)

    return best_dbscan, best_dbscan_index


def plot_dbscan(dbscan: DBSCAN, embedding: np.ndarray) -> None:
    """
    Plots the DBSCAN clustering results, highlighting core, border, and noise points.
    """

    core_mask = np.zeros_like(dbscan.labels_, dtype=bool)
    core_mask[dbscan.core_sample_indices_] = True
    anomalies_mask = dbscan.labels_ == -1
    border_mask = ~(core_mask | anomalies_mask)

    cores = dbscan.components_
    anomalies = embedding[anomalies_mask]
    non_cores = embedding[border_mask]

    plt.figure(figsize=(8, 8))
    if cores.shape[0] > 0:
        plt.scatter(x=cores[:, 0], y=cores[:, 1], marker='*', s=20, c=dbscan.labels_[core_mask], label='core')

    if anomalies.shape[0] > 0:
        plt.scatter(x=anomalies[:, 0], y=anomalies[:, 1], c="r", marker="x", s=100, label='anomalies')

    if non_cores.shape[0] > 0:
        plt.scatter(x=non_cores[:, 0], y=non_cores[:, 1], c=dbscan.labels_[border_mask], marker=".", label='border')

    clusters = np.unique(dbscan.labels_)
    n_clusters = len(clusters[clusters != -1])

    plt.xlabel("$x_1$", fontsize=14)
    plt.ylabel("$x_2$", rotation=0, fontsize=14)
    plt.title(f"eps={dbscan.eps:.2f}, min_samples={dbscan.min_samples}, n_clusters={n_clusters}\nDBSCAN Clustering", fontsize=14)
    plt.grid()
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=14)
    plt.show()


def select_latent_manifold(results_df: pd.DataFrame, best_k_means_index: int, best_dbscan_index: int) -> int:
    """
    Selects the best clustering result between k-means and DBSCAN based on clustering metrics.
    """

    if best_k_means_index is None:
        return best_dbscan_index

    if best_dbscan_index is None:
        return best_k_means_index

    return best_k_means_index if results_df.loc[best_k_means_index, 'silhouette_score'] > \
                                 results_df.loc[best_dbscan_index, 'validity_index'] else best_dbscan_index


def get_latent_manifold_details(clustering_results_df: pd.DataFrame, best_index: int) -> pd.DataFrame:
    """
    Retrieves details of the selected clustering result for further analysis or reporting.
    """

    try:
        silhouette_score = clustering_results_df.loc[best_index, 'silhouette_score']
    except KeyError:
        silhouette_score = None

    try:
        validity_index = clustering_results_df.loc[best_index, 'validity_index']
    except KeyError:
        validity_index = None

    return pd.DataFrame([{
        'number of classes in data set': 4,
        'UMAP n_components': clustering_results_df.loc[best_index, 'umap_n_components'],
        'UMAP min_dist': clustering_results_df.loc[best_index, 'umap_min_dist'],
        'UMAP n_neighbors': clustering_results_df.loc[best_index, 'umap_n_neighbors'],
        'UMAP metric': clustering_results_df.loc[best_index, 'umap_metric'],
        'trustworthiness': clustering_results_df.loc[best_index, 'trustworthiness'],
        'clustering algorithm': clustering_results_df.loc[best_index, 'algo'],
        'number of clusters found': clustering_results_df.loc[best_index, 'n_clusters_found'],
        'validity index or silhouette score': validity_index if clustering_results_df.loc[best_index, 'algo'] == 'dbscan' else silhouette_score,
    }])

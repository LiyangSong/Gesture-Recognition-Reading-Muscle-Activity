import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import cm
from sklearn.metrics import adjusted_rand_score, rand_score, fowlkes_mallows_score, normalized_mutual_info_score, \
    jaccard_score, f1_score, silhouette_samples
from sklearn.metrics.cluster import contingency_matrix

from utils.clustering import perform_k_means_and_get_internal_indices, get_randomly_distributed_data, get_k_means_best_index


def clustering_randomly_distributed_data(
        clustering_results_df: pd.DataFrame,
        afd_check_indices: list[int],
        random_data_sets_num: int,
        random_state: int = 42) -> pd.DataFrame:
    """
    Select several models from pipeline optimization results, and compare their internal indices
    with randomly distributed data sets for the purpose of false discovery detection.
    """

    afd_results_list = []
    for index in afd_check_indices:
        print('\n')
        print('*' * 60)
        print('*' * 60)

        print(f'Getting indices for clustering result {index}')

        data_set_name = f'clustering_result_{index}'
        embedding = clustering_results_df.loc[index, 'embedding']
        algo = clustering_results_df.loc[index, 'algo']

        # Add one actual clustering result
        afd_results_list.append({
            'data_set_name': data_set_name,
            'data_set_type': 'actual',
            'n_clusters': clustering_results_df.loc[index, 'n_clusters_found'],
            'silhouette_score_or_validity_index':
                clustering_results_df.loc[index, 'silhouette_score']
                if algo == 'k_means' else clustering_results_df.loc[index, 'validity_index']
        })

        # Add several random clustering results
        # As validity index can not be calculated with entirely random data sets,
        # only K-means will be applied to them
        for j in range(random_data_sets_num):
            print('\n')
            print('*' * 60)
            print(f'Generating randomly distributed data set {j} for clustering result {index} and getting indices')

            randomly_distributed_data = get_randomly_distributed_data(
                cap_x=embedding,
                random_state=random_state + index*random_data_sets_num + j
            )
            randomly_distributed_data_copy = randomly_distributed_data.copy()

            k_means_internal_indices_df = perform_k_means_and_get_internal_indices(randomly_distributed_data_copy)
            best_index_results_dict = get_k_means_best_index(k_means_internal_indices_df)
            index_found = best_index_results_dict['index_found']
            if index_found is None:
                index_found = best_index_results_dict['n_clusters_elbow']
                print('Would directly use n_clusters at the elbow of inertia curve')
            else:
                index_found = index_found.values[0]

            afd_results_list.append({
                'data_set_name': data_set_name,
                'data_set_type': 'random',
                'n_clusters': k_means_internal_indices_df.loc[index_found, 'n_clusters'],
                'silhouette_score_or_validity_index': k_means_internal_indices_df.loc[index_found, 'silhouette_score']
            })

    return pd.DataFrame(afd_results_list)


def plot_afd_results(afd_results_df: pd.DataFrame) -> None:
    """
    Create histogram plots of internal indices for true data sets and random data sets for comparison.
    """

    min_silhouette_scoreor_validity_index = afd_results_df['silhouette_score_or_validity_index'].min() - 0.1
    for data_set_name in afd_results_df['data_set_name'].unique():
        print('\n')
        print('*' * 60)
        print(data_set_name)

        # plot the results for significance testing
        temp_results_df = afd_results_df.loc[afd_results_df['data_set_name'] == data_set_name, :]

        plt.figure(figsize=(8, 8))
        sns.histplot(
            data=temp_results_df,
            x='silhouette_score_or_validity_index',
            hue='data_set_type',
            binwidth=0.02
        )

        plt.xlim([min_silhouette_scoreor_validity_index, 1])
        plt.xlabel('silhouette_score_or_validity_index', fontsize=14)
        plt.ylabel('Data Sets Count', fontsize=14)
        plt.title(f'Significance Testing Plot for {data_set_name}', fontsize=14)
        plt.show()


def get_external_indices(merged_df: pd.DataFrame, algo: str) -> pd.DataFrame:
    """
    Compute external indices by comparing clustering solution to the actual data object classes.
    """

    labels_true = np.array(merged_df['target'])
    labels_pred = np.array(merged_df['pred_labels'])

    non_noise_indices = None

    if algo == 'dbscan':
        return_dict = remove_noise_data_objects_from_labels(labels_true, labels_pred)
        labels_true = return_dict['labels_true']
        labels_pred = return_dict['labels_pred']
        non_noise_indices = return_dict['non_noise_indices']

    rand_score_ = rand_score(labels_true, labels_pred)
    print('rand_score: ', rand_score_)

    adjusted_rand_score_ = adjusted_rand_score(labels_true, labels_pred)
    print('adjusted_rand_score: ', adjusted_rand_score_)

    fowlkes_mallows_score_ = fowlkes_mallows_score(labels_true, labels_pred)
    print('fowlkes_mallows_score: ', fowlkes_mallows_score_)

    normalized_mutual_info_score_ = normalized_mutual_info_score(labels_true, labels_pred)
    print('normalized_mutual_info_score: ', normalized_mutual_info_score_)

    jaccard_score_ = jaccard_score(labels_true, labels_pred, average=None)
    print('jaccard_score: ', jaccard_score_)

    f1_score_ = f1_score(labels_true, labels_pred, average=None)
    print('f1_score: ', f1_score_)

    best_label_perm = get_approx_best_cluster_label_permutation(labels_true, labels_pred)
    best_contingency_matrix = best_label_perm['best_contingency_matrix']
    print('contingency_matrix:\n', best_contingency_matrix)

    purity_score = np.sum(np.amax(best_contingency_matrix, axis=0)) / np.sum(best_contingency_matrix)
    print('purity_score: ', purity_score)

    return pd.DataFrame([{
        'rand_score': rand_score_,
        'adjusted_rand_score': adjusted_rand_score_,
        'fowlkes_mallows_score': fowlkes_mallows_score_,
        'normalized_mutual_info_score': normalized_mutual_info_score_,
        'jaccard_score': jaccard_score_,
        'f1_score': f1_score_,
        'purity_score': purity_score,
        'best_perm_labels_pred': best_label_perm['best_perm_labels_pred'],
        'best_contingency_matrix': best_contingency_matrix,
        'non_noise_indices': non_noise_indices
    }])


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

        if np.max(initial_matrix) == 0:
            break

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

    # Handle unmapped predicted labels
    unique_unmapped_pred_labels = np.setdiff1d(pred_labels, list(best_permutation_mapping.keys()))
    unmapped_label_start = -2  # Start assigning unique labels from -2
    for unmapped_pred_label in unique_unmapped_pred_labels:
        best_permutation_mapping[unmapped_pred_label] = unmapped_label_start
        best_perm_labels_pred[labels_pred == unmapped_pred_label] = unmapped_label_start
        unmapped_label_start -= 1

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


def plot_score_hist(clustering_results_df: pd.DataFrame) -> None:

    clustering_results_df['silhouette_score_or_validity_index'] = clustering_results_df.apply(
        lambda x: x['silhouette_score'] if x['algo'] == 'k_means' else x['validity_index'], axis=1)

    plt.figure(figsize=(8, 8))
    sns.histplot(
        data=clustering_results_df,
        x='silhouette_score_or_validity_index',
        hue='algo',
        bins=30
    )

    plt.xlabel('silhouette_score_or_validity_index', fontsize=14)
    plt.ylabel('Data Sets Count', fontsize=14)
    plt.title(f'Distribution of silhouette_score or validity_index', fontsize=14)
    plt.show()


def plot_silhouette_score_analysis(clustering_results_df: pd.DataFrame, n_clusters_list: list) -> None:

    for n_clusters in n_clusters_list:
        print('\n')
        print('*' * 60)

        best_index = clustering_results_df.loc[clustering_results_df['n_clusters_found'] == n_clusters, 'silhouette_score'].idxmax()
        print(f'best_index for n_clusters: {n_clusters} is {best_index}')

        silhouette_avg = clustering_results_df.loc[best_index, 'silhouette_score']
        print(f'the average silhouette_score is {silhouette_avg}')

        # Compute the silhouette scores for each sample
        embedding = clustering_results_df.loc[best_index, 'embedding']
        cluster_labels = clustering_results_df.loc[best_index, 'cluster_labels']

        sample_silhouette_values = silhouette_samples(embedding, cluster_labels)

        fig, ax = plt.subplots(figsize=(8, 7))
        ax.set_ylim([0, len(embedding) + (n_clusters + 1) * 10])
        y_lower = 10

        for i in range(n_clusters):
            # Aggregate the silhouette scores for samples belonging to
            # cluster i, and sort them
            ith_cluster_silhouette_values = sample_silhouette_values[cluster_labels == i]

            ith_cluster_silhouette_values.sort()

            size_cluster_i = ith_cluster_silhouette_values.shape[0]
            y_upper = y_lower + size_cluster_i

            color = cm.get_cmap("Spectral")(float(i) / n_clusters)
            ax.fill_betweenx(
                np.arange(y_lower, y_upper),
                0,
                ith_cluster_silhouette_values,
                facecolor=color,
                edgecolor=color,
                alpha=0.7,
            )

            # Label the silhouette plots with their cluster numbers at the middle
            ax.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))

            # Compute the new y_lower for next plot
            y_lower = y_upper + 10  # 10 for the 0 samples

        ax.set_title(f"The silhouette plot for the various clusters\nn_clusters={n_clusters} ", fontsize=14)
        ax.set_xlabel("The silhouette coefficient values", fontsize=14)
        ax.set_ylabel("Cluster label", fontsize=14)

        # The vertical line for average silhouette score of all the values
        ax.axvline(x=silhouette_avg, color="red", linestyle="--")
        ax.set_yticks([])

        plt.show()


def plot_contingency_matrix(matrix) -> None:
    plt.figure(figsize=(8, 8))
    sns.heatmap(
        data=matrix,
        annot=True,
        fmt='d',
        linewidths=3,
        cmap='rocket_r'
    )
    plt.xlabel("Clusters", fontsize=14)
    plt.ylabel("Classes", fontsize=14)
    plt.title("Contingency Matrix", fontsize=14)
    plt.show()

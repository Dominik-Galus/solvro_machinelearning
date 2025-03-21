import numpy as np
import pandas as pd
from sklearn.cluster import KMeans  # type: ignore[import-untyped]
from sklearn.preprocessing import StandardScaler  # type: ignore[import-untyped]

from solvro_machinelearning.metrics.elbow_metrics import (
    cluster_and_evaluate,
    elbow_optimizer,
    plot_evaluation,
    reduce_dimensions,
)


def cocktail_cluster_score(
        df: pd.DataFrame,
        n_clusters_list: list[int],
        reduction_methods: list[str] | None = None,
    ) -> list[dict[str, object]]:
    features = extract_features(df)
    standard_scaler = StandardScaler()
    scaled_features = standard_scaler.fit_transform(features)

    results = []

    if reduction_methods:
        for reduction_method in reduction_methods:
            reduced_features = reduce_dimensions(scaled_features, reduction_method)

            inertias = []
            for k in range(1, 11):
                kmeans = KMeans(n_clusters=k, random_state=0).fit(reduced_features)
                inertias.append(kmeans.inertia_)

            silhouette_scores = []
            calinski_scores = []

            for n_clusters in n_clusters_list:
                labels, silhouette, davies_bouldin, calinski_harabasz, inertia = cluster_and_evaluate(
                    reduced_features, n_clusters,
                )

                silhouette_scores.append(silhouette)
                calinski_scores.append(calinski_harabasz)

                results.append({
                    "reduction": reduction_method,
                    "n_clusters": n_clusters,
                    "silhouette": silhouette,
                    "davies_bouldin": davies_bouldin,
                    "calinski_harabasz": calinski_harabasz,
                    "inertia": inertia,
                    "labels": labels,
                    "features": reduced_features,
                })

            title = f"Cocktail dataset with {reduction_method.upper()}"
            elbow_optimizer(inertias, title)

            plot_evaluation(silhouette_scores, calinski_scores, title, n_clusters_list)
    else:
        inertias = []
        for k in range(1, 11):
            kmeans = KMeans(n_clusters=k, random_state=0).fit(scaled_features)
            inertias.append(kmeans.inertia_)

        silhouette_scores = []
        calinski_scores = []

        for n_clusters in n_clusters_list:
            labels, silhouette, davies_bouldin, calinski_harabasz, inertia = cluster_and_evaluate(
                scaled_features, n_clusters,
            )

            silhouette_scores.append(silhouette)
            calinski_scores.append(calinski_harabasz)

            results.append({
                "reduction": "none",
                "n_clusters": n_clusters,
                "silhouette": silhouette,
                "davies_bouldin": davies_bouldin,
                "calinski_harabasz": calinski_harabasz,
                "inertia": inertia,
                "labels": labels,
                "features": scaled_features,
            })

        title = "Cocktail dataset without reduction"
        elbow_optimizer(inertias, title)

        plot_evaluation(silhouette_scores, calinski_scores, title, n_clusters_list)

    return results


def extract_features(df: pd.DataFrame) -> np.ndarray:
    feature_arrays = []

    numerical_features = [
        "ingredient_count", "alcoholic_ingredients",
        "non_alcoholic_ingredients", "alcoholic_ratio",
    ]
    feature_arrays.append(df[numerical_features].values)

    for col in ["category", "glass"]:

        col_array = np.array([np.array(x) for x in df[col].to_numpy()])
        feature_arrays.append(col_array)

    tag_arrays = np.array([np.array(x) for x in df["tags"].to_numpy()])
    tag_sums = np.sum(tag_arrays, axis=0)
    top_indices = np.argsort(tag_sums)

    selected_tags = tag_arrays[:, top_indices]
    feature_arrays.append(selected_tags)

    cluster_proportions = []

    for clusters in df["ingredient_clusters"]:

        counts = [0, 0, 0]
        for cluster in clusters:
            if 0 <= cluster <= 2:  # noqa: PLR2004
                counts[cluster] += 1

        total = sum(counts)
        proportions = [count / total for count in counts] if total > 0 else counts

        cluster_proportions.append(proportions)

    feature_arrays.append(np.array(cluster_proportions))

    return np.hstack(feature_arrays)

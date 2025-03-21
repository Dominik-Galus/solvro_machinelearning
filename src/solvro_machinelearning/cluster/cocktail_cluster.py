import pandas as pd
from sklearn.cluster import KMeans  # type: ignore[import-untyped]
from sklearn.metrics import (  # type: ignore[import-untyped]
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_score,
)
from sklearn.preprocessing import StandardScaler  # type: ignore[import-untyped]

from solvro_machinelearning.metrics.cocktail_cluster_score import extract_features
from solvro_machinelearning.metrics.elbow_metrics import reduce_dimensions


def cocktail_cluster(
        df: pd.DataFrame,
        n_clusters: int,
        reduction_method: str | None = None,
    ) -> tuple[pd.DataFrame, dict[str, object]]:

    df_copy = df.copy()

    features = extract_features(df_copy)

    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)

    features_reduced = reduce_dimensions(features_scaled, reduction_method) if reduction_method else features_scaled

    kmeans = KMeans(n_clusters=n_clusters, random_state=0)
    cluster_labels = kmeans.fit_predict(features_reduced)

    df_copy["cocktail_cluster"] = cluster_labels

    visualization_data = {
        "reduction": reduction_method or "None",
        "n_clusters": n_clusters,
        "labels": cluster_labels,
        "features": features_reduced,
        "silhouette": silhouette_score(features_reduced, cluster_labels),
        "davies_bouldin": davies_bouldin_score(features_reduced, cluster_labels),
        "calinski_harabasz": calinski_harabasz_score(features_reduced, cluster_labels),
    }

    return df_copy, visualization_data

from pathlib import Path

import numpy as np
import torch
from PIL import Image
from sklearn.cluster import KMeans  # type: ignore[import-untyped]
from sklearn.metrics import (  # type: ignore[import-untyped]
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_score,
)
from sklearn.preprocessing import StandardScaler  # type: ignore[import-untyped]
from transformers import (  # type: ignore[import-untyped]
    AutoImageProcessor,
    AutoModelForImageClassification,
    CLIPModel,
    CLIPProcessor,
)

from solvro_machinelearning.metrics.elbow_metrics import elbow_optimizer, plot_evaluation, reduce_dimensions


def ingredients_image_cluster_score(  # noqa: PLR0914
        models: dict[str, tuple],
        ingredients_path: str,
        n_clusters_list: list[int],
        reduction_methods: list[str] | None = None,
    ) -> list[dict[str, object]]:
    ingredients_image_dir = Path(ingredients_path)
    image_files = list(ingredients_image_dir.iterdir())
    features_dict = {}
    standard_scaler = StandardScaler()

    for model_name, (model, processor) in models.items():
        model.eval()
        features_dict[model_name] = standard_scaler.fit_transform(_extract_features(model, processor, image_files))

    results = []

    for model_name, features in features_dict.items():
        if reduction_methods:
            for reduction_method in reduction_methods:
                reduced_features = reduce_dimensions(features, reduction_method)

                inertias = []
                for k in range(1, 11):
                    kmeans = KMeans(n_clusters=k, random_state=0).fit(reduced_features)
                    inertias.append(kmeans.inertia_)

                silhouette_scores = []
                calinski_scores = []

                for n_clusters in n_clusters_list:
                    labels, silhouette, davies_bouldin, calinski_harabasz, inertia = _cluster_and_evaluate(
                        reduced_features, n_clusters,
                    )

                    silhouette_scores.append(silhouette)
                    calinski_scores.append(calinski_harabasz)

                    results.append({
                        "model": model_name,
                        "reduction": reduction_method,
                        "n_clusters": n_clusters,
                        "silhouette": silhouette,
                        "davies_bouldin": davies_bouldin,
                        "calinski_harabasz": calinski_harabasz,
                        "inertia": inertia,
                        "labels": labels,
                        "features": reduced_features,
                    })

                title = f"{model_name} with {reduction_method.upper()}"
                elbow_optimizer(inertias, title)

                plot_evaluation(silhouette_scores, calinski_scores, title, n_clusters_list)
        else:
            inertias = []
            for k in range(1, 11):
                kmeans = KMeans(n_clusters=k, random_state=0).fit(features)
                inertias.append(kmeans.inertia_)

            silhouette_scores = []
            calinski_scores = []

            for n_clusters in n_clusters_list:
                labels, silhouette, davies_bouldin, calinski_harabasz, inertia = _cluster_and_evaluate(
                    features, n_clusters,
                )

                silhouette_scores.append(silhouette)
                calinski_scores.append(calinski_harabasz)

                results.append({
                    "model": model_name,
                    "reduction": "none",
                    "n_clusters": n_clusters,
                    "silhouette": silhouette,
                    "davies_bouldin": davies_bouldin,
                    "calinski_harabasz": calinski_harabasz,
                    "inertia": inertia,
                    "labels": labels,
                    "features": features,
                })

            title = f"{model_name} without reduction"
            elbow_optimizer(inertias, title)

            plot_evaluation(silhouette_scores, calinski_scores, title, n_clusters_list)

    return results


def _cluster_and_evaluate(features: np.ndarray, n_clusters: int) -> tuple[np.ndarray, float, float, float, float]:
    kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(features)
    labels = kmeans.labels_
    inertia = kmeans.inertia_

    silhouette = silhouette_score(features, labels)
    davies_bouldin = davies_bouldin_score(features, labels)
    calinski_harabasz = calinski_harabasz_score(features, labels)

    return labels, silhouette, davies_bouldin, calinski_harabasz, inertia


def _extract_features(
        model: AutoModelForImageClassification | CLIPModel,
        processor: AutoImageProcessor | CLIPProcessor,
        image_files: list[Path],
    ) -> np.ndarray:
    features = []
    for image_file in image_files:
        image = Image.open(image_file).convert("RGB")
        with torch.no_grad():
            if isinstance(model, CLIPModel):
                inputs = processor(images=image, return_tensors="pt", padding=True)
                outputs = model.get_image_features(**inputs)
            else:
                inputs = processor(image, return_tensors="pt")
                outputs = model(**inputs).logits
        features.append(outputs.cpu().numpy())
    return np.vstack(features)

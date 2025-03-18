from pathlib import Path

import numpy as np
import torch
from PIL import Image
from sklearn.cluster import KMeans  # type: ignore[import-untyped]
from sklearn.decomposition import PCA, KernelPCA, TruncatedSVD  # type: ignore[import-untyped]
from sklearn.manifold import TSNE, Isomap, LocallyLinearEmbedding  # type: ignore[import-untyped]
from sklearn.metrics import (  # type: ignore[import-untyped]
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_score,
)
from transformers import (  # type: ignore[import-untyped]
    AutoImageProcessor,
    AutoModelForImageClassification,
    CLIPModel,
    CLIPProcessor,
)


def ingredients_cluster_score(
        models: dict[str, tuple[AutoModelForImageClassification | CLIPModel, AutoImageProcessor | CLIPProcessor]],
        ingredients_path: str,
    ) -> list[dict[str, object]]:
    ingredients_image_dir = Path(ingredients_path)
    image_files = list(ingredients_image_dir.iterdir())
    features_dict = {}
    for model_name, (model, processor) in models.items():
        model.eval()
        features_dict[model_name] = _extract_features(model, processor, image_files)

    results = []
    reduction_methods = ["pca", "tsne", "truncated_svd", "isomap", "kernel_pca"]
    for model_name, features in features_dict.items():
        for reduction_method in reduction_methods:
            reduced_features = _reduce_dimensions(features, reduction_method)
            for n_clusters in [2, 3, 4, 5, 6, 7, 8, 9]:
                labels, silhouette, davies_bouldin, calinski_harabasz = _cluster_and_evaluate(
                    reduced_features, n_clusters,
                )

                results.append({
                    "model": model_name,
                    "reduction": reduction_method,
                    "n_clusters": n_clusters,
                    "silhouette": silhouette,
                    "davies_bouldin": davies_bouldin,
                    "calinski_harabasz": calinski_harabasz,
                    "labels": labels,
                    "features": reduced_features,
                })

    return results


def _cluster_and_evaluate(features: np.ndarray, n_clusters: int) -> tuple[str, float, float, float]:
    kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(features)
    labels = kmeans.labels_

    silhouette = silhouette_score(features, labels)
    davies_bouldin = davies_bouldin_score(features, labels)
    calinski_harabasz = calinski_harabasz_score(features, labels)

    return labels, silhouette, davies_bouldin, calinski_harabasz


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


def _reduce_dimensions(features: np.ndarray, method: str) -> np.ndarray:
    if method == "pca":
        reducer = PCA(n_components=2, random_state=0)
    elif method == "tsne":
        reducer = TSNE(n_components=2, random_state=0)
    elif method == "truncated_svd":
        reducer = TruncatedSVD(n_components=2, random_state=0)
    elif method == "isomap":
        reducer = Isomap(n_components=2)
    elif method == "lle":
        reducer = LocallyLinearEmbedding(n_components=2, random_state=0)
    elif method == "kernel_pca":
        reducer = KernelPCA(n_components=2, kernel="rbf", random_state=0)
    else:
        msg = f"Unknown reduction method: {method}"
        raise ValueError(msg)
    return reducer.fit_transform(features)  # type: ignore[no-any-return]

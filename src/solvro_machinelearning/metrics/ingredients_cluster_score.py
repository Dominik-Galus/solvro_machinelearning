from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from kneed import KneeLocator  # type: ignore[import-untyped]
from PIL import Image
from sklearn.cluster import KMeans  # type: ignore[import-untyped]
from sklearn.decomposition import PCA, KernelPCA, TruncatedSVD  # type: ignore[import-untyped]
from sklearn.manifold import Isomap  # type: ignore[import-untyped]
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

REDUCTION_METHODS = {
    "pca": PCA(n_components=2, random_state=0),
    "truncated_svd": TruncatedSVD(n_components=2, random_state=0),
    "isomap": Isomap(n_components=2),
    "kernel_pca": KernelPCA(n_components=2, kernel="rbf", random_state=0),
    }


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
                reduced_features = _reduce_dimensions(features, reduction_method)

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


def _reduce_dimensions(features: np.ndarray, method: str) -> np.ndarray:
    reducer = REDUCTION_METHODS.get(method)
    if not reducer:
        msg = f"Unknown reduction method: {method}"
        raise ValueError(msg)
    return reducer.fit_transform(features)  # type: ignore[no-any-return]


def plot_evaluation(sil: list[float], cal: list[float], title: str, x: range | list[int] = range(2, 11)) -> None:  # noqa: B008
    _, ax = plt.subplots(1, 2, figsize=(20, 8), dpi=100)
    ax[0].plot(x, sil, color="#99582a", marker="o", ms=15, mfc="#6f1d1b")
    ax[1].plot(x, cal, color="#99582a", marker="o", ms=15, mfc="#6f1d1b")
    ax[0].set_xlabel("Number of Clusters", labelpad=20)
    ax[0].set_ylabel("Silhouette Coefficient", labelpad=20)
    ax[1].set_xlabel("Number of Clusters", labelpad=20)
    ax[1].set_ylabel("Calinski Harabasz Coefficient", labelpad=20)

    best_sil_idx = np.argmax(sil)
    best_cal_idx = np.argmax(cal)

    ax[0].annotate(f"Best: {x[best_sil_idx]} clusters",
                  xy=(x[best_sil_idx], sil[best_sil_idx]),
                  xytext=(x[best_sil_idx], sil[best_sil_idx] * 0.9),
                  arrowprops={"facecolor": "#6f1d1b", "shrink": 0.05},
                  ha="center")

    ax[1].annotate(f"Best: {x[best_cal_idx]} clusters",
                  xy=(x[best_cal_idx], cal[best_cal_idx]),
                  xytext=(x[best_cal_idx], cal[best_cal_idx] * 0.9),
                  arrowprops={"facecolor": "#6f1d1b", "shrink": 0.05},
                  ha="center")

    plt.suptitle(f"Evaluate {title} Clustering", y=0.92)
    plt.tight_layout(pad=3)
    plt.show()


def elbow_optimizer(inertias: list[float], title: str) -> None:
    plt.figure(figsize=(10, 6))
    kl = KneeLocator(range(1, 11), inertias, curve="convex", direction="decreasing")
    plt.style.use("fivethirtyeight")
    sns.lineplot(x=range(1, 11), y=inertias, color="#99582a")
    plt.xticks(range(1, 11))
    plt.xlabel("Number of Clusters", labelpad=20)
    plt.ylabel("Inertia", labelpad=20)
    plt.title(f"Elbow Method for {title}", y=1)

    if kl.elbow is not None:
        plt.axvline(x=kl.elbow, color="#6f1d1b", label=f"Optimal clusters: {kl.elbow}", ls="--")
        plt.legend()

    plt.tight_layout()
    plt.show()


def apply_clustering(
        df: pd.DataFrame,
        ingredients_path: str,
        model_and_processor: tuple[AutoModelForImageClassification | CLIPModel, AutoImageProcessor | CLIPProcessor],
        n_clusters: int,
        reduction_method: str | None,
    ) -> pd.DataFrame:
    model, processor = model_and_processor
    model.eval()

    df_copy = df.copy()

    all_ingredients: set[str] = set()
    for _, row in df_copy.iterrows():
        all_ingredients.update(ingredient["name"] for ingredient in row["ingredients"])

    ingredients_image_dir = Path(ingredients_path)
    image_files = list(ingredients_image_dir.iterdir())

    image_path_map = {}
    for file in image_files:
        base_name = file.stem.lower()
        image_path_map[base_name] = file
        image_path_map[base_name.replace("_", "").replace("-", "")] = file

    ingredient_features = _get_features(all_ingredients, image_path_map, model, processor)

    if not ingredient_features:
        for idx, row in df_copy.iterrows():
            df_copy.at[idx, "ingredients"] = [  # noqa: PD008
                {k: v for k, v in ing.items() if k != "imageUrl"}
                for ing in row["ingredients"]
            ]
        return df_copy

    ingredient_to_cluster = _cluster(ingredient_features, reduction_method, n_clusters)

    for idx, row in df_copy.iterrows():
        updated_ingredients = []
        for ingredient in row["ingredients"]:
            updated_ing = {k: v for k, v in ingredient.items() if k != "imageUrl"}

            ing_name = ingredient["name"]
            updated_ing["cluster"] = ingredient_to_cluster.get(ing_name, -1)

            updated_ingredients.append(updated_ing)

        df_copy.at[idx, "ingredients"] = updated_ingredients  # noqa: PD008

    return df_copy


def _get_features(
        all_ingredients: set[str],
        image_path_map: dict[str, Path],
        model: AutoModelForImageClassification | CLIPModel,
        processor: AutoImageProcessor | CLIPProcessor,
    ) -> dict[str, list[float]]:
    ingredient_features = {}
    for ingredient in all_ingredients:
        normalized_names = [
            ingredient.lower(),
            ingredient.lower().replace(" ", "_"),
            ingredient.lower().replace(" ", "-"),
            ingredient.lower().replace(" ", ""),
        ]

        for norm_name in normalized_names:
            if norm_name in image_path_map:
                img_path = image_path_map[norm_name]

                image = Image.open(img_path).convert("RGB")
                with torch.no_grad():
                    if isinstance(model, CLIPModel):
                        inputs = processor(images=image, return_tensors="pt", padding=True)
                        outputs = model.get_image_features(**inputs)
                    else:
                        inputs = processor(image, return_tensors="pt")
                        outputs = model(**inputs).logits

                ingredient_features[ingredient] = outputs.cpu().numpy().flatten()
                break

    return ingredient_features


def _cluster(
        ingredient_features: dict[str, list[float]],
        reduction_method: str | None, n_clusters: int,
    ) -> dict[str, int]:
    ingredients_with_features = list(ingredient_features.keys())
    features_array = np.vstack([ingredient_features[ing] for ing in ingredients_with_features])

    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features_array)

    if reduction_method:
        reducer = REDUCTION_METHODS.get(reduction_method)
        if reducer is None:
            msg = "There is no such reducer"
            raise ValueError(msg)

        features_reduced = reducer.fit_transform(features_scaled)
    else:
        features_reduced = features_scaled

    kmeans = KMeans(n_clusters=n_clusters, random_state=0)
    cluster_labels = kmeans.fit_predict(features_reduced)

    return {ing: int(label) for ing, label in zip(
        ingredients_with_features,
        cluster_labels,
        strict=False,
        )
    }

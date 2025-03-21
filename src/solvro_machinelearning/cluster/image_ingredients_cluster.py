from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.cluster import KMeans  # type: ignore[import-untyped]
from sklearn.preprocessing import StandardScaler  # type: ignore[import-untyped]
from transformers import (  # type: ignore[import-untyped]
    AutoImageProcessor,
    AutoModelForImageClassification,
    CLIPModel,
    CLIPProcessor,
)

from solvro_machinelearning.config.reducers import REDUCTION_METHODS


def image_ingredients_cluster(
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

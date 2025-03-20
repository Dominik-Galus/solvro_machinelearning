from io import BytesIO
from pathlib import Path

import pandas as pd
import requests
from PIL import Image


def extract_ingredients_features(df: pd.DataFrame) -> pd.DataFrame:
    ingredient_counts = []
    alcohol_ingredients = []
    non_alcohol_ingredients = []
    ingredient_types = []
    ingredient_names = []
    ingredient_urls = []

    for _, row in df.iterrows():
        ingredients = row["ingredients"]

        ingredient_counts.append(len(ingredients))

        alcoholic_count = sum(1 for ing in ingredients if ing.get("alcohol") == 1)
        alcohol_ingredients.append(alcoholic_count)

        non_alcoholic_count = sum(1 for ing in ingredients if ing.get("alcohol") == 0)
        non_alcohol_ingredients.append(non_alcoholic_count)

        types = [ing.get("type") for ing in ingredients if ing.get("type") is not None]
        ingredient_types.append(types)

        names = [ing.get("name") for ing in ingredients if ing.get("name") is not None]
        ingredient_names.append(names)

        url = [ing.get("imageUrl") for ing in ingredients if ing.get("imageUrl") is not None]
        ingredient_urls.append(url)

    features_df = pd.DataFrame({
        "ingredient_count": ingredient_counts,
        "alcoholic_ingredients": alcohol_ingredients,
        "non_alcoholic_ingredients": non_alcohol_ingredients,
        "ingredient_types": ingredient_types,
        "ingredient_names": ingredient_names,
        "imageUrl": ingredient_urls,
    })

    features_df["alcoholic_ratio"] = features_df["alcoholic_ingredients"] / features_df["ingredient_count"]

    return features_df


def get_null_values(ingredients_list: pd.Series) -> list[tuple[str, str]]:
    null_values = []
    for ingredients in ingredients_list:
        for ingredient in ingredients:
            for key, value in ingredient.items():
                if value is None:
                    null_values.append((ingredient["name"], key))
    return null_values


def count_null_values(ingredients_list: pd.Series) -> dict[str, int]:
    null_counts = {}
    for ingredients in ingredients_list:
        for ingredient in ingredients:
            for key, value in ingredient.items():
                if value is None:
                    if key not in null_counts:
                        null_counts[key] = 0
                    null_counts[key] += 1
    return null_counts


def download_ingredients_images(directory_path: str, images: pd.DataFrame) -> None:
    for _, row in images.iterrows():
        name, url = row["name"], row["imageUrl"]

        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status()

            img = Image.open(BytesIO(response.content))
            file_path = Path(directory_path) / Path(f"{name}.png")
            img.save(file_path)

        except requests.exceptions.RequestException as e:
            msg: str = "Error occured"
            raise requests.HTTPError(msg) from e

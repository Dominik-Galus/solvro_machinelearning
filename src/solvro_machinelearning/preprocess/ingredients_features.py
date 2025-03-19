import pandas as pd


def extract_ingredients_features(df: pd.DataFrame) -> pd.DataFrame:
    ingredient_counts = []
    alcohol_ingredients = []
    non_alcohol_ingredients = []
    ingredient_types = []
    ingredient_names = []

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

    features_df = pd.DataFrame({
        "ingredient_count": ingredient_counts,
        "alcoholic_ingredients": alcohol_ingredients,
        "non_alcoholic_ingredients": non_alcohol_ingredients,
        "ingredient_types": ingredient_types,
        "ingredient_names": ingredient_names,
    })

    features_df["alcoholic_ratio"] = features_df["alcoholic_ingredients"] / features_df["ingredient_count"]

    return features_df

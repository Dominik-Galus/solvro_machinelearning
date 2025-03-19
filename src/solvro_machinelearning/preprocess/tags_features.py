import operator

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors


def fill_null_tags(df: pd.DataFrame) -> pd.DataFrame:  # noqa: PLR0914
    df_with_tags = df[df["tags"].apply(lambda x: isinstance(x, list) and len(x) > 0)].copy()
    df_missing_tags = df[df["tags"].apply(lambda x: not isinstance(x, list) or len(x) == 0)].copy()
    features = []

    for _, row in df.iterrows():
        ingredient_names = [ing["name"].lower() for ing in row["ingredients"]]
        features.append(" ".join(ingredient_names))

    for i, row in enumerate(df.itertuples()):
        features[i] += " " + row.instructions

    for i, row in enumerate(df.itertuples()):
        features[i] += " " + str(row.category) + " " + str(row.glass)

    vectorizer = TfidfVectorizer(stop_words="english", min_df=2, max_features=500)
    x_all = vectorizer.fit_transform(features)

    x_with_tags = x_all[df.index.isin(df_with_tags.index)]
    x_missing_tags = x_all[df.index.isin(df_missing_tags.index)]

    knn = NearestNeighbors(n_neighbors=5, metric="cosine")
    knn.fit(x_with_tags)

    _, indices = knn.kneighbors(x_missing_tags)

    index_to_position = {idx: i for i, idx in enumerate(df_with_tags.index)}

    imputed_tags = []
    for neighbor_indices in indices:
        neighbor_rows = [df_with_tags.iloc[index_to_position[idx]] for idx in df_with_tags.index[neighbor_indices]]

        tag_counts = {}
        for row in neighbor_rows:
            for tag in row["tags"]:
                tag_counts[tag] = tag_counts.get(tag, 0) + 1

        common_tags = [tag for tag, count in tag_counts.items() if count >= 2]  # noqa: PLR2004

        if not common_tags:
            common_tags = sorted(tag_counts.items(), key=operator.itemgetter(1), reverse=True)[:3]
            common_tags = [tag for tag, _ in common_tags]

        imputed_tags.append(common_tags)

    df_copy = df.copy()
    for i, idx in enumerate(df_missing_tags.index):
        df_copy.at[idx, "tags"] = imputed_tags[i]  # noqa: PD008

    return df_copy


def extract_tags_features(df: pd.DataFrame) -> pd.DataFrame:
    df["tags"] = df["tags"].apply(lambda x: [] if x is None else x)

    all_tags = set()
    for tags in df["tags"]:
        all_tags.update(tags)

    tag_features = pd.DataFrame(index=df.index)
    for tag in all_tags:
        tag_features[f"tag_{tag}"] = df["tags"].apply(lambda x, tag=tag: 1 if tag in x else 0)

    return tag_features

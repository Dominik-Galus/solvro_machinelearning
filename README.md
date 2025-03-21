# Klastrowanie koktajli

Instalowanie z uzyciem pip:
```bash
pip install .
```

Cała analiza zbioru, czyszczenie i klastrowanie zbioru zostało zwizualizowane w notebooks/data_analysis_and_clustering.ipynb

```
src
└── solvro_machinelearning
    ├── __init__.py
    ├── cluster
    │   ├── __init__.py
    │   ├── cocktail_cluster.py
    │   ├── image_ingredients_cluster.py
    │   └── py.typed
    ├── config
    │   ├── __init__.py
    │   ├── py.typed
    │   └── reducers.py
    ├── metrics
    │   ├── __init__.py
    │   ├── cocktail_cluster_score.py
    │   ├── elbow_metrics.py
    │   ├── ingredients_cluster_score.py
    │   └── py.typed
    ├── preprocess
    │   ├── __init__.py
    │   ├── column_one_hot.py
    │   ├── ingredients_features.py
    │   ├── py.typed
    │   └── tags_features.py
    └── py.typed
```
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from kneed import KneeLocator  # type: ignore[import-untyped]
from sklearn.cluster import KMeans  # type: ignore[import-untyped]
from sklearn.metrics import (  # type: ignore[import-untyped]
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_score,
)

from solvro_machinelearning.config.reducers import REDUCTION_METHODS


def reduce_dimensions(features: np.ndarray, method: str) -> np.ndarray:
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
    kl = KneeLocator(range(1, len(inertias) + 1), inertias, curve="convex", direction="decreasing")
    plt.style.use("fivethirtyeight")
    sns.lineplot(x=range(1, len(inertias) + 1), y=inertias, color="#99582a")
    plt.xticks(range(1, 11))
    plt.xlabel("Number of Clusters", labelpad=20)
    plt.ylabel("Inertia", labelpad=20)
    plt.title(f"Elbow Method for {title}", y=1)

    if kl.elbow is not None:
        plt.axvline(x=kl.elbow, color="#6f1d1b", label=f"Optimal clusters: {kl.elbow}", ls="--")
        plt.legend()

    plt.tight_layout()
    plt.show()


def cluster_and_evaluate(features: np.ndarray, n_clusters: int) -> tuple[np.ndarray, float, float, float, float]:
    kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(features)
    labels = kmeans.labels_
    inertia = kmeans.inertia_

    silhouette = silhouette_score(features, labels)
    davies_bouldin = davies_bouldin_score(features, labels)
    calinski_harabasz = calinski_harabasz_score(features, labels)

    return labels, silhouette, davies_bouldin, calinski_harabasz, inertia
